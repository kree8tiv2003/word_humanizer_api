#!/usr/bin/env python3
"""Build the "God Inside" ComfyUI workflows.

Inputs
  source/ltx2_audio_to_video_original.json   the uploaded LTX-2 audio-to-video workflow
  scenes.json                                the 27-scene script (timings + prompts)

Outputs (under comfyui_music_video/)
  workflows/01_ltx2_audio_to_video_FIXED.json      the original workflow, repaired
  workflows/02_god_inside_scene_stills.json        27 character-consistent start frames (Qwen-Image-Edit-2509)
  workflows/03_god_inside_music_video_ltx2.json    27 lip-synced LTX-2 clips, one per scene
  api/stills/scene_XX.json, api/clips/clip_XX.json API-format prompts used by run_music_video.py

Run:  python3 tools/build_workflows.py
"""
import copy
import json
import os
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "source", "ltx2_audio_to_video_original.json")
SCENES = json.load(open(os.path.join(ROOT, "scenes.json")))

MAIN_SG = "3458504c-2bd1-4267-8722-6f2478115b42"   # "Audio to Video(LTX-2.0)"
BASIC_SG = "4944b5c3-502f-45bd-ba28-f90941e8055f"  # "Basic Sampling"

WIDTH, HEIGHT, FPS = 1280, 704, 24.0                # 16:9, divisible by 32, under the 1600x900 limit
AUDIO_FILE = "God_Inside.mp3"
REF_CLOSEUP = "god_inside_ref_closeup.png"
REF_SHEET = "god_inside_ref_sheet.png"

# Qwen-Image-Edit-2509 (core ComfyUI nodes) for the start frames
QWEN_UNET = "qwen_image_edit_2509_fp8_e4m3fn.safetensors"
QWEN_LORA = "Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors"
QWEN_STEPS = 8
QWEN_CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
QWEN_VAE = "qwen_image_vae.safetensors"

# Hand-quality settings (hands are only ~2 latent pixels wide at stage-1 size, so detail comes from
# a 2x latent upscale + short refine pass, calmer motion, and less aggressive NAG)
LTX_NEGATIVE = ("still image with no motion, subtitles, text, scene change, distorted hands, extra fingers, "
                "fused fingers, missing fingers, morphing hands, melting hands, fast arm movement, motion blur")
NAG_SCALE = 6
IMG_COMPRESSION = 15
STAGE1_STEPS, STAGE1_TERMINAL = 10, 0.05
DECODE_TILE, DECODE_OVERLAP = 1536, 128
LATENT_UPSCALER = "ltx-2-spatial-upscaler-x2-1.0.safetensors"
STAGE2_SIGMAS = "0.909375, 0.725, 0.421875, 0.0"  # LTX-2 distilled stage-2 schedule

UE = {"widget_ue_connectable": {}, "version": "7.8", "input_ue_unconnectable": {}}
SKIP_WIDGETS = {"control_after_generate", "upload", "videopreview"}
VIRTUAL = {"Reroute", "MarkdownNote", "Note", "PrimitiveNode"}


def secs(t):
    m, s = t.split(":")
    return int(m) * 60 + float(s)


def scene_dur(sc):
    return int(round(secs(sc["end"]) - secs(sc["start"])))


def still_name(i):
    return f"god_inside_scene_{i:02d}.png"


# ---------------------------------------------------------------------------
# small graph-building helpers (root graphs use list links, subgraphs use dicts)
# ---------------------------------------------------------------------------
class Graph:
    def __init__(self, first_node=1, first_link=1):
        self.nodes, self.links = [], []
        self.nid, self.lid = first_node, first_link

    def node(self, ntype, pos, widgets=None, named=None, inputs=(), outputs=(), title=None,
             size=(320, 100), mode=0, color=None):
        n = {"id": self.nid, "type": ntype, "pos": list(pos), "size": list(size), "flags": {},
             "order": self.nid, "mode": mode,
             "inputs": [dict(i, link=None) for i in inputs],
             "outputs": [dict(o, links=[]) for o in outputs],
             "properties": {"Node name for S&R": ntype, "ue_properties": UE},
             "widgets_values": widgets if widgets is not None else [],
             "widgets_values_named": named or {}}
        if title:
            n["title"] = title
        if color:
            n["color"], n["bgcolor"] = color
        self.nid += 1
        self.nodes.append(n)
        return n

    def link(self, a, aslot, b, bslot):
        typ = a["outputs"][aslot]["type"]
        self.links.append([self.lid, a["id"], aslot, b["id"], bslot, typ])
        a["outputs"][aslot]["links"].append(self.lid)
        b["inputs"][bslot]["link"] = self.lid
        self.lid += 1


def I(name, typ, widget=False, shape=None):
    d = {"name": name, "localized_name": name, "type": typ}
    if widget:
        d["widget"] = {"name": name}
    if shape is not None:
        d["shape"] = shape
    return d


def O(name, typ):
    return {"name": name, "localized_name": name, "type": typ}


def prim_string(g, pos, value, title, multiline=False):
    t = "PrimitiveStringMultiline" if multiline else "PrimitiveString"
    return g.node(t, pos, [value], {"value": value}, [I("value", "STRING", True)], [O("STRING", "STRING")],
                  title=title, size=(420, 160 if multiline else 60), color=("#232", "#353"))


def prim_int(g, pos, value, title):
    return g.node("PrimitiveInt", pos, [value, "fixed"], {"value": value}, [I("value", "INT", True)],
                  [O("INT", "INT")], title=title, size=(260, 82))


def prim_float(g, pos, value, title):
    return g.node("PrimitiveFloat", pos, [value], {"value": value}, [I("value", "FLOAT", True)],
                  [O("FLOAT", "FLOAT")], title=title, size=(260, 60))


def load_image(g, pos, name, title=None):
    return g.node("LoadImage", pos, [name, "image"], {"image": name}, [],
                  [O("IMAGE", "IMAGE"), O("MASK", "MASK")], title=title, size=(320, 330))


def load_audio(g, pos, name):
    return g.node("LoadAudio", pos, [name, None, None], {"audio": name}, [], [O("AUDIO", "AUDIO")],
                  size=(340, 136))


def save_video(g, pos, prefix, title=None):
    return g.node("SaveVideo", pos, [prefix, "auto", "auto"],
                  {"filename_prefix": prefix, "format": "auto", "codec": "auto"},
                  [I("video", "VIDEO")], [], title=title, size=(480, 400))


def note(g, pos, text, title, size=(560, 400)):
    return g.node("MarkdownNote", pos, [text], {"text": text}, [], [], title=title, size=size,
                  color=("#222", "#000"))


# ---------------------------------------------------------------------------
# 1) Repair the uploaded workflow
# ---------------------------------------------------------------------------
def sg_by_id(wf, sid):
    return next(s for s in wf["definitions"]["subgraphs"] if s["id"] == sid)


def drop_subgraph_inputs(sg, names):
    """Remove trailing subgraph inputs whose values must live on the inner widget instead.

    The uploaded file declared these inputs on the subgraph but the parent instances never had
    matching slots, so the frontend shifted the parent's widget values (e.g. sampler_name=9) and
    left the inner widgets without a source ("Required input is missing")."""
    keep = [i for i in sg["inputs"] if i["name"] not in names]
    removed = [i for i in sg["inputs"] if i["name"] in names]
    assert sg["inputs"][:len(keep)] == keep, "only trailing inputs can be dropped safely"
    dead = {lid for i in removed for lid in i["linkIds"]}
    sg["inputs"] = keep
    sg["links"] = [l for l in sg["links"] if l["id"] not in dead]
    for n in sg["nodes"]:
        for inp in n.get("inputs", []):
            if inp.get("link") in dead:
                inp["link"] = None


def set_widget(node, name, value):
    named = node["widgets_values_named"]
    named[name] = value
    node["widgets_values"][list(named).index(name)] = value


def apply_hand_fixes(wf):
    """Settings that reduce hand/finger distortion, applied to every LTX subgraph."""
    for s in wf["definitions"]["subgraphs"]:
        for n in s["nodes"]:
            t = n["type"]
            if t == "CLIPTextEncode" and n["id"] == 922:  # negative, fed to NAG (CFG is 1)
                set_widget(n, "text", LTX_NEGATIVE)
            elif t == "LTX2_NAG":
                set_widget(n, "nag_scale", NAG_SCALE)
            elif t == "LTXVPreprocess":
                set_widget(n, "img_compression", IMG_COMPRESSION)
            elif t == "LTXVScheduler":
                set_widget(n, "steps", STAGE1_STEPS)
                set_widget(n, "terminal", STAGE1_TERMINAL)
            elif t == "VAEDecodeTiled":
                set_widget(n, "tile_size", DECODE_TILE)
                set_widget(n, "overlap", DECODE_OVERLAP)


def fix_instance(node, sg):
    names = [i["name"] for i in sg["inputs"]]
    node["inputs"] = [i for i in node["inputs"] if i["name"] in names]
    node["properties"].pop("proxyWidgetErrorQuarantine", None)
    node["widgets_values"], node["widgets_values_named"] = [], {}


def fix_original(wf):
    wf = copy.deepcopy(wf)
    wf["extra"].pop("prompt", None)  # stale embedded API prompt from an unrelated checkpoint
    main = sg_by_id(wf, MAIN_SG)
    nodes = {n["id"]: n for n in main["nodes"]}

    # inner sampler subgraphs: bake seed / sampler / audio offsets into the inner nodes
    drop_subgraph_inputs(sg_by_id(wf, BASIC_SG), {"noise_seed", "sampler_name"})
    for sid in ("fb0656bb-8f7c-4a7b-bfbe-63c3fe6a0c35", "162d4fcc-f41c-4f10-8bda-e47e74c9eba6",
                "37752fe8-8231-473f-9cb8-e8578d78cd60", "4706ea42-7590-47b6-8487-9c450a88460b"):
        drop_subgraph_inputs(sg_by_id(wf, sid), {"start_index", "sampler_name"})
    for nid in (978, 969, 988, 989, 990):
        fix_instance(nodes[nid], sg_by_id(wf, nodes[nid]["type"]))
    for s in wf["definitions"]["subgraphs"]:
        s["name"] = s["name"].replace("VIdeo Extrndion", "Video Extension")
        for o in s["outputs"]:
            o["linkIds"] = sorted(set(o["linkIds"]))

    # bogus frame rates / settings
    nodes[995]["widgets_values"][0] = 24
    nodes[995]["widgets_values_named"]["fps"] = 24
    nodes[923]["widgets_values"][6] = 32  # LTX needs width/height divisible by 32
    nodes[923]["widgets_values_named"]["divisible_by"] = 32
    apply_hand_fixes(wf)
    prompt = ("a woman in a red beaded gown is singing passionately into a microphone on a nightclub stage, "
              "holding the microphone steadily in one hand, her other hand relaxed, gentle natural movement")
    nodes[921]["widgets_values"] = [prompt]
    nodes[921]["widgets_values_named"]["text"] = prompt
    for vid in (1006, 1007, 1008, 1009):  # preview writers (bypassed by default)
        v = nodes[vid]["widgets_values"]
        v.update(frame_rate=24, format="video/h264-mp4", filename_prefix=f"ltx2_a2v/preview_{vid}")
        nodes[vid]["widgets_values_named"] = copy.deepcopy(v)
    # 1009 had no frame_rate input, unlike its three siblings
    lid = wf["last_link_id"] + 1
    nodes[1009]["inputs"].append({"localized_name": "frame_rate", "name": "frame_rate", "type": "FLOAT",
                                  "widget": {"name": "frame_rate"}, "link": lid})
    nodes[1011]["outputs"][0]["links"].append(lid)
    main["links"].append({"id": lid, "origin_id": 1011, "origin_slot": 0, "target_id": 1009,
                          "target_slot": 4, "type": "FLOAT"})

    # root: give the subgraph node every slot its definition declares and drive them with primitives
    root = {n["id"]: n for n in wf["nodes"]}
    inst = root[994]
    inst["properties"].pop("proxyWidgetErrorQuarantine", None)
    inst["widgets_values"], inst["widgets_values_named"] = [], {}
    old = {i["name"]: i for i in inst["inputs"]}
    inst["inputs"] = []
    for d in main["inputs"]:
        e = old.get(d["name"], {"name": d["name"], "type": d["type"], "link": None})
        if d.get("label"):
            e["label"] = d["label"]
        inst["inputs"].append(e)
    root[444]["widgets_values"] = [REF_CLOSEUP, "image"]
    root[444]["widgets_values_named"]["image"] = REF_CLOSEUP
    root[444]["properties"].pop("image", None)
    root[565]["widgets_values"][0] = AUDIO_FILE
    root[565]["widgets_values_named"]["audio"] = AUDIO_FILE
    for sv in (984, 982):
        root[sv]["widgets_values"][0] = "god_inside/ltx2_a2v"
        root[sv]["widgets_values_named"]["filename_prefix"] = "god_inside/ltx2_a2v"

    g = Graph(first_node=wf["last_node_id"] + 1, first_link=lid + 1)
    g.nodes, g.links = wf["nodes"], wf["links"]
    x = -2020
    srcs = {
        "text": prim_string(g, (x, 6260), prompt, "Prompt", multiline=True),
        "width": prim_int(g, (x, 6460), WIDTH, "Width"),
        "height": prim_int(g, (x, 6570), HEIGHT, "Height"),
        "start_time": prim_string(g, (x, 6680), "0:56", "Audio start (m:ss)"),
        "end_time": prim_string(g, (x, 6770), "1:46", "Audio end (m:ss, >= start + 48s)"),
        "value": prim_float(g, (x, 6860), FPS, "FPS"),
    }
    for slot, i in enumerate(inst["inputs"]):
        if i["name"] in srcs:
            g.link(srcs[i["name"]], 0, inst, slot)
    note(g, (-2811, 7260), FIX_NOTES, "What was fixed", size=(740, 520))
    wf["last_node_id"], wf["last_link_id"] = g.nid - 1, g.lid - 1
    for s in wf["definitions"]["subgraphs"]:
        s["state"].update(lastNodeId=wf["last_node_id"], lastLinkId=wf["last_link_id"])
    return wf


FIX_NOTES = """## Fixes applied to the uploaded workflow

1. **Audio to Video (LTX-2.0) node was missing 3 of its 8 inputs** (`text`, `start_time`, `end_time`). The frontend quarantined the prompt, width, height, crop times and FPS (`proxyWidgetErrorQuarantine`). All 8 slots are restored and driven by explicit Primitive nodes, so nothing depends on promoted widgets.
2. **Basic Sampling + 4 Video Extension subgraphs** declared `noise_seed` / `start_index` / `sampler_name` inputs that the parent nodes never had. This shifted the widget values (`sampler_name = 9, 18, 27, 36`, which is not a sampler) and left the inner TrimAudioDuration / KSamplerSelect / RandomNoise inputs with no source. Those inputs were removed and the values now live on the inner nodes (`lcm`, seed 44/42, audio offsets 9/18/27/36 s).
3. `CreateVideo` FPS `24.2421875` changed to 24. The preview VHS writers were set to 7.75 fps GIF; they are now 24 fps H.264. The 4th preview writer was also missing its FPS link.
4. `ImageResizeKJv2 divisible_by` changed from 2 to 32 (LTX latents need multiples of 32). The prompt said "the man"; it now describes the singer. The stale embedded API prompt was removed, along with duplicate output link ids.
5. Defaults: 1280x704 (16:9), song `God_Inside.mp3`, crop 0:56 to 1:46 (Chorus 1 onward, the 48 s the chain needs).

## Hand-distortion fixes
- Negative prompt (fed to NAG, since CFG is 1) now names distorted/extra/fused fingers, morphing hands and fast arm movement. `nag_scale` changed from 11 to 6, so the model is pushed less toward big movements.
- Every LTXVScheduler: 8 to 10 steps, `terminal` 0.1 to 0.05, so the final low-noise steps that resolve fingers aren't cut short.
- `LTXVPreprocess img_compression` 25 to 15. Every VAEDecodeTiled: tile 1024 to 1536 and overlap 64 to 128, so a 1280-wide frame decodes in one tile with no seam through a hand.
- The extension overlap stays at 25 frames on purpose: it equals the 1 s keep window and the 9/18/27/36 s audio offsets, so changing it would break lip-sync.
- This single-take workflow has no upscale/refine stage. For the best hands, use workflow 03, which renders at half size, upscales the latent 2x and refines at full size.
"""


# ---------------------------------------------------------------------------
# 2) "Scene Clip" subgraph: the main LTX-2 graph with a single sampling pass per scene
# ---------------------------------------------------------------------------
def build_scene_clip_sg(fixed):
    main = copy.deepcopy(sg_by_id(fixed, MAIN_SG))
    main["id"] = "9d1c7a52-6f7e-4f0e-9b61-5c3a2f1e0a01"
    main["name"] = "Scene Clip (LTX-2 audio to video)"
    remove = {969, 988, 989, 990, 1006, 1007, 1008, 1009, 1011, 1012, 983, 1002, 857, 965, 963}
    main["nodes"] = [n for n in main["nodes"] if n["id"] not in remove]
    dead = {l["id"] for l in main["links"] if l["origin_id"] in remove or l["target_id"] in remove}
    # the preview output (slot 0) goes away, the final video becomes slot 0
    dead |= {l["id"] for l in main["links"] if l["target_id"] == -20 and l["target_slot"] == 0}
    main["links"] = [l for l in main["links"] if l["id"] not in dead]
    main["outputs"] = [main["outputs"][1]]
    main["outputs"][0]["label"] = "clip"
    for l in main["links"]:
        if l["target_id"] == -20:
            l["target_slot"] = 0
    for n in main["nodes"]:
        for i in n.get("inputs", []):
            if i.get("link") in dead:
                i["link"] = None
        for o in n.get("outputs", []):
            o["links"] = [x for x in (o.get("links") or []) if x not in dead]
    for i in main["inputs"]:
        i["linkIds"] = [x for x in i["linkIds"] if x not in dead]
    nodes = {n["id"]: n for n in main["nodes"]}
    lid = max(l["id"] for l in main["links"]) + 1

    def add_link(o, oslot, t, tslot, typ):
        nonlocal lid
        main["links"].append({"id": lid, "origin_id": o, "origin_slot": oslot, "target_id": t,
                              "target_slot": tslot, "type": typ})
        if o == -10:
            main["inputs"][oslot]["linkIds"].append(lid)
        else:
            nodes[o]["outputs"][oslot]["links"].append(lid)
        nodes[t]["inputs"][tslot]["link"] = lid
        lid += 1

    # new inputs: frames -> EmptyLTXVLatentVideo.length, duration -> audio trim
    main["inputs"] += [
        {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, "frames")), "name": "frames", "type": "INT",
         "linkIds": [], "localized_name": "frames", "pos": [-2366, 6634]},
        {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, "duration")), "name": "duration", "type": "FLOAT",
         "linkIds": [], "localized_name": "duration", "pos": [-2366, 6654]},
    ]
    nodes[855]["inputs"].append({"localized_name": "length", "name": "length", "type": "INT",
                                 "widget": {"name": "length"}, "link": None})
    add_link(-10, 8, 855, 2, "INT")

    # ---- two-stage sampling (replaces the single full-size "Basic Sampling" pass) ----
    def unlink(link_id):
        l = next(x for x in main["links"] if x["id"] == link_id)
        main["links"].remove(l)
        if l["origin_id"] == -10:
            main["inputs"][l["origin_slot"]]["linkIds"].remove(link_id)
        elif l["origin_id"] in nodes:
            nodes[l["origin_id"]]["outputs"][l["origin_slot"]]["links"].remove(link_id)
        if l["target_id"] in nodes:
            nodes[l["target_id"]]["inputs"][l["target_slot"]]["link"] = None

    def drop_node(nid):
        for l in [x for x in main["links"] if nid in (x["origin_id"], x["target_id"])]:
            unlink(l["id"])
        main["nodes"].remove(nodes.pop(nid))

    next_id = [max(nodes) + 1]

    def add(proto, pos, title=None, **widgets):
        n = copy.deepcopy(proto)
        n["id"], n["pos"] = next_id[0], list(pos)
        next_id[0] += 1
        for i in n.get("inputs", []):
            i["link"] = None
        for o in n.get("outputs", []):
            o["links"] = []
        n.setdefault("widgets_values_named", {})
        for k, v in widgets.items():
            set_widget(n, k, v)
        if title:
            n["title"] = title
        main["nodes"].append(n)
        nodes[n["id"]] = n
        return n["id"]

    def proto(ntype, named, inputs, outputs):
        return {"id": 0, "type": ntype, "pos": [0, 0], "size": [300, 90], "flags": {}, "order": 0, "mode": 0,
                "inputs": inputs, "outputs": outputs,
                "properties": {"Node name for S&R": ntype, "ue_properties": UE},
                "widgets_values": list(named.values()), "widgets_values_named": dict(named)}

    def L(o, oslot, t, tname):
        tslot = next(k for k, i in enumerate(nodes[t]["inputs"]) if i["name"] == tname)
        typ = nodes[t]["inputs"][tslot]["type"]
        add_link(o, oslot, t, tslot, "*" if typ == "COMBO" else typ)

    basic = {n["id"]: n for n in sg_by_id(fixed, BASIC_SG)["nodes"]}
    drop_node(978)
    for nid in (2101, 2102):  # stage-1 latent is half size; it no longer takes 923's width/height
        if any(l["id"] == nid for l in main["links"]):
            unlink(nid)
    set_widget(nodes[923], "divisible_by", 64)  # final size /2 must still be a multiple of 32
    nodes[923]["title"] = "Resize start frame to FINAL size"

    # half-size start frame for stage 1
    half = add(proto("ImageScaleBy", {"upscale_method": "lanczos", "scale_by": 0.5},
                     [I("image", "IMAGE")], [O("IMAGE", "IMAGE")]), (-190, 6560), "Stage 1: half-size start frame")
    size = add(proto("GetImageSize", {}, [I("image", "IMAGE")],
                     [O("width", "INT"), O("height", "INT"), O("batch_size", "INT")]), (-190, 6680))
    L(821, 0, half, "image")
    L(half, 0, size, "image")
    L(size, 0, 855, "width")
    L(size, 1, 855, "height")
    for i in nodes[842]["inputs"]:  # stage-1 first-frame guide uses the half-size frame
        if i["name"] == "num_images.image_1" and i["link"] is not None:
            unlink(i["link"])
    L(half, 0, 842, "num_images.image_1")
    nodes[855]["title"] = "Stage 1 latent (half size)"

    # shared: guider, audio latent (frozen by a zero noise mask)
    x1, x2 = 1000, 1900
    guider = add(basic[828], (x1, 5560), "CFG guider (cfg 1)")
    L(979, 0, guider, "model")
    L(927, 0, guider, "positive")
    L(927, 1, guider, "negative")
    trim = add(basic[991], (x1, 6300), "Trim vocals to scene length")
    L(966, 0, trim, "audio")
    add_link(-10, 9, trim, next(k for k, i in enumerate(nodes[trim]["inputs"]) if i["name"] == "duration"), "FLOAT")
    aenc = add(basic[907], (x1, 6440))
    L(trim, 0, aenc, "audio")
    L(964, 0, aenc, "audio_vae")
    zero = add(basic[818], (x1, 6560), "Audio mask = 0 (keep the song)")
    am1 = add(basic[817], (x1, 6700))
    L(aenc, 0, am1, "samples")
    L(zero, 0, am1, "mask")

    # stage 1: half-size generation
    cat1 = add(basic[822], (x1, 5700))
    L(842, 0, cat1, "video_latent")
    L(am1, 0, cat1, "audio_latent")
    noise1 = add(basic[863], (x1, 5820), "Stage 1 noise")
    ksel1 = add(basic[944], (x1, 5940))
    sched1 = add(basic[939], (x1, 6060), "Stage 1 sigmas (10 steps)")
    L(842, 0, sched1, "latent")
    samp1 = add(basic[883], (x1 + 380, 5700), "Stage 1 sample (half size)")
    for src, name in ((noise1, "noise"), (guider, "guider"), (ksel1, "sampler"), (sched1, "sigmas"),
                      (cat1, "latent_image")):
        L(src, 0, samp1, name)
    sep1 = add(basic[825], (x1 + 380, 5960))
    L(samp1, 0, sep1, "av_latent")

    # stage 2: 2x latent upscale, re-apply the first frame at full size, short refine
    uload = add(proto("LatentUpscaleModelLoader", {"model_name": LATENT_UPSCALER}, [],
                      [O("LATENT_UPSCALE_MODEL", "LATENT_UPSCALE_MODEL")]), (x2, 5560), "LTX-2 spatial upscaler x2")
    ups = add(proto("LTXVLatentUpsampler", {},
                    [I("samples", "LATENT"), I("upscale_model", "LATENT_UPSCALE_MODEL"), I("vae", "VAE")],
                    [O("LATENT", "LATENT")]), (x2, 5700), "Stage 2: upscale latent 2x")
    L(sep1, 0, ups, "samples")
    L(uload, 0, ups, "upscale_model")
    L(962, 0, ups, "vae")
    guide2 = add(nodes[842], (x2, 5840), "Stage 2: first frame at full size")
    L(962, 0, guide2, "vae")
    L(ups, 0, guide2, "latent")
    L(821, 0, guide2, "num_images.image_1")
    am2 = add(basic[817], (x2, 6000))
    L(sep1, 1, am2, "samples")
    L(zero, 0, am2, "mask")
    cat2 = add(basic[822], (x2, 6120))
    L(guide2, 0, cat2, "video_latent")
    L(am2, 0, cat2, "audio_latent")
    noise2 = add(basic[863], (x2, 6240), "Stage 2 noise", noise_seed=45)
    ksel2 = add(basic[944], (x2, 6360), sampler_name="euler")
    sig2 = add(proto("ManualSigmas", {"sigmas": STAGE2_SIGMAS}, [I("sigmas", "STRING", True)],
                     [O("SIGMAS", "SIGMAS")]), (x2, 6480), "Stage 2 refine sigmas")
    samp2 = add(basic[883], (x2 + 380, 5700), "Stage 2 refine (full size)")
    for src, name in ((noise2, "noise"), (guider, "guider"), (ksel2, "sampler"), (sig2, "sigmas"),
                      (cat2, "latent_image")):
        L(src, 0, samp2, name)
    sep2 = add(basic[825], (x2 + 380, 5960))
    L(samp2, 0, sep2, "av_latent")
    dec = add(basic[826], (x2 + 380, 6100), "Decode (single 1536 tile)")
    L(sep2, 0, dec, "samples")
    L(962, 0, dec, "vae")
    L(dec, 0, 995, "images")

    main["state"]["lastNodeId"] = next_id[0]
    main["state"]["lastLinkId"] = lid
    main["groups"] = [gr for gr in main["groups"] if gr["title"] != "Output Preview"]
    return main


def instance_for(sg, g, pos, title):
    return g.node(sg["id"], pos, [], {},
                  [dict(I(i["name"], i["type"]), label=i.get("label", i["name"])) for i in sg["inputs"]],
                  [dict(O(o["name"], o["type"]), label=o.get("label", o["name"])) for o in sg["outputs"]],
                  title=title, size=(420, 260))


def motion_prompt(sc):
    return sc["motion"] + (" " + SCENES["hands_motion"] if sc["singer"] else "")


def still_prompt(sc):
    return sc["still"] + (" " + SCENES["hands_still"] if sc["singer"] else "")


def add_scene_clip(g, sg, sc, x, y, shared):
    """One scene: start frame + prompt + audio window -> Scene Clip -> SaveVideo."""
    i, dur = sc["id"], scene_dur(sc)
    img = load_image(g, (x, y), still_name(i), title=f"Scene {i:02d} start frame")
    txt = prim_string(g, (x, y + 360), motion_prompt(sc), f"Scene {i:02d} motion prompt", multiline=True)
    st = prim_string(g, (x, y + 540), sc["start"], "audio start (m:ss)")
    en = prim_string(g, (x, y + 610), sc["end"], "audio end (m:ss)")
    fr = prim_int(g, (x + 440, y + 540), dur * 24 + 1, "frames (8n+1)")
    du = prim_float(g, (x + 440, y + 640), float(dur), "seconds")
    clip = instance_for(sg, g, (x + 440, y), f"Scene {i:02d} | {sc['section']} {sc['start']}-{sc['end']}")
    src = {"image": (img, 0), "audio": shared["audio"], "text": (txt, 0), "width": shared["width"],
           "height": shared["height"], "start_time": (st, 0), "end_time": (en, 0), "value": shared["fps"],
           "frames": (fr, 0), "duration": (du, 0)}
    for slot, inp in enumerate(clip["inputs"]):
        n, s = src[inp["name"]]
        g.link(n, s, clip, slot)
    sv = save_video(g, (x + 880, y), f"god_inside/clip_{i:02d}", title=f"Scene {i:02d} clip")
    g.link(clip, 0, sv, 0)
    return [img, txt, st, en, fr, du, clip, sv]


def workflow_shell(g, subgraphs, groups):
    return {"id": str(uuid.uuid4()), "revision": 0, "last_node_id": g.nid - 1, "last_link_id": g.lid - 1,
            "nodes": g.nodes, "links": g.links, "groups": groups,
            "definitions": {"subgraphs": subgraphs}, "config": {},
            "extra": {"ds": {"scale": 0.35, "offset": [2600, 400]}, "frontendVersion": "1.28.0"},
            "version": 0.4}


def build_music_video(fixed, clip_sg, scenes):
    g = Graph()
    shared_x = -1400
    audio = load_audio(g, (shared_x, 0), AUDIO_FILE)
    w = prim_int(g, (shared_x, 170), WIDTH, "FINAL width (multiple of 64; stage 1 renders at half)")
    h = prim_int(g, (shared_x, 280), HEIGHT, "FINAL height (multiple of 64; stage 1 renders at half)")
    f = prim_float(g, (shared_x, 390), FPS, "FPS")
    note(g, (shared_x, 500), MV_NOTES, "How to run", size=(560, 720))
    shared = {"audio": (audio, 0), "width": (w, 0), "height": (h, 0), "fps": (f, 0)}
    groups = []
    for k, sc in enumerate(scenes):
        x, y = (k % 3) * 1500, (k // 3) * 820
        add_scene_clip(g, clip_sg, sc, x, y + 40, shared)
        groups.append({"id": k + 1, "title": f"Scene {sc['id']:02d} - {sc['section']} ({sc['start']}-{sc['end']})",
                       "bounding": [x - 20, y - 20, 1420, 780], "color": "#3f789e", "flags": {}})
    return workflow_shell(g, [clip_sg], groups)


MV_NOTES = """# God Inside: LTX-2 music video (27 scenes)

Each group renders one scene of the script. Each scene takes its start frame (from workflow 02), the exact slice of the song for that scene (vocals are isolated so the model lip-syncs), and a motion prompt. It saves `output/god_inside/clip_XX.mp4`.

**Order**
1. Run `02_god_inside_scene_stills` first. Copy `output/god_inside_scene_XX_00001_.png` to `input/god_inside_scene_XX.png` (`python3 tools/collect_stills.py <ComfyUI dir>` does this).
2. Queue this workflow. For low RAM, queue a few groups at a time: select the group's SaveVideo, then right-click > *Queue Selected Output Nodes*.
3. Assemble with the full song: `python3 tools/assemble_video.py --clips <ComfyUI>/output/god_inside`.

Or run everything headless: `python3 tools/run_music_video.py --server http://127.0.0.1:8188`.

**Two-stage sampling for clean hands**: each Scene Clip renders at half size (640x352), upscales the latent 2x with `latent_upscale_models/ltx-2-spatial-upscaler-x2-1.0.safetensors` (Lightricks/LTX-2), re-applies the start frame at full size, and refines with 3 steps. Fingers get their detail in that last pass.

**Models**: see the Model Links note inside any Scene Clip subgraph (LTX-2 19B distilled, Gemma-3 12B, LTX2 video/audio VAE) plus the spatial upscaler above.
If you run out of VRAM, lower Width/Height to 1024x576 (must stay a multiple of 64). If the output is black, the resolution is too high.
"""


# ---------------------------------------------------------------------------
# 3) Scene stills (Qwen-Image-Edit-2509, character locked to the reference images)
# ---------------------------------------------------------------------------
def build_stills(scenes):
    g = Graph()
    x0 = -1500
    unet = g.node("UNETLoader", (x0, 0), [QWEN_UNET, "default"], {"unet_name": QWEN_UNET, "weight_dtype": "default"},
                  [], [O("MODEL", "MODEL")], size=(420, 82))
    lora = g.node("LoraLoaderModelOnly", (x0, 120), [QWEN_LORA, 1.0], {"lora_name": QWEN_LORA, "strength_model": 1.0},
                  [I("model", "MODEL")], [O("MODEL", "MODEL")], size=(420, 82))
    aura = g.node("ModelSamplingAuraFlow", (x0, 240), [3.0], {"shift": 3.0}, [I("model", "MODEL")],
                  [O("MODEL", "MODEL")], size=(420, 60))
    cfgn = g.node("CFGNorm", (x0, 340), [1.0], {"strength": 1.0}, [I("model", "MODEL")], [O("MODEL", "MODEL")],
                  size=(420, 60))
    clip = g.node("CLIPLoader", (x0, 440), [QWEN_CLIP, "qwen_image", "default"],
                  {"clip_name": QWEN_CLIP, "type": "qwen_image", "device": "default"}, [], [O("CLIP", "CLIP")],
                  size=(420, 106))
    vae = g.node("VAELoader", (x0, 580), [QWEN_VAE], {"vae_name": QWEN_VAE}, [], [O("VAE", "VAE")], size=(420, 60))
    ref1 = load_image(g, (x0, 680), REF_CLOSEUP, "Reference: singer close-up (image 1)")
    ref2 = load_image(g, (x0, 1040), REF_SHEET, "Reference: character / wardrobe sheet (image 2)")
    note(g, (x0, 1400), STILLS_NOTES, "Scene stills", size=(420, 520))
    g.link(unet, 0, lora, 0)
    g.link(lora, 0, aura, 0)
    g.link(aura, 0, cfgn, 0)
    groups, venue = [], None
    for k, sc in enumerate(scenes):
        x, y = (k % 4) * 1350, (k // 4) * 760
        i = sc["id"]
        enc_in = [I("clip", "CLIP"), I("vae", "VAE", shape=7), I("image1", "IMAGE", shape=7),
                  I("image2", "IMAGE", shape=7), I("image3", "IMAGE", shape=7), I("prompt", "STRING", True)]
        pos = g.node("TextEncodeQwenImageEditPlus", (x, y + 40), [still_prompt(sc)], {"prompt": still_prompt(sc)}, enc_in,
                     [O("CONDITIONING", "CONDITIONING")], title=f"Scene {i:02d} prompt", size=(420, 260),
                     color=("#232", "#353"))
        neg = g.node("TextEncodeQwenImageEditPlus", (x, y + 330), [STILL_NEG], {"prompt": STILL_NEG}, enc_in,
                     [O("CONDITIONING", "CONDITIONING")], title="negative", size=(420, 130), color=("#322", "#533"))
        lat = g.node("EmptySD3LatentImage", (x, y + 490), [WIDTH, 720, 1],
                     {"width": WIDTH, "height": 720, "batch_size": 1},
                     [I("width", "INT", True), I("height", "INT", True), I("batch_size", "INT", True)],
                     [O("LATENT", "LATENT")], size=(420, 106))
        ks = g.node("KSampler", (x + 440, y + 40), [1000 + i, "fixed", QWEN_STEPS, 1.0, "euler", "simple", 1.0],
                    {"seed": 1000 + i, "steps": QWEN_STEPS, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
                     "denoise": 1.0},
                    [I("model", "MODEL"), I("positive", "CONDITIONING"), I("negative", "CONDITIONING"),
                     I("latent_image", "LATENT")], [O("LATENT", "LATENT")], size=(320, 262))
        dec = g.node("VAEDecode", (x + 440, y + 330), [], {}, [I("samples", "LATENT"), I("vae", "VAE")],
                     [O("IMAGE", "IMAGE")], size=(200, 46))
        sv = g.node("SaveImage", (x + 780, y + 40), [f"god_inside_scene_{i:02d}"],
                    {"filename_prefix": f"god_inside_scene_{i:02d}"}, [I("images", "IMAGE")], [],
                    title=f"Scene {i:02d} still", size=(500, 330))
        for enc in (pos, neg):
            g.link(clip, 0, enc, 0)
            g.link(vae, 0, enc, 1)
            if i != 1:
                g.link(ref1, 0, enc, 2)
                g.link(ref2, 0, enc, 3)
                if venue is not None:
                    g.link(venue, 0, enc, 4)
        g.link(cfgn, 0, ks, 0)
        g.link(pos, 0, ks, 1)
        g.link(neg, 0, ks, 2)
        g.link(lat, 0, ks, 3)
        g.link(ks, 0, dec, 0)
        g.link(vae, 0, dec, 1)
        g.link(dec, 0, sv, 0)
        if i == 1:
            venue = dec  # the empty-stage plate locks the venue for every later scene (image 3)
        groups.append({"id": k + 1, "title": f"Scene {i:02d} - {sc['section']}", "bounding": [x - 20, y - 20, 1320, 740],
                       "color": "#3f789e", "flags": {}})
    return workflow_shell(g, [], groups)


STILL_NEG = ("blurry, deformed face, extra fingers, different hairstyle, long hair, curly hair, different dress, "
             "blue dress, text, watermark, logo, cartoon, 3d render, " + SCENES["still_negative_hands"])
STILLS_NOTES = """Qwen-Image-Edit-2509 + Lightning 8-step LoRA (all core ComfyUI nodes).

image 1 = singer close-up, image 2 = wardrobe/character sheet, image 3 = the Scene 01 empty-stage plate, reused as the venue reference for every later scene so the club stays the same.

Models:
- diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors (Comfy-Org/Qwen-Image-Edit_ComfyUI)
- loras/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors (lightx2v/Qwen-Image-Lightning)
- text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors, vae/qwen_image_vae.safetensors (Comfy-Org/Qwen-Image_ComfyUI)

Check every still's hands before rendering video: the LTX clip inherits whatever hand is in its first frame. Change a scene's KSampler seed to re-roll only that shot.
"""


# ---------------------------------------------------------------------------
# UI workflow -> API prompt (flattens subgraphs, drops reroutes / notes / bypassed nodes)
# ---------------------------------------------------------------------------
def to_api(wf):
    sgs = {s["id"]: s for s in wf.get("definitions", {}).get("subgraphs", [])}
    api = {}

    def norm_links(links):
        out = {}
        for l in links:
            if isinstance(l, list):
                l = dict(id=l[0], origin_id=l[1], origin_slot=l[2], target_id=l[3], target_slot=l[4], type=l[5])
            out[l["id"]] = l
        return out

    def expand(nodes, links, prefix, inmap):
        L = norm_links(links)
        N = {n["id"]: n for n in nodes}
        expanded = {}

        def source(link_id):
            l = L[link_id]
            o = l["origin_id"]
            if o == -10:
                return inmap.get(l["origin_slot"])
            n = N[o]
            if n["type"] == "Reroute":
                lk = n["inputs"][0].get("link")
                return source(lk) if lk is not None else None
            if n["type"] in sgs:
                if o not in expanded:
                    expanded[o] = instance(n)
                return expanded[o][l["origin_slot"]]
            if n.get("mode", 0) in (2, 4):
                return None
            return [f"{prefix}{o}", l["origin_slot"]]

        def instance(n):
            sg = sgs[n["type"]]
            m = {}
            for slot, d in enumerate(sg["inputs"]):
                inp = next((i for i in n["inputs"] if i["name"] == d["name"]), None)
                if inp is not None and inp.get("link") is not None:
                    m[slot] = source(inp["link"])
            outs = expand(sg["nodes"], sg["links"], f"{prefix}{n['id']}:", m)
            res = {}
            for l in norm_links(sg["links"]).values():
                if l["target_id"] == -20:
                    res[l["target_slot"]] = outs(l["id"])
            return res

        for n in nodes:
            if n["type"] in VIRTUAL or n.get("mode", 0) in (2, 4):
                continue
            if n["type"] in sgs:
                if n["id"] not in expanded:
                    expanded[n["id"]] = instance(n)
                continue
            inputs = {}
            named = n.get("widgets_values_named")
            if named is None:
                if n.get("widgets_values"):
                    raise ValueError(f"node {n['id']} {n['type']} has widget values but no names")
                named = {}
            for k, v in named.items():
                if k not in SKIP_WIDGETS and v is not None:
                    inputs[k] = v
            for inp in n.get("inputs", []):
                if inp.get("link") is None:
                    continue
                s = source(inp["link"])
                if s is None:
                    inputs.pop(inp["name"], None)
                else:
                    inputs[inp["name"]] = s
            api[f"{prefix}{n['id']}"] = {"class_type": n["type"], "inputs": inputs,
                                          "_meta": {"title": n.get("title", n["type"])}}
        return source

    expand(wf["nodes"], wf["links"], "", {})
    return api


def single_scene_api(clip_sg, basic, sc):
    g = Graph()
    audio = load_audio(g, (0, 0), AUDIO_FILE)
    w = prim_int(g, (0, 0), WIDTH, "Width")
    h = prim_int(g, (0, 0), HEIGHT, "Height")
    f = prim_float(g, (0, 0), FPS, "FPS")
    add_scene_clip(g, clip_sg, sc, 0, 0, {"audio": (audio, 0), "width": (w, 0), "height": (h, 0), "fps": (f, 0)})
    return to_api(workflow_shell(g, [clip_sg], []))


def single_still_api(sc):
    wf = build_stills([sc] if sc["id"] == 1 else [dict(sc)])
    if sc["id"] != 1:
        # outside the full graph the venue plate comes from the saved Scene 01 still
        g = Graph(first_node=wf["last_node_id"] + 1, first_link=wf["last_link_id"] + 1)
        g.nodes, g.links = wf["nodes"], wf["links"]
        plate = load_image(g, (0, 0), still_name(1))
        for n in wf["nodes"]:
            if n["type"] == "TextEncodeQwenImageEditPlus":
                g.link(plate, 0, n, 4)
        wf["nodes"], wf["links"] = g.nodes, g.links
    return to_api(wf)


def main():
    src = json.load(open(SRC))
    fixed = fix_original(src)
    clip_sg = build_scene_clip_sg(fixed)
    basic = copy.deepcopy(sg_by_id(fixed, BASIC_SG))
    scenes = SCENES["scenes"]
    out = os.path.join(ROOT, "workflows")
    os.makedirs(out, exist_ok=True)
    wfs = {"01_ltx2_audio_to_video_FIXED.json": fixed,
           "02_god_inside_scene_stills.json": build_stills(scenes),
           "03_god_inside_music_video_ltx2.json": build_music_video(fixed, clip_sg, scenes)}
    for name, wf in wfs.items():
        json.dump(wf, open(os.path.join(out, name), "w"), indent=1)
    for sub in ("stills", "clips"):
        os.makedirs(os.path.join(ROOT, "api", sub), exist_ok=True)
    json.dump(to_api(fixed), open(os.path.join(ROOT, "api", "01_ltx2_audio_to_video_FIXED.api.json"), "w"), indent=1)
    for sc in scenes:
        json.dump(single_still_api(sc), open(os.path.join(ROOT, "api", "stills", f"scene_{sc['id']:02d}.json"), "w"),
                  indent=1)
        json.dump(single_scene_api(clip_sg, basic, sc),
                  open(os.path.join(ROOT, "api", "clips", f"clip_{sc['id']:02d}.json"), "w"), indent=1)
    print("built", ", ".join(wfs), "+ API prompts for", len(scenes), "scenes")


if __name__ == "__main__":
    main()
