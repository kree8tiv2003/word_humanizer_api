"""Generates the example ComfyUI workflows in ../workflows from the live node
definitions (so widget order always matches the code).

    python tools/build_workflows.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PKG))
nodes_mod = __import__(os.path.basename(PKG) + ".nodes", fromlist=["x"])
S2S = nodes_mod.NODE_CLASS_MAPPINGS

# Core ComfyUI nodes used by the examples: (inputs, widgets[(name, type, default)], outputs)
CORE = {
    "CheckpointLoaderSimple": ([], [("ckpt_name", "COMBO", "sd_xl_base_1.0.safetensors")],
                               [("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")]),
    "CLIPTextEncode": ([("clip", "CLIP")], [("text", "STRING", "")], [("CONDITIONING", "CONDITIONING")]),
    "FluxGuidance": ([("conditioning", "CONDITIONING")], [("guidance", "FLOAT", 3.5)],
                     [("CONDITIONING", "CONDITIONING")]),
    "EmptyLatentImage": ([], [("width", "INT", 1024), ("height", "INT", 1024), ("batch_size", "INT", 1)],
                         [("LATENT", "LATENT")]),
    "KSampler": ([("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                  ("latent_image", "LATENT")],
                 [("seed", "INT", 0), ("control_after_generate", "CONTROL", "fixed"), ("steps", "INT", 28),
                  ("cfg", "FLOAT", 6.0), ("sampler_name", "COMBO", "dpmpp_2m"),
                  ("scheduler", "COMBO", "karras"), ("denoise", "FLOAT", 1.0)],
                 [("LATENT", "LATENT")]),
    "VAEDecode": ([("samples", "LATENT"), ("vae", "VAE")], [], [("IMAGE", "IMAGE")]),
    "LoadImage": ([], [("image", "COMBO", "example.png"), ("upload", "UPLOAD", "image")],
                  [("IMAGE", "IMAGE"), ("MASK", "MASK")]),
    "StyleModelLoader": ([], [("style_model_name", "COMBO", "flux1-redux-dev.safetensors")],
                         [("STYLE_MODEL", "STYLE_MODEL")]),
    "CLIPVisionLoader": ([], [("clip_name", "COMBO", "sigclip_vision_patch14_384.safetensors")],
                         [("CLIP_VISION", "CLIP_VISION")]),
    "Note": ([], [("text", "STRING", "")], []),
}
WIDGET_TYPES = ("INT", "FLOAT", "STRING", "BOOLEAN")


def s2s_spec(cls):
    it = cls.INPUT_TYPES()
    inputs, widgets = [], []
    for section in ("required", "optional"):
        for name, spec in it.get(section, {}).items():
            typ, opts = spec[0], (spec[1] if len(spec) > 1 else {})
            if isinstance(typ, list):
                widgets.append((name, "COMBO", opts.get("default", typ[0])))
            elif typ in WIDGET_TYPES:
                widgets.append((name, typ, opts.get("default", {"INT": 0, "FLOAT": 0.0, "STRING": "",
                                                                "BOOLEAN": False}[typ])))
                if opts.get("control_after_generate"):
                    widgets.append(("control_after_generate", "CONTROL", "fixed"))
            else:
                inputs.append((name, typ))
    outputs = list(zip(cls.RETURN_NAMES if hasattr(cls, "RETURN_NAMES") else cls.RETURN_TYPES,
                       cls.RETURN_TYPES))
    return inputs, widgets, outputs


class Graph:
    def __init__(self):
        self.nodes, self.links, self.groups = [], [], []

    def add(self, type_, pos, title=None, size=(340, 200), **values):
        inputs, widgets, outputs = S2S_SPEC[type_] if type_ in S2S else CORE[type_]
        vals = []
        for name, wtype, default in widgets:
            vals.append(values.pop(name, default))
        if values:
            raise KeyError(f"{type_}: unknown widgets {list(values)}")
        node = {
            "id": len(self.nodes) + 1, "type": type_, "pos": list(pos), "size": list(size),
            "flags": {}, "order": len(self.nodes), "mode": 0,
            "inputs": [{"name": n, "type": t, "link": None} for n, t in inputs],
            "outputs": [{"name": n, "type": t, "links": [], "slot_index": i} for i, (n, t) in enumerate(outputs)],
            "properties": {"Node name for S&R": type_},
            "widgets_values": vals,
            "_widgets": widgets,
        }
        if title:
            node["title"] = title
        self.nodes.append(node)
        return node

    def link(self, src, out_name, dst, in_name):
        oslot = next(i for i, o in enumerate(src["outputs"]) if o["name"] == out_name)
        otype = src["outputs"][oslot]["type"]
        islot = next((i for i, x in enumerate(dst["inputs"]) if x["name"] == in_name), None)
        if islot is None:  # widget converted to an input
            wtype = next(t for n, t, _ in dst["_widgets"] if n == in_name)
            dst["inputs"].append({"name": in_name, "type": wtype if wtype != "COMBO" else otype,
                                  "widget": {"name": in_name}, "link": None})
            islot = len(dst["inputs"]) - 1
        lid = len(self.links) + 1
        self.links.append([lid, src["id"], oslot, dst["id"], islot, otype])
        src["outputs"][oslot]["links"].append(lid)
        dst["inputs"][islot]["link"] = lid

    def group(self, title, bounding, color="#3f789e"):
        self.groups.append({"title": title, "bounding": bounding, "color": color, "font_size": 24})

    def dump(self, path):
        for n in self.nodes:
            n.pop("_widgets", None)
        wf = {"last_node_id": len(self.nodes), "last_link_id": len(self.links), "nodes": self.nodes,
              "links": self.links, "groups": self.groups, "config": {}, "extra": {}, "version": 0.4}
        with open(path, "w") as f:
            json.dump(wf, f, indent=1)


S2S_SPEC = {k: s2s_spec(v) for k, v in S2S.items()}
OUT = os.path.join(PKG, "workflows")
os.makedirs(OUT, exist_ok=True)


def note(g, pos, text, size=(420, 220)):
    return g.add("Note", pos, size=size, text=text)


# ------------------------------------------------------------------ 01 plan
g = Graph()
load = g.add("S2S_LoadScript", (40, 80), size=(360, 170))
parse = g.add("S2S_ParseScript", (430, 80), size=(360, 200))
bible = g.add("S2S_CharacterBible", (820, 80), size=(420, 420), overrides=json.dumps({
    "EXAMPLE NAME": {"description": "tall woman, 40s, cropped silver hair, green field jacket",
                     "voice": "en-US-JennyNeural"}}, indent=1))
plan = g.add("S2S_PlanShots", (1270, 80), size=(420, 520))
enh = g.add("S2S_EnhancePrompts", (1720, 80), size=(380, 330))
save = g.add("S2S_SaveProject", (2130, 80), size=(420, 300))
g.link(load, "script", parse, "script")
g.link(parse, "screenplay", bible, "screenplay")
g.link(parse, "screenplay", plan, "screenplay")
g.link(bible, "bible", plan, "bible")
g.link(plan, "shots", enh, "shots")
g.link(bible, "bible", enh, "bible")
g.link(parse, "screenplay", save, "screenplay")
g.link(bible, "bible", save, "bible")
g.link(enh, "shots", save, "shots")
note(g, (40, 300), "STEP 1 - PLAN\n\n1. Click 'upload script' (.fountain .fdx .pdf .docx .txt, up to 1000 pages).\n"
     "2. Queue. Read the Character Bible output; fix/extend descriptions in 'overrides' and queue again.\n"
     "3. Optional: set Enhance Prompts backend to anthropic (needs ANTHROPIC_API_KEY) or ollama.\n"
     "4. Save Project writes output/storyboards/<project_name>/ - use the SAME project_name in every other "
     "workflow.\n\nChanging a character's look mid-script: write [[LOOK NAME: new look]] (persists), "
     "[[SCENE LOOK NAME: ...]] (one scene), [[LOOK NAME: reset]], [[AGE NAME: 70]] in the script, or plain "
     "prose like 'Mara, now wearing a raincoat, ...'.", size=(700, 260))
g.dump(os.path.join(OUT, "01_plan_storyboard.json"))

# ------------------------------------------------------------------ 02 reference images
g = Graph()
refs = g.add("S2S_SetReferenceImages", (460, 60), size=(420, 900),
             bind_1="all", bind_2="character:NAME_HERE", bind_3="location:DINER",
             bind_4="shot:SC0001_SH001", usage_4="use_as_frame", bind_5="style", usage_5="reference")
for i in range(1, 6):
    li = g.add("LoadImage", (40, 60 + (i - 1) * 340), title=f"Reference {i}", size=(380, 320))
    g.link(li, "IMAGE", refs, f"image_{i}")
note(g, (920, 60), "REFERENCE IMAGES (up to 5)\n\nUpload an image into each Load Image node (delete the ones you "
     "don't need - every slot is optional) and set how each one is used:\n\n"
     "bind:  all | style | character:NAME | location:TEXT | scenes:3-10 | shot:SC0003_SH002 | shot:42\n"
     "usage:\n  reference     - steers generation (Redux / IP-Adapter) for every matching shot\n"
     "  init_image    - img2img starting frame for matching shots (init_denoise)\n"
     "  use_as_frame  - this exact image IS the storyboard frame for that shot "
     "(or first shot of each scene in a scenes: range); the shot isn't generated\n\n"
     "A character: reference also becomes that character's identity reference for every shot they appear in.",
     size=(560, 420))
g.dump(os.path.join(OUT, "02_reference_images.json"))

# ------------------------------------------------------------------ 03 character sheets
g = Graph()
it = g.add("S2S_CharacterIterator", (40, 60), size=(380, 330))
ck = g.add("CheckpointLoaderSimple", (40, 440), size=(380, 110))
pos = g.add("CLIPTextEncode", (460, 60), title="Positive", size=(380, 160))
neg = g.add("CLIPTextEncode", (460, 260), title="Negative", size=(380, 160))
lat = g.add("EmptyLatentImage", (460, 460), size=(300, 110))
ks = g.add("KSampler", (880, 60), size=(320, 270))
dec = g.add("VAEDecode", (1230, 60), size=(200, 50))
sv = g.add("S2S_SaveCharacterReference", (1230, 160), size=(360, 380))
g.link(it, "positive", pos, "text"); g.link(it, "negative", neg, "text")
g.link(ck, "CLIP", pos, "clip"); g.link(ck, "CLIP", neg, "clip")
g.link(it, "width", lat, "width"); g.link(it, "height", lat, "height")
g.link(it, "seed", ks, "seed")
g.link(ck, "MODEL", ks, "model"); g.link(pos, "CONDITIONING", ks, "positive")
g.link(neg, "CONDITIONING", ks, "negative"); g.link(lat, "LATENT", ks, "latent_image")
g.link(ks, "LATENT", dec, "samples"); g.link(ck, "VAE", dec, "vae")
g.link(dec, "IMAGE", sv, "images"); g.link(it, "character", sv, "character")
g.link(it, "project_name", sv, "project_name")
note(g, (40, 600), "OPTIONAL: CHARACTER REFERENCE SHEETS\n\nGenerates one reference sheet per character that "
     "doesn't already have an uploaded reference. Set Batch count = number of characters and queue. "
     "Re-queue a character with mode=index to re-roll. These sheets are then used automatically as "
     "identity references in every shot the character appears in.", size=(560, 200))
g.dump(os.path.join(OUT, "03_character_sheets.json"))


# ------------------------------------------------------------------ 04 / 05 render shots
def render_graph(flux: bool):
    g = Graph()
    it = g.add("S2S_ShotIterator", (40, 60), size=(380, 330))
    ck = g.add("CheckpointLoaderSimple", (40, 440), size=(380, 110),
               ckpt_name="flux1-dev-fp8.safetensors" if flux else "sd_xl_base_1.0.safetensors")
    pos = g.add("CLIPTextEncode", (460, 60), title="Positive (from shot)", size=(380, 160))
    neg = g.add("CLIPTextEncode", (460, 260), title="Negative (from shot)", size=(380, 160))
    lat = g.add("S2S_ShotLatent", (460, 460), size=(380, 200), latent_channels="16" if flux else "4")
    ks = g.add("KSampler", (1320, 60), size=(320, 270),
               **({"cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "steps": 24} if flux else {}))
    dec = g.add("VAEDecode", (1680, 60), size=(200, 50))
    sv = g.add("S2S_SaveShotImage", (1680, 160), size=(420, 420))
    st = g.add("S2S_ProjectStatus", (40, 600), size=(380, 200))
    g.link(it, "positive", pos, "text"); g.link(it, "negative", neg, "text")
    g.link(ck, "CLIP", pos, "clip"); g.link(ck, "CLIP", neg, "clip")
    g.link(ck, "VAE", lat, "vae")
    for a, b in (("shot_index", "shot_index"), ("width", "width"), ("height", "height"),
                 ("project_name", "project_name")):
        g.link(it, a, lat, b)
    g.link(it, "seed", ks, "seed"); g.link(lat, "denoise", ks, "denoise")
    g.link(ck, "MODEL", ks, "model"); g.link(neg, "CONDITIONING", ks, "negative")
    g.link(lat, "latent", ks, "latent_image")
    g.link(ks, "LATENT", dec, "samples"); g.link(ck, "VAE", dec, "vae")
    g.link(dec, "IMAGE", sv, "images"); g.link(it, "shot_index", sv, "shot_index")
    g.link(it, "project_name", sv, "project_name")
    if flux:
        guide = g.add("FluxGuidance", (880, 60), size=(300, 60))
        sm = g.add("StyleModelLoader", (460, 700), size=(380, 60))
        cv = g.add("CLIPVisionLoader", (460, 800), size=(380, 60))
        rc = g.add("S2S_ReferenceConditioning", (880, 200), size=(380, 250))
        g.link(pos, "CONDITIONING", guide, "conditioning")
        g.link(guide, "CONDITIONING", rc, "conditioning")
        g.link(sm, "STYLE_MODEL", rc, "style_model"); g.link(cv, "CLIP_VISION", rc, "clip_vision")
        g.link(it, "shot_index", rc, "shot_index"); g.link(it, "project_name", rc, "project_name")
        g.link(rc, "conditioning", ks, "positive")
        txt = ("RENDER SHOTS - FLUX + REDUX REFERENCES\n\nEach shot's reference images (uploaded refs bound to "
               "the characters/location in frame, generated character sheets, global style refs) are applied "
               "with Flux Redux. Shots without references pass through untouched.\n"
               "Models: flux1-dev-fp8 checkpoint, flux1-redux-dev (models/style_models), "
               "sigclip_vision_patch14_384 (models/clip_vision).\n\n")
    else:
        g.link(pos, "CONDITIONING", ks, "positive")
        txt = ("RENDER SHOTS - SDXL\n\nFor identity from reference images with SDXL, route the Shot Iterator's "
               "reference_images output into IP-Adapter / InstantID / PuLID nodes before the KSampler.\n\n")
    note(g, (880, 520) if flux else (880, 400), txt +
         "mode=next_missing: every queued run renders the next shot that has no image yet, in storyboard "
         "order. Set Batch count to the 'shots_remaining' shown by Project Status (or use Auto Queue) - you "
         "can stop and resume any time, across sessions. Use mode=index to re-render one shot. Change "
         "seed_offset to re-roll everything deterministically.", size=(560, 300))
    return g


render_graph(False).dump(os.path.join(OUT, "04_render_shots_sdxl.json"))
render_graph(True).dump(os.path.join(OUT, "05_render_shots_flux_redux.json"))

# ------------------------------------------------------------------ 06 audio, lipsync, assemble
g = Graph()
tts = g.add("S2S_DialogueAudio", (40, 60), size=(400, 420))
ls = g.add("S2S_LipSyncBatch", (480, 60), size=(420, 480))
asm = g.add("S2S_AssembleVideo", (940, 60), size=(420, 520))
st = g.add("S2S_ProjectStatus", (1400, 60), size=(380, 200))
g.link(tts, "project_name", ls, "project_name")
g.link(ls, "project_name", asm, "project_name")
note(g, (40, 560), "STEP 3 - DIALOGUE, LIP-SYNC, FINAL VIDEO\n\n1. Dialogue Audio: edge-tts (pip install edge-tts) "
     "gives every character a consistent voice (set in the Character Bible or voice_overrides). Put real "
     "recordings named SC0002_SH003.wav in recorded_audio_folder to use them instead.\n"
     "2. Lip-Sync: point repo_dir at a Wav2Lip or SadTalker checkout, or use custom-command for LatentSync / "
     "MuseTalk / anything with a CLI ({image} {audio} {output} {python}). To lip-sync with ComfyUI nodes "
     "instead, bypass this node and use workflow 07.\n"
     "3. Assemble: all shots in storyboard order -> exports/<name>.mp4 + .srt + edit list CSV. Lip-synced "
     "clips replace their still frames automatically. Shot durations follow the dialogue audio.\n"
     "Everything is cached/resumable - re-run after fixing any shot.", size=(900, 260))
g.dump(os.path.join(OUT, "06_audio_lipsync_assemble.json"))

# ------------------------------------------------------------------ 07 lipsync via comfy nodes
g = Graph()
inp = g.add("S2S_LipSyncShotInputs", (40, 60), size=(400, 320))
sv = g.add("S2S_SaveLipSyncClip", (900, 60), size=(380, 200))
g.link(inp, "frames", sv, "frames")
g.link(inp, "shot_index", sv, "shot_index")
g.link(inp, "fps", sv, "fps")
g.link(inp, "project_name", sv, "project_name")
note(g, (480, 60), "INSERT YOUR LIP-SYNC NODE HERE\n\nThis workflow hands one dialogue shot at a time "
     "(frames + AUDIO) to any ComfyUI lip-sync node pack - e.g. LatentSync, MuseTalk, Wav2Lip, Sonic, "
     "InfiniteTalk/Wan S2V.\n\n  frames/audio -> [lip-sync node] -> IMAGE frames -> Save Lip-Sync Clip.frames\n\n"
     "As wired now it saves the un-animated still as the clip (useful to test). Run workflow 06's Dialogue "
     "Audio first. Set Batch count = number of dialogue shots; next_missing mode walks through them in order.",
     size=(380, 360))
g.dump(os.path.join(OUT, "07_lipsync_with_comfy_nodes.json"))

print("wrote", sorted(os.listdir(OUT)))
