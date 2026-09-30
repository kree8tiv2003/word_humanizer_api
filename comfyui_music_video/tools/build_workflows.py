"""Generates the ComfyUI workflow files (UI format + API format) from one graph definition.

    python tools/build_workflows.py

Writes workflows/music_video_lipsync.json (drag into ComfyUI) and
workflows/music_video_lipsync_api.json (for tools/queue_all.py / the HTTP API).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "workflows")

HF = "https://huggingface.co/Comfy-Org"
MODELS = {
    "unet": ("wan2.2_s2v_14B_fp8_scaled.safetensors", f"{HF}/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors", "diffusion_models"),
    "lora": ("wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors", f"{HF}/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors", "loras"),
    "clip": ("umt5_xxl_fp8_e4m3fn_scaled.safetensors", f"{HF}/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders"),
    "vae": ("wan_2.1_vae.safetensors", f"{HF}/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors", "vae"),
    "audio": ("wav2vec2_large_english_fp16.safetensors", f"{HF}/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/audio_encoders/wav2vec2_large_english_fp16.safetensors", "audio_encoders"),
}

NEGATIVE = (
    "oversaturated, overexposed, static, frozen face, closed mouth while singing, "
    "out of sync lips, blurry details, subtitles, text, watermark, grey overall tone, "
    "worst quality, low quality, jpeg artifacts, ugly, deformed, extra fingers, "
    "poorly drawn hands, poorly drawn face, disfigured, malformed limbs, fused fingers, "
    "cluttered background, three legs, crowd in background, walking backwards"
)
STYLE = "cinematic music video, dramatic lighting, shallow depth of field, 35mm film look, rich color grade"

GUIDE = """## 🎬 Music Video — Wan2.2 S2V lip-sync, storyboard + batch

**1. Upload** (buttons on *MV Script Builder*, 50 files per batch, or drag files onto it)
- 🎵 Audio: the **full 3:03 song** (best — cut frame-exactly here) *or* your 5-second clips (sorted by name: `001.wav`, `002.wav`…)
- 📤 Storyboard images (≈40, name them in story order `01.png`, `02.png`…)
- 📝 Optional script (.txt one line per segment, `image.png | prompt` allowed, or .json/.csv)

**2. Script**: `auto` uses your script → else Claude writes prompts from the storyboard (needs `ANTHROPIC_API_KEY`) → else template prompts. Paste lyrics to feed them into the prompts.

**3. Render**: set `segment_index` = 0 with **increment**, set the Queue count to the number of segments shown in the log (3:03 at 5s = **37**), press Queue. Each run renders one segment; after the last one the final video with the uncut song is written to `output/music_video/<project>/<project>_final.mp4`.

**Quality**: 4-step LightX2V LoRA = fast (steps 4, cfg 1). For maximum lip-sync quality bypass the LoRA node and use steps 20, cfg 6. Nudge `sync_offset_ms` on the save node if lips look early/late.
"""


class Graph:
    def __init__(self):
        self.nodes, self.links = [], []
        self.next_link = 1

    def node(self, nid, ntype, pos, size, widgets=(), wnames=(), title=None, mode=0, models=None, color=None):
        n = {"id": nid, "type": ntype, "pos": list(pos), "size": list(size), "flags": {}, "order": len(self.nodes),
             "mode": mode, "inputs": [], "outputs": [], "properties": {"Node name for S&R": ntype},
             "widgets_values": list(widgets)}
        if title:
            n["title"] = title
        if models:
            n["properties"]["models"] = [{"name": m[0], "url": m[1], "directory": m[2]} for m in models]
        if color:
            n["color"], n["bgcolor"] = color
        n["_wnames"] = list(wnames)
        self.nodes.append(n)
        return n

    def out(self, node, name, typ):
        node["outputs"].append({"name": name, "type": typ, "links": []})

    def connect(self, src, src_name, dst, dst_name, typ, widget=False):
        slot = next(i for i, o in enumerate(src["outputs"]) if o["name"] == src_name)
        lid = self.next_link
        self.next_link += 1
        src["outputs"][slot]["links"].append(lid)
        inp = {"name": dst_name, "type": typ, "link": lid}
        if widget:
            inp["widget"] = {"name": dst_name}
        dst["inputs"].append(inp)
        self.links.append([lid, src["id"], slot, dst["id"], len(dst["inputs"]) - 1, typ])

    def ui_json(self, groups):
        nodes = []
        for n in self.nodes:
            n = dict(n)
            n.pop("_wnames")
            nodes.append(n)
        return {"id": "7d1a0c3e-5b8f-4b7e-9a51-6f0c2d9e4a11", "revision": 0,
                "last_node_id": max(n["id"] for n in self.nodes), "last_link_id": self.next_link - 1,
                "nodes": nodes, "links": self.links, "groups": groups, "config": {},
                "extra": {"ds": {"scale": 0.6, "offset": [40, 60]}}, "version": 0.4}

    def api_json(self):
        prompt = {}
        for n in self.nodes:
            if n["type"] in ("MarkdownNote", "Note") or n["mode"] in (2, 4):
                continue
            inputs = {}
            # widgets_values may contain the extra "control_after_generate" entry; skip names marked None.
            for name, val in zip(n["_wnames"], n["widgets_values"]):
                if name is not None:
                    inputs[name] = val
            for inp in n["inputs"]:
                link = next(l for l in self.links if l[0] == inp["link"])
                inputs[inp["name"]] = [str(link[1]), link[2]]
            prompt[str(n["id"])] = {"class_type": n["type"], "inputs": inputs,
                                    "_meta": {"title": n.get("title", n["type"])}}
        return prompt


def build():
    g = Graph()
    purple = ("#323", "#535")
    green = ("#232", "#353")

    note = g.node(1, "MarkdownNote", (-620, -40), (560, 620), [GUIDE], title="READ ME")

    # ---- models
    unet = g.node(10, "UNETLoader", (-620, 620), (400, 82), [MODELS["unet"][0], "default"], ["unet_name", "weight_dtype"], models=[MODELS["unet"]])
    g.out(unet, "MODEL", "MODEL")
    lora = g.node(11, "LoraLoaderModelOnly", (-620, 740), (400, 82), [MODELS["lora"][0], 1.0], ["lora_name", "strength_model"],
                  title="LightX2V 4-step LoRA (bypass for max quality: steps 20 / cfg 6)", models=[MODELS["lora"]])
    g.out(lora, "MODEL", "MODEL")
    ms = g.node(12, "ModelSamplingSD3", (-620, 860), (400, 58), [8], ["shift"])
    g.out(ms, "MODEL", "MODEL")
    clip = g.node(13, "CLIPLoader", (-620, 960), (400, 106), [MODELS["clip"][0], "wan", "default"], ["clip_name", "type", "device"], models=[MODELS["clip"]])
    g.out(clip, "CLIP", "CLIP")
    vae = g.node(14, "VAELoader", (-620, 1100), (400, 58), [MODELS["vae"][0]], ["vae_name"], models=[MODELS["vae"]])
    g.out(vae, "VAE", "VAE")
    aenc = g.node(15, "AudioEncoderLoader", (-620, 1200), (400, 58), [MODELS["audio"][0]], ["audio_encoder_name"], models=[MODELS["audio"]])
    g.out(aenc, "AUDIO_ENCODER", "AUDIO_ENCODER")

    # ---- project / script / segment
    sb = g.node(20, "MVScriptBuilder", (0, -40), (440, 560),
                ["my_music_video", "auto", 5.0, STYLE, "", "claude-opus-5-5", False],
                ["project", "script_source", "segment_seconds", "style", "lyrics", "claude_model", "regenerate"],
                title="🎬 1. Project, uploads & script", color=purple)
    g.out(sb, "script_json", "STRING")
    g.out(sb, "summary", "STRING")

    ld = g.node(21, "MVSegmentLoader", (0, 580), (440, 420),
                [0, "increment", 832, 480, 42, True, NEGATIVE],
                ["segment_index", None, "width", "height", "base_seed", "continue_motion", "negative_prompt"],
                title="🎬 2. Segment loader (queue once per segment)", color=purple)
    for name, typ in [("ref_image", "IMAGE"), ("audio", "AUDIO"), ("positive", "STRING"), ("negative", "STRING"),
                      ("width", "INT"), ("height", "INT"), ("length", "INT"), ("frames", "INT"),
                      ("ref_motion", "IMAGE"), ("seed", "INT"), ("segment_index", "INT"), ("info", "STRING")]:
        g.out(ld, name, typ)
    g.connect(sb, "script_json", ld, "script_json", "STRING")

    # ---- conditioning
    pos = g.node(30, "CLIPTextEncode", (520, -40), (400, 160), [""], ["text"], title="Positive (from script)", color=green)
    g.out(pos, "CONDITIONING", "CONDITIONING")
    g.connect(clip, "CLIP", pos, "clip", "CLIP")
    g.connect(ld, "positive", pos, "text", "STRING", widget=True)
    neg = g.node(31, "CLIPTextEncode", (520, 160), (400, 160), [""], ["text"], title="Negative", color=("#322", "#533"))
    g.out(neg, "CONDITIONING", "CONDITIONING")
    g.connect(clip, "CLIP", neg, "clip", "CLIP")
    g.connect(ld, "negative", neg, "text", "STRING", widget=True)

    ae = g.node(32, "AudioEncoderEncode", (520, 360), (400, 50))
    g.out(ae, "AUDIO_ENCODER_OUTPUT", "AUDIO_ENCODER_OUTPUT")
    g.connect(aenc, "AUDIO_ENCODER", ae, "audio_encoder", "AUDIO_ENCODER")
    g.connect(ld, "audio", ae, "audio", "AUDIO")

    s2v = g.node(33, "WanSoundImageToVideo", (520, 460), (400, 260), [832, 480, 81, 1],
                 ["width", "height", "length", "batch_size"], title="Wan S2V (audio-driven lip-sync)")
    for name, typ in [("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent", "LATENT")]:
        g.out(s2v, name, typ)
    g.connect(pos, "CONDITIONING", s2v, "positive", "CONDITIONING")
    g.connect(neg, "CONDITIONING", s2v, "negative", "CONDITIONING")
    g.connect(vae, "VAE", s2v, "vae", "VAE")
    g.connect(ae, "AUDIO_ENCODER_OUTPUT", s2v, "audio_encoder_output", "AUDIO_ENCODER_OUTPUT")
    g.connect(ld, "ref_image", s2v, "ref_image", "IMAGE")
    g.connect(ld, "ref_motion", s2v, "ref_motion", "IMAGE")
    g.connect(ld, "width", s2v, "width", "INT", widget=True)
    g.connect(ld, "height", s2v, "height", "INT", widget=True)
    g.connect(ld, "length", s2v, "length", "INT", widget=True)

    # ---- model chain
    g.connect(unet, "MODEL", lora, "model", "MODEL")
    g.connect(lora, "MODEL", ms, "model", "MODEL")

    ks = g.node(40, "KSampler", (1000, -40), (320, 262), [42, "fixed", 4, 1.0, "uni_pc", "simple", 1.0],
                ["seed", None, "steps", "cfg", "sampler_name", "scheduler", "denoise"], title="KSampler (4 steps/cfg 1 with LoRA; 20/6 without)")
    g.out(ks, "LATENT", "LATENT")
    g.connect(ms, "MODEL", ks, "model", "MODEL")
    g.connect(s2v, "positive", ks, "positive", "CONDITIONING")
    g.connect(s2v, "negative", ks, "negative", "CONDITIONING")
    g.connect(s2v, "latent", ks, "latent_image", "LATENT")
    g.connect(ld, "seed", ks, "seed", "INT", widget=True)

    # ---- first-frame VAE fix (from the official ComfyUI S2V template)
    cut = g.node(41, "LatentCut", (1000, 260), (320, 106), ["t", 0, 1], ["dim", "index", "amount"], title="First-frame fix: take 1st latent")
    g.out(cut, "LATENT", "LATENT")
    g.connect(ks, "LATENT", cut, "samples", "LATENT")
    cat = g.node(42, "LatentConcat", (1000, 400), (320, 70), ["t"], ["dim"], title="First-frame fix: prepend it")
    g.out(cat, "LATENT", "LATENT")
    g.connect(cut, "LATENT", cat, "samples1", "LATENT")
    g.connect(ks, "LATENT", cat, "samples2", "LATENT")
    dec = g.node(43, "VAEDecode", (1000, 510), (320, 50))
    g.out(dec, "IMAGE", "IMAGE")
    g.connect(cat, "LATENT", dec, "samples", "LATENT")
    g.connect(vae, "VAE", dec, "vae", "VAE")
    ifb = g.node(44, "ImageFromBatch", (1000, 600), (320, 82), [3, 4096], ["batch_index", "length"], title="First-frame fix: drop duplicated frames")
    g.out(ifb, "IMAGE", "IMAGE")
    g.connect(dec, "IMAGE", ifb, "image", "IMAGE")

    # ---- save + assemble
    sv = g.node(50, "MVSaveSegment", (1400, -40), (420, 520), [True, 0], ["auto_assemble", "sync_offset_ms"],
                title="🎬 3. Save segment → final video", color=purple)
    g.out(sv, "video_path", "STRING")
    g.connect(ifb, "IMAGE", sv, "images", "IMAGE")
    g.connect(sb, "script_json", sv, "script_json", "STRING")
    g.connect(ld, "segment_index", sv, "segment_index", "INT")
    g.connect(ld, "frames", sv, "frames", "INT")
    g.connect(ld, "audio", sv, "audio", "AUDIO")

    asm = g.node(51, "MVAssembleVideo", (1400, 520), (420, 110), [0, 17], ["sync_offset_ms", "crf"],
                 title="Re-stitch final video manually (unmute with Ctrl+M)", mode=2, color=purple)
    g.out(asm, "video_path", "STRING")
    g.connect(sb, "script_json", asm, "script_json", "STRING")

    groups = [
        {"id": 1, "title": "Models (Wan2.2 S2V 14B)", "bounding": [-640, 540, 440, 740], "color": "#3f789e", "font_size": 24, "flags": {}},
        {"id": 2, "title": "Project · Uploads · Script · Segment", "bounding": [-20, -120, 480, 1140], "color": "#a1309b", "font_size": 24, "flags": {}},
        {"id": 3, "title": "Lip-sync generation", "bounding": [500, -120, 840, 820], "color": "#b58b2a", "font_size": 24, "flags": {}},
        {"id": 4, "title": "Output", "bounding": [1380, -120, 460, 770], "color": "#8A8", "font_size": 24, "flags": {}},
    ]
    return g, groups


if __name__ == "__main__":
    g, groups = build()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "music_video_lipsync.json"), "w", encoding="utf-8") as f:
        json.dump(g.ui_json(groups), f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT, "music_video_lipsync_api.json"), "w", encoding="utf-8") as f:
        json.dump(g.api_json(), f, indent=2, ensure_ascii=False)
    print("wrote", OUT)
