"""ComfyUI node definitions for the Script-to-Storyboard pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import time

from .core.script_io import SUPPORTED_EXTENSIONS, load_script, normalize_text, check_length, estimate_pages
from .core.parser import parse_screenplay, screenplay_stats, Screenplay
from .core.characters import build_bible, parse_overrides, slug
from .core.shots import STYLE_PRESETS, DEFAULT_NEGATIVE, PlannerSettings, plan_shots, shots_preview
from .core.project import Project, safe_name, MAX_REFERENCES, REF_USAGES, parse_bind
from .core import llm as llm_mod
from .core import media

try:
    import folder_paths
except ImportError:  # allows importing outside ComfyUI (tests / workflow builder)
    folder_paths = None

CATEGORY = "Script2Storyboard"
ASPECTS = {
    "16:9 (1344x768)": (1344, 768), "2.39:1 scope (1536x640)": (1536, 640),
    "1.85:1 (1344x736)": (1344, 736), "4:3 (1152x864)": (1152, 864),
    "1:1 (1024x1024)": (1024, 1024), "9:16 vertical (768x1344)": (768, 1344),
    "16:9 SD1.5 (768x432)": (768, 432),
}


# ---------------------------------------------------------------- helpers
def _output_dir():
    return folder_paths.get_output_directory() if folder_paths else os.path.abspath("output")


def _input_dir():
    return folder_paths.get_input_directory() if folder_paths else os.path.abspath("input")


def project_root(project_name: str) -> str:
    if os.path.isabs(project_name):
        return project_name
    return os.path.join(_output_dir(), "storyboards", safe_name(project_name))


def _list_scripts():
    base = _input_dir()
    out = []
    if os.path.isdir(base):
        for root, _, files in os.walk(base):
            for f in files:
                if f.lower().endswith(SUPPORTED_EXTENSIONS):
                    out.append(os.path.relpath(os.path.join(root, f), base).replace("\\", "/"))
    return sorted(out) or ["(upload a script)"]


def _progress(total):
    try:
        import comfy.utils
        bar = comfy.utils.ProgressBar(total)
        return lambda done, tot: bar.update_absolute(done, tot)
    except Exception:
        return None


def _pil_to_tensor(img):
    import numpy as np
    import torch
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None]


def _tensor_to_pil(t):
    import numpy as np
    from PIL import Image
    arr = (t.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr[..., :3])


def _fit(img, w, h, mode="cover"):
    from PIL import Image, ImageOps
    img = img.convert("RGB")
    if mode == "cover":
        return ImageOps.fit(img, (w, h), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h))
    im = ImageOps.contain(img, (w, h), Image.LANCZOS)
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return canvas


def _ui_image(path):
    out = _output_dir()
    rel = os.path.relpath(path, out)
    if rel.startswith(".."):
        return None
    return {"filename": os.path.basename(rel), "subfolder": os.path.dirname(rel), "type": "output"}


def _load_project(name):
    return Project.load(project_root(name))


def _pick_shot(p: Project, mode: str, shot_index: int, predicate=None):
    n = len(p.shots)
    if n == 0:
        raise ValueError("Project has no shots")
    if mode == "index":
        return max(0, min(shot_index, n - 1))
    idx = p.next_missing(shot_index, predicate)
    if idx is None:
        raise RuntimeError(f"All shots from #{shot_index} onward are done "
                           f"({n} total). Nothing left to render - stop Auto Queue.")
    return idx


# ---------------------------------------------------------------- 1. load
class S2S_LoadScript:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "script_file": (_list_scripts(), {"tooltip": "Script in ComfyUI/input. Use the upload button. "
                                                      "Formats: .fountain .txt .fdx .pdf .docx .md"}),
                    "max_pages": ("INT", {"default": 1000, "min": 1, "max": 5000}),
                },
                "optional": {
                    "script_text": ("STRING", {"multiline": True, "default": "",
                                               "tooltip": "Paste a script here to ignore script_file"}),
                }}

    RETURN_TYPES = ("S2S_SCRIPT", "STRING")
    RETURN_NAMES = ("script", "info")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, script_file, max_pages, script_text=""):
        if script_text and script_text.strip():
            return hashlib.sha256(script_text.encode()).hexdigest()
        p = os.path.join(_input_dir(), script_file)
        return f"{p}:{os.path.getmtime(p)}" if os.path.exists(p) else script_file

    def load(self, script_file, max_pages, script_text=""):
        if script_text and script_text.strip():
            text, name = normalize_text(script_text), "pasted_script"
        else:
            path = os.path.join(_input_dir(), script_file)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Script not found: {path}")
            text, name = load_script(path), os.path.splitext(os.path.basename(path))[0]
        pages = check_length(text, max_pages)
        info = f"{name}: ~{pages} pages, {len(text):,} characters"
        return {"ui": {"text": [info]}, "result": ({"name": name, "text": text, "pages": pages}, info)}


# ---------------------------------------------------------------- 2. parse
class S2S_ParseScript:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "script": ("S2S_SCRIPT",),
            "fallback_paragraphs_per_scene": ("INT", {"default": 8, "min": 1, "max": 100,
                "tooltip": "Only used when the text has no INT./EXT. scene headings"}),
        }}

    RETURN_TYPES = ("S2S_SCREENPLAY", "STRING")
    RETURN_NAMES = ("screenplay", "stats")
    FUNCTION = "parse"
    CATEGORY = CATEGORY

    def parse(self, script, fallback_paragraphs_per_scene):
        sp = parse_screenplay(script["text"], title=script["name"],
                              fallback_paragraphs_per_scene=fallback_paragraphs_per_scene)
        sp.pages = max(sp.pages, script.get("pages", 1))
        st = screenplay_stats(sp)
        return {"ui": {"text": [st]}, "result": (sp, st)}


# ---------------------------------------------------------------- 3. characters
class S2S_CharacterBible:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "screenplay": ("S2S_SCREENPLAY",),
            "consistent_characters": ("BOOLEAN", {"default": True,
                "tooltip": "Inject each character's fixed description into every shot they appear in"}),
            "project_seed": ("INT", {"default": 1234, "min": 0, "max": 0xFFFFFFFF}),
            "min_dialogue_lines": ("INT", {"default": 1, "min": 0, "max": 1000}),
            "include_non_speaking": ("BOOLEAN", {"default": True}),
            "overrides": ("STRING", {"multiline": True, "default": "",
                "tooltip": 'JSON: {"MARA": {"description": "...", "voice": "en-US-JennyNeural", '
                           '"gender": "female", "age": "35"}}  or one "NAME: description" per line'}),
        }}

    RETURN_TYPES = ("S2S_BIBLE", "STRING")
    RETURN_NAMES = ("bible", "summary")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, screenplay, consistent_characters, project_seed, min_dialogue_lines,
              include_non_speaking, overrides):
        bible = build_bible(screenplay, project_seed=project_seed, consistent=consistent_characters,
                            overrides=parse_overrides(overrides), min_lines=min_dialogue_lines,
                            include_nonspeaking=include_non_speaking)
        sm = bible.summary()
        return {"ui": {"text": [sm]}, "result": (bible, sm)}


# ---------------------------------------------------------------- 4. shots
class S2S_PlanShots:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "screenplay": ("S2S_SCREENPLAY",),
            "bible": ("S2S_BIBLE",),
            "style": (list(STYLE_PRESETS.keys()),),
            "style_prefix": ("STRING", {"default": "", "multiline": False}),
            "style_suffix": ("STRING", {"default": "", "multiline": False}),
            "negative_prompt": ("STRING", {"default": DEFAULT_NEGATIVE, "multiline": True}),
            "establishing_shots": ("BOOLEAN", {"default": True}),
            "dialogue_shots": ("BOOLEAN", {"default": True,
                                           "tooltip": "One shot per dialogue line (needed for lip-sync)"}),
            "max_shots_per_scene": ("INT", {"default": 12, "min": 1, "max": 200}),
            "max_action_words": ("INT", {"default": 45, "min": 10, "max": 200}),
            "include_character_names": ("BOOLEAN", {"default": False,
                "tooltip": "Off: names become 'the woman/the man' (image models can't know names)"}),
            "scene_start": ("INT", {"default": 1, "min": 1, "max": 100000}),
            "scene_end": ("INT", {"default": -1, "min": -1, "max": 100000, "tooltip": "-1 = last scene"}),
        }}

    RETURN_TYPES = ("S2S_SHOTS", "STRING")
    RETURN_NAMES = ("shots", "preview")
    FUNCTION = "plan"
    CATEGORY = CATEGORY

    def plan(self, screenplay, bible, style, style_prefix, style_suffix, negative_prompt,
             establishing_shots, dialogue_shots, max_shots_per_scene, max_action_words,
             include_character_names, scene_start, scene_end):
        s = PlannerSettings(style=style, style_prefix=style_prefix, style_suffix=style_suffix,
                            negative=negative_prompt, establishing_shots=establishing_shots,
                            dialogue_shots=dialogue_shots, max_shots_per_scene=max_shots_per_scene,
                            max_action_words=max_action_words, include_names=include_character_names,
                            project_seed=bible.project_seed)
        shots = plan_shots(screenplay, bible, s, scene_start - 1, scene_end - 1 if scene_end > 0 else -1)
        for i, sh in enumerate(shots):  # re-index so the saved order is contiguous
            sh.index = i
        head = f"{len(shots)} shots, ~{sum(x.duration for x in shots) / 60:.1f} min runtime\n\n"
        pv = head + shots_preview(shots)
        return {"ui": {"text": [pv]}, "result": ({"shots": shots, "settings": s.__dict__}, pv)}


# ---------------------------------------------------------------- 5. LLM enhance (optional)
class S2S_EnhancePrompts:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "shots": ("S2S_SHOTS",),
            "bible": ("S2S_BIBLE",),
            "backend": (list(llm_mod.LLM_BACKENDS),),
            "model": ("STRING", {"default": "claude-opus-5-5",
                                 "tooltip": "anthropic: a Claude model id. ollama: a local model name, e.g. llama3.1"}),
            "effort": (["low", "medium", "high"], {"default": "low"}),
            "ollama_host": ("STRING", {"default": "http://127.0.0.1:11434"}),
            "project_name": ("STRING", {"default": "my_storyboard",
                                        "tooltip": "Used for the per-scene response cache (resume)"}),
            "scene_start": ("INT", {"default": 1, "min": 1, "max": 100000}),
            "scene_end": ("INT", {"default": -1, "min": -1, "max": 100000}),
        }}

    RETURN_TYPES = ("S2S_SHOTS", "STRING")
    RETURN_NAMES = ("shots", "report")
    FUNCTION = "enhance"
    CATEGORY = CATEGORY

    def enhance(self, shots, bible, backend, model, effort, ollama_host, project_name,
                scene_start, scene_end):
        root = project_root(project_name)
        os.makedirs(root, exist_ok=True)
        style = ", ".join(x for x in (shots["settings"].get("style_prefix"),
                                      STYLE_PRESETS.get(shots["settings"].get("style"), ("",))[0]) if x)
        n_scenes = len({s.scene_index for s in shots["shots"]})
        rep = llm_mod.enhance_prompts(shots["shots"], bible, backend, model, style_hint=style,
                                      effort=effort, ollama_host=ollama_host,
                                      cache_path=os.path.join(root, "llm_cache.json"),
                                      scene_start=scene_start, scene_end=scene_end,
                                      progress=_progress(n_scenes))
        txt = json.dumps(rep, indent=1) + "\n\n" + shots_preview(shots["shots"])
        return {"ui": {"text": [txt]}, "result": (shots, txt)}


# ---------------------------------------------------------------- 6. save project
class S2S_SaveProject:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "screenplay": ("S2S_SCREENPLAY",),
            "bible": ("S2S_BIBLE",),
            "shots": ("S2S_SHOTS",),
            "project_name": ("STRING", {"default": "my_storyboard"}),
            "aspect": (list(ASPECTS.keys()),),
            "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("project_name", "summary")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def save(self, screenplay, bible, shots, project_name, aspect, fps):
        w, h = ASPECTS[aspect]
        p = Project.create(project_root(project_name), screenplay, bible, shots["shots"],
                           {"width": w, "height": h, "fps": fps, "planner": shots["settings"]})
        st = p.status()
        txt = (f"Saved project to {p.root}\n{json.dumps(st, indent=1)}\n\n"
               f"Characters:\n{bible.summary()}")
        with open(p.path("prompts.txt"), "w", encoding="utf-8") as f:
            f.write(shots_preview(p.shots, limit=10 ** 9))
        return {"ui": {"text": [txt]}, "result": (project_name, txt)}


# ---------------------------------------------------------------- 7. reference images (max 5)
class S2S_SetReferenceImages:
    @classmethod
    def INPUT_TYPES(cls):
        opt = {}
        for i in range(1, MAX_REFERENCES + 1):
            opt[f"image_{i}"] = ("IMAGE",)
            opt[f"bind_{i}"] = ("STRING", {"default": "all" if i == 1 else "",
                "tooltip": "all | style | character:NAME | location:DINER | scenes:3-10 | shot:SC0003_SH002"})
            opt[f"usage_{i}"] = (list(REF_USAGES), {"default": "reference",
                "tooltip": "reference: condition generation (Redux/IP-Adapter). init_image: img2img start "
                           "frame. use_as_frame: put this exact image in the storyboard (shot/scene binds)"})
            opt[f"weight_{i}"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05})
        return {"required": {
                    "project_name": ("STRING", {"default": "my_storyboard"}),
                    "init_denoise": ("FLOAT", {"default": 0.65, "min": 0.05, "max": 1.0, "step": 0.05,
                                               "tooltip": "Denoise used for init_image references"}),
                    "clear_unconnected": ("BOOLEAN", {"default": False,
                        "tooltip": "Remove stored references for slots with no image connected"}),
                },
                "optional": opt}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("project_name", "summary")
    FUNCTION = "set_refs"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def set_refs(self, project_name, init_denoise, clear_unconnected, **kw):
        p = _load_project(project_name)
        previews = []
        for i in range(1, MAX_REFERENCES + 1):
            img = kw.get(f"image_{i}")
            if img is None:
                if clear_unconnected:
                    p.set_reference(i, None, "")
                continue
            bind = kw.get(f"bind_{i}") or "all"
            parse_bind(bind)  # validate early with a clear error
            dst = p.path("references", f"_upload_ref{i}.png")
            _tensor_to_pil(img[0]).save(dst)
            p.set_reference(i, dst, bind, kw.get(f"usage_{i}", "reference"),
                            kw.get(f"weight_{i}", 1.0), init_denoise)
            os.remove(dst)
            ui = _ui_image(p.path("references", f"ref{i}.png"))
            if ui:
                previews.append(ui)
        lines = [f"{r['id']}: bind={r['bind_raw']} usage={r['usage']} weight={r['weight']}"
                 for r in p.references]
        used = {}
        for sh in p.shots:
            for r in p.references_for_shot(sh):
                used[r["id"]] = used.get(r["id"], 0) + 1
        lines.append("shots using each reference: " + json.dumps(used))
        txt = "\n".join(lines) or "no references set"
        return {"ui": {"images": previews, "text": [txt]}, "result": (project_name, txt)}


# ---------------------------------------------------------------- 8/9. character reference sheets
class S2S_CharacterIterator:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "project_name": ("STRING", {"default": "my_storyboard"}),
            "mode": (["next_missing", "index"],),
            "character_index": ("INT", {"default": 0, "min": 0, "max": 10000, "control_after_generate": True}),
            "portrait_template": ("STRING", {"multiline": True, "default":
                "character reference sheet, full body and close-up portrait of the same person, "
                "neutral pose, plain light grey background, even studio lighting, {description}, {style}"}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("positive", "negative", "seed", "character", "project_name", "width", "height")
    FUNCTION = "next"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, mode, **kw):
        return float("nan") if mode == "next_missing" else ""

    def next(self, project_name, mode, character_index, portrait_template):
        p = _load_project(project_name)
        chars = sorted(p.bible.characters.values(), key=lambda c: (-c.dialogue_lines, c.name))
        chars = [c for c in chars if not c.variant_of]
        if not chars:
            raise ValueError("No characters in project")
        if mode == "index":
            c = chars[min(character_index, len(chars) - 1)]
        else:
            todo = [c for c in chars[character_index:]
                    if not (c.reference_image and os.path.exists(p.path(c.reference_image)))]
            if not todo:
                raise RuntimeError("Every character already has a reference image.")
            c = todo[0]
        planner = p.settings.get("planner", {})
        style = ", ".join(x for x in (planner.get("style_prefix"),
                                      STYLE_PRESETS.get(planner.get("style"), ("",))[0]) if x)
        desc = p.bible.prompt_fragment(c.name, -1) or c.name.title()
        pos = portrait_template.replace("{description}", desc).replace("{style}", style).replace("{name}", c.name.title())
        neg = planner.get("negative", DEFAULT_NEGATIVE) + ", multiple different people"
        return {"ui": {"text": [f"{c.name}\n{pos}"]}, "result": (pos, neg, c.seed, c.name, project_name, 1024, 1024)}


class S2S_SaveCharacterReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "project_name": ("STRING", {"default": "my_storyboard"}),
            "character": ("STRING", {"default": ""}),
        }}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def save(self, images, project_name, character):
        p = _load_project(project_name)
        name = p.bible.resolve(character)
        if not name:
            raise ValueError(f"Unknown character {character!r}")
        rel = os.path.join("characters", slug(name) + ".png")
        _tensor_to_pil(images[0]).save(p.path(rel))
        p.bible.characters[name].reference_image = rel
        p.save_meta()
        ui = _ui_image(p.path(rel))
        return {"ui": {"images": [ui] if ui else [], "text": [f"{name} -> {rel}"]}}


# ---------------------------------------------------------------- 10. shot iterator
class S2S_ShotIterator:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "project_name": ("STRING", {"default": "my_storyboard"}),
            "mode": (["next_missing", "index"], {"tooltip": "next_missing: each queued run renders the next "
                                                  "shot without an image (resumable). index: exact shot."}),
            "shot_index": ("INT", {"default": 0, "min": 0, "max": 10 ** 6, "control_after_generate": True}),
            "reference_size": ("INT", {"default": 1024, "min": 224, "max": 2048, "step": 32}),
            "seed_offset": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF,
                                    "tooltip": "Change to re-roll every shot while keeping them deterministic"}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "INT", "INT", "IMAGE", "INT", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("positive", "negative", "seed", "width", "height", "shot_index",
                    "reference_images", "reference_count", "has_references", "info", "project_name")
    FUNCTION = "next"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, mode, **kw):
        return float("nan") if mode == "next_missing" else ""

    def next(self, project_name, mode, shot_index, reference_size, seed_offset):
        import torch
        from PIL import Image
        p = _load_project(project_name)
        idx = _pick_shot(p, mode, shot_index)
        sh = p.shots[idx]
        refs = [r for r in p.references_for_shot(sh) if r.get("usage", "reference") == "reference"]
        imgs = []
        for r in refs:
            fp = p.path(r["file"])
            if os.path.exists(fp):
                imgs.append(_pil_to_tensor(_fit(Image.open(fp), reference_size, reference_size, "contain")))
        batch = torch.cat(imgs, 0) if imgs else torch.zeros((1, reference_size, reference_size, 3))
        w, h = p.settings.get("width", 1344), p.settings.get("height", 768)
        info = (f"#{idx} {sh.id} [{sh.kind} / {sh.shot_size}] scene {sh.scene_index + 1}: {sh.heading}\n"
                f"characters: {', '.join(sh.characters) or '-'}\nreferences: "
                f"{', '.join(r['id'] for r in refs) or '-'}\n"
                + (f'{sh.dialogue["character"]}: "{sh.dialogue["text"]}"' if sh.dialogue else sh.source_text[:300]))
        seed = (sh.seed + seed_offset) % 0xFFFFFFFF
        return {"ui": {"text": [info + "\n\n" + sh.prompt]},
                "result": (sh.prompt, sh.negative, seed, w, h, idx, batch, len(imgs), bool(imgs), info, project_name)}


# ---------------------------------------------------------------- 11. reference conditioning (core, no deps)
class S2S_ReferenceConditioning:
    """Applies the shot's reference images with a style model (e.g. Flux Redux)
    using ComfyUI core objects. Shots without references pass through
    unchanged, so one workflow handles both."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "style_model": ("STYLE_MODEL",),
            "clip_vision": ("CLIP_VISION",),
            "project_name": ("STRING", {"default": "my_storyboard"}),
            "shot_index": ("INT", {"default": 0, "min": 0, "max": 10 ** 6}),
            "strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
            "max_references": ("INT", {"default": 3, "min": 1, "max": MAX_REFERENCES + 20}),
            "crop": (["center", "none"],),
        }}

    RETURN_TYPES = ("CONDITIONING", "INT")
    RETURN_NAMES = ("conditioning", "applied")
    FUNCTION = "apply"
    CATEGORY = CATEGORY

    def apply(self, conditioning, style_model, clip_vision, project_name, shot_index, strength,
              max_references, crop):
        import torch
        from PIL import Image
        p = _load_project(project_name)
        sh = p.shots[shot_index]
        refs = [r for r in p.references_for_shot(sh) if r.get("usage", "reference") == "reference"]
        applied = 0
        out = conditioning
        for r in refs[:max_references]:
            fp = p.path(r["file"])
            if not os.path.exists(fp):
                continue
            img = _pil_to_tensor(Image.open(fp))
            cv = clip_vision.encode_image(img, crop=(crop == "center"))
            cond = style_model.get_cond(cv).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
            cond = cond * (strength * float(r.get("weight", 1.0)))
            out = [[torch.cat((t[0], cond.to(t[0].device, t[0].dtype)), dim=1), t[1].copy()] for t in out]
            applied += 1
        return (out, applied)


# ---------------------------------------------------------------- 12. latent (empty or init image)
class S2S_ShotLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "vae": ("VAE",),
            "project_name": ("STRING", {"default": "my_storyboard"}),
            "shot_index": ("INT", {"default": 0, "min": 0, "max": 10 ** 6}),
            "width": ("INT", {"default": 1344, "min": 64, "max": 8192, "step": 8}),
            "height": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 8}),
            "latent_channels": (["4", "16", "128"], {"default": "4",
                "tooltip": "4 = SD1.5/SDXL, 16 = SD3/Flux, 128 = Flux 2. Empty latents only."}),
        }}

    RETURN_TYPES = ("LATENT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("latent", "denoise", "uses_init_image")
    FUNCTION = "make"
    CATEGORY = CATEGORY

    def make(self, vae, project_name, shot_index, width, height, latent_channels):
        import torch
        from PIL import Image
        p = _load_project(project_name)
        sh = p.shots[shot_index]
        w, h = width // 8 * 8, height // 8 * 8
        for r in p.references_for_shot(sh):
            if r.get("usage") == "init_image" and os.path.exists(p.path(r["file"])):
                px = _pil_to_tensor(_fit(Image.open(p.path(r["file"])), w, h, "cover"))
                return ({"samples": vae.encode(px[:, :, :, :3])}, float(r.get("denoise", 0.65)), True)
        return ({"samples": torch.zeros([1, int(latent_channels), h // 8, w // 8])}, 1.0, False)


# ---------------------------------------------------------------- 13. save shot image
class S2S_SaveShotImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "project_name": ("STRING", {"default": "my_storyboard"}),
            "shot_index": ("INT", {"default": 0, "min": 0, "max": 10 ** 6}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def save(self, images, project_name, shot_index):
        from PIL.PngImagePlugin import PngInfo
        p = _load_project(project_name)
        sh = p.shots[shot_index]
        meta = PngInfo()
        meta.add_text("s2s_shot", json.dumps({"id": sh.id, "index": sh.index, "prompt": sh.prompt,
                                              "seed": sh.seed}))
        path = p.image_path(shot_index)
        _tensor_to_pil(images[0]).save(path, pnginfo=meta, compress_level=4)
        for extra in range(1, images.shape[0]):  # keep alternates for picking later
            _tensor_to_pil(images[extra]).save(path[:-4] + f"_alt{extra}.png")
        st = p.status()
        txt = f"saved {sh.id} (#{shot_index}) - {st['images']}/{st['shots']} shots rendered"
        ui = _ui_image(path)
        return {"ui": {"images": [ui] if ui else [], "text": [txt]}, "result": (txt,)}


# ---------------------------------------------------------------- 14. load sequence
class S2S_LoadStoryboardSequence:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "project_name": ("STRING", {"default": "my_storyboard"}),
            "start_shot": ("INT", {"default": 0, "min": 0, "max": 10 ** 6}),
            "count": ("INT", {"default": 64, "min": 1, "max": 5000,
                              "tooltip": "Keep this modest - every frame is held in memory"}),
            "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
            "height": ("INT", {"default": 576, "min": 64, "max": 8192, "step": 8}),
            "missing": (["placeholder", "skip"],),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("images", "count", "timeline")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("nan")

    def load(self, project_name, start_shot, count, width, height, missing):
        import torch
        from PIL import Image
        p = _load_project(project_name)
        frames, lines = [], []
        for sh in p.shots[start_shot:start_shot + count]:
            fp = p.frame_for_shot(sh)
            if fp:
                frames.append(_pil_to_tensor(_fit(Image.open(fp), width, height, "contain")))
            elif missing == "placeholder":
                frames.append(torch.full((1, height, width, 3), 0.12))
            else:
                continue
            lines.append(f"{sh.index}\t{sh.id}\t{sh.duration:.2f}s\t{sh.kind}")
        if not frames:
            raise RuntimeError("No frames in range")
        return (torch.cat(frames, 0), len(frames), "\n".join(lines))


# ---------------------------------------------------------------- 15. dialogue audio
class S2S_DialogueAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "project_name": ("STRING", {"default": "my_storyboard"}),
            "tts_backend": (list(media.TTS_BACKENDS),),
            "start_shot": ("INT", {"default": 0, "min": 0, "max": 10 ** 6}),
            "max_lines": ("INT", {"default": -1, "min": -1, "max": 10 ** 6, "tooltip": "-1 = all"}),
            "overwrite": ("BOOLEAN", {"default": False}),
            "recorded_audio_folder": ("STRING", {"default": "",
                "tooltip": "Optional folder of real recordings named SC0002_SH003.wav / shot_000005.wav; "
                           "these take priority over TTS"}),
            "voice_overrides": ("STRING", {"multiline": True, "default": "",
                                           "tooltip": '{"MARA": "en-GB-SoniaNeural"}'}),
            "command_template": ("STRING", {"default": "",
                "tooltip": "For tts_backend=command, e.g. piper --model {voice} --output_file {output} < {text_file}"}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("project_name", "report")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def run(self, project_name, tts_backend, start_shot, max_lines, overwrite, recorded_audio_folder,
            voice_overrides, command_template):
        p = _load_project(project_name)
        vo = json.loads(voice_overrides) if voice_overrides.strip() else {}
        vo = {(p.bible.resolve(k) or k.upper()): v for k, v in vo.items()}
        n = sum(1 for s in p.shots if s.dialogue)
        rep = media.generate_dialogue_audio(p, tts_backend, start_shot, max_lines, overwrite,
                                            command_template, recorded_audio_folder, vo, _progress(n))
        txt = json.dumps(rep, indent=1)
        return {"ui": {"text": [txt]}, "result": (project_name, txt)}


# ---------------------------------------------------------------- 16. external lip-sync (batch)
class S2S_LipSyncBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "project_name": ("STRING", {"default": "my_storyboard"}),
            "backend": (list(media.LIPSYNC_BACKENDS),),
            "repo_dir": ("STRING", {"default": "", "tooltip": "Path to Wav2Lip / SadTalker / other repo"}),
            "checkpoint": ("STRING", {"default": "", "tooltip": "wav2lip: .pth file. sadtalker: checkpoints dir"}),
            "python_executable": ("STRING", {"default": "", "tooltip": "Blank = ComfyUI's python"}),
            "command_template": ("STRING", {"default": "",
                "tooltip": "custom-command: e.g. {python} scripts/inference.py --video {image} --audio {audio} --out {output}"}),
            "extra_args": ("STRING", {"default": ""}),
            "start_shot": ("INT", {"default": 0, "min": 0, "max": 10 ** 6}),
            "max_clips": ("INT", {"default": -1, "min": -1, "max": 10 ** 6}),
            "overwrite": ("BOOLEAN", {"default": False}),
            "only_on_screen_speakers": ("BOOLEAN", {"default": True,
                "tooltip": "Skip V.O./O.S. lines where the speaker's face isn't the subject"}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("project_name", "report")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def run(self, project_name, backend, repo_dir, checkpoint, python_executable, command_template,
            extra_args, start_shot, max_clips, overwrite, only_on_screen_speakers):
        p = _load_project(project_name)
        n = sum(1 for s in p.shots if s.dialogue)
        rep = media.lipsync_project(p, backend, start_shot, max_clips, overwrite, only_on_screen_speakers,
                                    progress=_progress(n), repo_dir=repo_dir, checkpoint=checkpoint,
                                    command_template=command_template, python=python_executable,
                                    extra_args=extra_args)
        txt = json.dumps(rep, indent=1)
        return {"ui": {"text": [txt]}, "result": (project_name, txt)}


# ---------------------------------------------------------------- 17/18. lip-sync via ComfyUI nodes
class S2S_LipSyncShotInputs:
    """Feeds one dialogue shot (frame + line audio) into any ComfyUI lip-sync
    node pack (LatentSync, MuseTalk, Wav2Lip, Sonic, InfiniteTalk ...)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "project_name": ("STRING", {"default": "my_storyboard"}),
            "mode": (["next_missing", "index"],),
            "shot_index": ("INT", {"default": 0, "min": 0, "max": 10 ** 6, "control_after_generate": True}),
            "fps": ("INT", {"default": 25, "min": 1, "max": 60}),
            "repeat_frames": ("BOOLEAN", {"default": True,
                "tooltip": "Output the still repeated for the line's duration (video-driven lip-sync nodes)"}),
            "only_on_screen_speakers": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "INT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("frames", "audio", "shot_index", "frame_count", "fps", "info", "project_name")
    FUNCTION = "get"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, mode, **kw):
        return float("nan") if mode == "next_missing" else ""

    def get(self, project_name, mode, shot_index, fps, repeat_frames, only_on_screen_speakers):
        import torch
        from PIL import Image
        p = _load_project(project_name)

        def needs(i):
            s = p.shots[i]
            return (s.dialogue and (s.dialogue.get("on_screen", True) or not only_on_screen_speakers)
                    and not os.path.exists(p.clip_path(i)) and os.path.exists(p.audio_path(i))
                    and p.frame_for_shot(s) is not None)
        idx = _pick_shot(p, mode, shot_index, needs)
        sh = p.shots[idx]
        if not sh.dialogue:
            raise ValueError(f"Shot {idx} has no dialogue")
        fp, ap = p.frame_for_shot(sh), p.audio_path(idx)
        if not fp or not os.path.exists(ap):
            raise FileNotFoundError(f"Shot {idx} needs both a rendered image and dialogue audio first")
        w, h = p.settings.get("width", 1344), p.settings.get("height", 768)
        img = _pil_to_tensor(_fit(Image.open(fp), w // 8 * 8, h // 8 * 8, "contain"))
        wav, sr = media.read_wav(ap)
        dur = wav.shape[1] / sr
        n = max(1, int(round(dur * fps)))
        frames = img.repeat(n, 1, 1, 1) if repeat_frames else img
        audio = {"waveform": torch.from_numpy(wav.copy())[None], "sample_rate": sr}
        info = f'#{idx} {sh.id} {sh.dialogue["character"]}: "{sh.dialogue["text"]}" ({dur:.2f}s)'
        return {"ui": {"text": [info]}, "result": (frames, audio, idx, n, float(fps), info, project_name)}


class S2S_SaveLipSyncClip:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "frames": ("IMAGE",),
                    "project_name": ("STRING", {"default": "my_storyboard"}),
                    "shot_index": ("INT", {"default": 0, "min": 0, "max": 10 ** 6}),
                    "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 120.0}),
                },
                "optional": {"audio": ("AUDIO",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("clip_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def save(self, frames, project_name, shot_index, fps, audio=None):
        import numpy as np
        p = _load_project(project_name)
        wav = p.audio_path(shot_index)
        if audio is not None:
            wf = audio["waveform"][0].detach().cpu().float().numpy()
            media.write_wav(wav, wf, audio["sample_rate"])
        arr = (frames.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)[..., :3]
        out = p.clip_path(shot_index)
        media.frames_to_mp4(arr, fps, out, wav if os.path.exists(wav) else None)
        return {"ui": {"text": [f"saved {out}"]}, "result": (out,)}


# ---------------------------------------------------------------- 19. assemble
class S2S_AssembleVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "project_name": ("STRING", {"default": "my_storyboard"}),
            "output_name": ("STRING", {"default": "storyboard"}),
            "width": ("INT", {"default": 1280, "min": 64, "max": 7680, "step": 2}),
            "height": ("INT", {"default": 720, "min": 64, "max": 4320, "step": 2}),
            "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
            "scene_start": ("INT", {"default": 1, "min": 1, "max": 100000}),
            "scene_end": ("INT", {"default": -1, "min": -1, "max": 100000}),
            "include_dialogue_audio": ("BOOLEAN", {"default": True}),
            "use_lipsync_clips": ("BOOLEAN", {"default": True}),
            "ken_burns": ("BOOLEAN", {"default": True, "tooltip": "Slow push-in on non-dialogue stills"}),
            "burn_subtitles": ("BOOLEAN", {"default": False}),
            "missing_frames": (["placeholder", "skip"],),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "report")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def run(self, project_name, output_name, width, height, fps, scene_start, scene_end,
            include_dialogue_audio, use_lipsync_clips, ken_burns, burn_subtitles, missing_frames):
        p = _load_project(project_name)
        rep = media.assemble_video(p, safe_name(output_name), width, height, fps, scene_start, scene_end,
                                   include_dialogue_audio, use_lipsync_clips, ken_burns, burn_subtitles,
                                   missing_frames, _progress(len(p.shots)))
        txt = json.dumps(rep, indent=1)
        ui = {"text": [txt]}
        v = _ui_image(rep["video"])
        if v:
            ui["gifs"] = [{**v, "format": "video/h264-mp4"}]
        return {"ui": ui, "result": (rep["video"], txt)}


# ---------------------------------------------------------------- 20. status
class S2S_ProjectStatus:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"project_name": ("STRING", {"default": "my_storyboard"})}}

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("status", "shots_remaining")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kw):
        return time.time()

    def run(self, project_name):
        p = _load_project(project_name)
        st = p.status()
        remaining = st["missing_frames"]
        txt = (f"{p.meta.get('title')} @ {p.root}\n{json.dumps(st, indent=1)}\n"
               f"Set 'Batch count' to {remaining} on the render workflow to finish all shots.")
        return {"ui": {"text": [txt]}, "result": (txt, remaining)}


NODE_CLASS_MAPPINGS = {
    "S2S_LoadScript": S2S_LoadScript,
    "S2S_ParseScript": S2S_ParseScript,
    "S2S_CharacterBible": S2S_CharacterBible,
    "S2S_PlanShots": S2S_PlanShots,
    "S2S_EnhancePrompts": S2S_EnhancePrompts,
    "S2S_SaveProject": S2S_SaveProject,
    "S2S_SetReferenceImages": S2S_SetReferenceImages,
    "S2S_CharacterIterator": S2S_CharacterIterator,
    "S2S_SaveCharacterReference": S2S_SaveCharacterReference,
    "S2S_ShotIterator": S2S_ShotIterator,
    "S2S_ReferenceConditioning": S2S_ReferenceConditioning,
    "S2S_ShotLatent": S2S_ShotLatent,
    "S2S_SaveShotImage": S2S_SaveShotImage,
    "S2S_LoadStoryboardSequence": S2S_LoadStoryboardSequence,
    "S2S_DialogueAudio": S2S_DialogueAudio,
    "S2S_LipSyncBatch": S2S_LipSyncBatch,
    "S2S_LipSyncShotInputs": S2S_LipSyncShotInputs,
    "S2S_SaveLipSyncClip": S2S_SaveLipSyncClip,
    "S2S_AssembleVideo": S2S_AssembleVideo,
    "S2S_ProjectStatus": S2S_ProjectStatus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S2S_LoadScript": "S2S 1 · Load Script (upload)",
    "S2S_ParseScript": "S2S 2 · Parse Screenplay",
    "S2S_CharacterBible": "S2S 3 · Character Bible",
    "S2S_PlanShots": "S2S 4 · Plan Shots & Prompts",
    "S2S_EnhancePrompts": "S2S 4b · Enhance Prompts (LLM, optional)",
    "S2S_SaveProject": "S2S 5 · Save Storyboard Project",
    "S2S_SetReferenceImages": "S2S · Reference Images (up to 5)",
    "S2S_CharacterIterator": "S2S · Character Sheet Iterator",
    "S2S_SaveCharacterReference": "S2S · Save Character Reference",
    "S2S_ShotIterator": "S2S 6 · Shot Iterator",
    "S2S_ReferenceConditioning": "S2S · Apply Shot References (Redux)",
    "S2S_ShotLatent": "S2S · Shot Latent (empty / init image)",
    "S2S_SaveShotImage": "S2S 7 · Save Shot Image",
    "S2S_LoadStoryboardSequence": "S2S · Load Storyboard Sequence",
    "S2S_DialogueAudio": "S2S 8 · Dialogue Audio (TTS)",
    "S2S_LipSyncBatch": "S2S 9 · Lip-Sync (Wav2Lip/SadTalker/CLI)",
    "S2S_LipSyncShotInputs": "S2S 9 · Lip-Sync Shot Inputs (for lip-sync nodes)",
    "S2S_SaveLipSyncClip": "S2S 9 · Save Lip-Sync Clip",
    "S2S_AssembleVideo": "S2S 10 · Assemble Video",
    "S2S_ProjectStatus": "S2S · Project Status",
}
