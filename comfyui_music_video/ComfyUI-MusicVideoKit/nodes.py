import hashlib
import json
import os
import logging

import numpy as np
import torch

import folder_paths

from . import mv_core as core

log = logging.getLogger("MusicVideoKit")


def _roots():
    return folder_paths.get_input_directory(), folder_paths.get_output_directory()


def _load_script(script_json):
    try:
        script = json.loads(script_json)
    except Exception as e:
        raise ValueError("script_json must come from the 'MV Script Builder' node.") from e
    if not script.get("segments"):
        raise ValueError("The script has no segments. Upload audio and images first.")
    return script


class MVScriptBuilder:
    """Builds the per-segment plan: which storyboard image + which prompt goes with each audio segment."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project": ("STRING", {"default": "my_music_video", "tooltip": "Folder name under ComfyUI/input/music_video/. Use the upload buttons on this node."}),
                "script_source": (["auto", "uploaded_script", "claude_vision", "template"], {"default": "auto", "tooltip": "auto = uploaded script if present, else Claude (if ANTHROPIC_API_KEY is set), else template prompts."}),
                "segment_seconds": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 30.0, "step": 0.5, "tooltip": "Only used when you upload the full song (recommended). Pre-split clips keep their own lengths."}),
                "style": ("STRING", {"multiline": True, "default": core.DEFAULT_STYLE}),
                "lyrics": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional. Paste lyrics (one line per sung line); they are spread across segments and fed to the prompts."}),
                "claude_model": ("STRING", {"default": "claude-opus-5-5"}),
                "regenerate": ("BOOLEAN", {"default": False, "tooltip": "Force a new Claude script instead of reusing the saved one."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("script_json", "summary")
    FUNCTION = "build"
    CATEGORY = "MusicVideoKit"

    @classmethod
    def IS_CHANGED(cls, project, **kwargs):
        input_root, _ = _roots()
        return hashlib.sha256(core.folder_fingerprint(input_root, project).encode()).hexdigest()

    def build(self, project, script_source, segment_seconds, style, lyrics, claude_model, regenerate):
        input_root, output_root = _roots()
        project = core.safe_name(project)
        images = core.list_files(input_root, project, "images")
        if not images:
            raise ValueError(f"No storyboard images in {core.kind_dir(input_root, project, 'images')}. Use 'Upload images' on this node.")
        timeline = core.build_timeline(input_root, project, segment_seconds)
        scripts = core.list_files(input_root, project, "script")
        out_dir = core.project_output_dir(output_root, project)
        os.makedirs(out_dir, exist_ok=True)
        cache_path = os.path.join(out_dir, "claude_script_cache.json")
        cache_key = hashlib.sha256(json.dumps(
            [core.folder_fingerprint(input_root, project), segment_seconds, style, lyrics, claude_model]).encode()).hexdigest()

        source = script_source
        if source == "auto":
            if scripts:
                source = "uploaded_script"
            elif os.environ.get("ANTHROPIC_API_KEY") or os.path.exists(cache_path):
                source = "claude_vision"
            else:
                source = "template"

        entries, note = [], ""
        if source == "uploaded_script":
            if not scripts:
                raise ValueError("script_source is 'uploaded_script' but no script was uploaded.")
            entries = core.parse_script_file(os.path.join(core.kind_dir(input_root, project, "script"), scripts[0]))
            note = f"script file {scripts[0]} ({len(entries)} lines)"
        elif source == "claude_vision":
            cached = None
            if os.path.exists(cache_path) and not regenerate:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
            if cached and cached.get("key") == cache_key:
                entries, note = cached["entries"], "Claude script (cached)"
            else:
                try:
                    entries = core.claude_script_entries(
                        core.kind_dir(input_root, project, "images"), images, timeline, style, lyrics, claude_model)
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump({"key": cache_key, "entries": entries}, f, indent=2)
                    note = "Claude script (new)"
                except Exception as e:
                    log.warning("Claude script generation failed (%s); using template prompts.", e)
                    entries, source, note = [], "template", f"template (Claude failed: {e})"
        if source == "template" and not note:
            note = "template prompts"

        script = core.assemble_script(project, segment_seconds, timeline, images, entries, style, lyrics, source)
        with open(os.path.join(out_dir, "script_used.json"), "w", encoding="utf-8") as f:
            json.dump(script, f, indent=2)

        n = len(script["segments"])
        summary = (f"Project '{project}': {len(images)} images, {n} segments, "
                   f"{script['total_duration']:.2f}s song ({script['audio_mode']} audio), "
                   f"{script['total_frames']} frames @ {core.S2V_FPS}fps. Prompts: {note}. "
                   f"Queue the workflow {n} times (segment_index -1 renders the next unfinished segment each run). "
                   f"Editable plan saved to output/music_video/{project}/script_used.json")
        log.info(summary)
        return (json.dumps(script), summary)


class MVSegmentLoader:
    """Loads everything one segment needs: reference image, audio slice, prompts, frame counts."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script_json": ("STRING", {"forceInput": True}),
                "segment_index": ("INT", {"default": -1, "min": -1, "max": 9999, "tooltip": "-1 = automatic: render the next segment that isn't done yet (queue the workflow once per segment). Set a number to re-render that one segment."}),
                "width": ("INT", {"default": 832, "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 480, "min": 256, "max": 2048, "step": 16}),
                "base_seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffff, "tooltip": "Base seed; each segment uses seed + index so re-renders are reproducible."}),
                "continue_motion": ("BOOLEAN", {"default": True, "tooltip": "When consecutive segments share an image, feed the previous segment's last frames as reference motion for a seamless shot."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": core.DEFAULT_NEGATIVE}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "STRING", "INT", "INT", "INT", "INT", "IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("ref_image", "audio", "positive", "negative", "width", "height", "length",
                    "frames", "ref_motion", "seed", "segment_index", "info")
    FUNCTION = "load"
    CATEGORY = "MusicVideoKit"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-run: which segment is next depends on what is already rendered on disk.
        return float("nan")

    def load(self, script_json, segment_index, width, height, base_seed, continue_motion, negative_prompt):
        input_root, output_root = _roots()
        script = _load_script(script_json)
        segs = script["segments"]
        if segment_index >= len(segs):
            raise ValueError(f"segment_index {segment_index} is past the last segment ({len(segs) - 1}). "
                             f"All segments are queued - check output/music_video/{script['project']}/.")
        if segment_index < 0:
            segment_index = core.next_unrendered(output_root, script)
            if segment_index is None:
                raise ValueError(f"All {len(segs)} segments are already rendered. Your video is "
                                 f"output/music_video/{script['project']}/{script['project']}_final.mp4. "
                                 "To redo one segment, set segment_index to its number; to start over, "
                                 f"delete the folder output/music_video/{script['project']}/segments.")
        seg = segs[segment_index]

        img_path = os.path.join(core.kind_dir(input_root, script["project"], "images"), seg["image"])
        from PIL import Image, ImageOps
        im = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
        ref = torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0)[None]

        wav, sr = core.segment_audio(input_root, script, seg)
        audio = {"waveform": torch.from_numpy(wav)[None], "sample_rate": sr}

        ref_motion = None
        if continue_motion and segment_index > 0 and segs[segment_index - 1]["image"] == seg["image"]:
            prev = core.segment_path(output_root, script["project"], segment_index - 1)
            if os.path.exists(prev):
                frames = core.read_video_frames(prev, last_n=core.MAX_REF_MOTION_FRAMES)
                if frames is not None:
                    ref_motion = torch.from_numpy(frames)

        length = core.s2v_length_for(seg["frames"])
        info = (f"segment {segment_index + 1}/{len(segs)} | {seg['start']:.2f}-{seg['end']:.2f}s | "
                f"{seg['frames']} frames (S2V length {length}) | image {seg['image']} | "
                f"motion continued: {ref_motion is not None}")
        log.info(info)
        return (ref, audio, seg["prompt"], negative_prompt, width, height, length,
                seg["frames"], ref_motion, base_seed + segment_index, segment_index, info)


class MVSaveSegment:
    """Trims the rendered clip to its exact frame count, saves it, and builds the final video after the last one."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "script_json": ("STRING", {"forceInput": True}),
                "segment_index": ("INT", {"forceInput": True}),
                "frames": ("INT", {"forceInput": True}),
                "auto_assemble": ("BOOLEAN", {"default": True, "tooltip": "When every segment exists, stitch the final music video automatically."}),
                "sync_offset_ms": ("INT", {"default": 0, "min": -500, "max": 500, "step": 5, "tooltip": "Fine lip-sync nudge for the final video. + delays the audio, - advances it."}),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "MusicVideoKit"

    def save(self, images, script_json, segment_index, frames, auto_assemble, sync_offset_ms, audio=None):
        input_root, output_root = _roots()
        script = _load_script(script_json)
        project = script["project"]
        imgs = images
        if imgs.shape[0] < frames:
            log.warning("Segment %d produced %d frames, expected %d; holding the last frame.",
                        segment_index, imgs.shape[0], frames)
            pad = imgs[-1:].repeat(frames - imgs.shape[0], 1, 1, 1)
            imgs = torch.cat([imgs, pad], dim=0)
        imgs = imgs[:frames]

        aud = None
        if audio is not None:
            aud = (audio["waveform"][0].float().cpu().numpy(), int(audio["sample_rate"]))
            if aud[0].shape[0] == 1:
                aud = (np.repeat(aud[0], 2, axis=0), aud[1])
            aud = (aud[0][:2], aud[1])
        path = core.segment_path(output_root, project, segment_index)
        core.write_video(path, imgs, fps=script.get("fps", core.S2V_FPS), audio=aud)

        preview_path = path
        n = len(script["segments"])
        done = [s["index"] for s in script["segments"] if os.path.exists(core.segment_path(output_root, project, s["index"]))]
        if auto_assemble and len(done) == n:
            final, count, secs = core.assemble_final(input_root, output_root, script, sync_offset_ms)
            log.info("Final music video: %s (%d frames, %.2fs)", final, count, secs)
            preview_path = final
        else:
            log.info("Saved %s (%d/%d segments rendered)", path, len(done), n)

        rel = os.path.relpath(os.path.dirname(preview_path), output_root)
        return {"ui": {"images": [{"filename": os.path.basename(preview_path), "subfolder": rel, "type": "output"}],
                       "animated": (True,)},
                "result": (preview_path,)}


class MVAssembleVideo:
    """Stitches all rendered segments with the original, uncut song (run manually if needed)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script_json": ("STRING", {"forceInput": True}),
                "sync_offset_ms": ("INT", {"default": 0, "min": -500, "max": 500, "step": 5}),
                "crf": ("INT", {"default": 17, "min": 0, "max": 40}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "assemble"
    OUTPUT_NODE = True
    CATEGORY = "MusicVideoKit"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def assemble(self, script_json, sync_offset_ms, crf):
        input_root, output_root = _roots()
        script = _load_script(script_json)
        final, count, secs = core.assemble_final(input_root, output_root, script, sync_offset_ms, crf)
        rel = os.path.relpath(os.path.dirname(final), output_root)
        return {"ui": {"images": [{"filename": os.path.basename(final), "subfolder": rel, "type": "output"}],
                       "animated": (True,)},
                "result": (final,)}


NODE_CLASS_MAPPINGS = {
    "MVScriptBuilder": MVScriptBuilder,
    "MVSegmentLoader": MVSegmentLoader,
    "MVSaveSegment": MVSaveSegment,
    "MVAssembleVideo": MVAssembleVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MVScriptBuilder": "🎬 MV Script Builder (upload + storyboard → script)",
    "MVSegmentLoader": "🎬 MV Segment Loader",
    "MVSaveSegment": "🎬 MV Save Segment (+ auto final assembly)",
    "MVAssembleVideo": "🎬 MV Assemble Final Video",
}
