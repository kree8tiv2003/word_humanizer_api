"""Core helpers for the Music Video Kit: project folders, audio timeline,
script parsing/generation, segment video I/O and final assembly.

Kept free of ComfyUI imports so it can be tested standalone; the nodes pass in
ComfyUI's input/output directories.
"""
import base64
import csv
import io
import json
import math
import os
import re
from fractions import Fraction

import av
import numpy as np
from PIL import Image, ImageOps

# Wan2.2-S2V is trained at 16 fps and its audio conditioning assumes 16 fps.
S2V_FPS = 16
# Decoded frames lost to the "double first latent" VAE fix (see ImageFromBatch in the workflow).
VAE_FIX_DROPPED_FRAMES = 3
# Maximum reference-motion frames WanSoundImageToVideo consumes.
MAX_REF_MOTION_FRAMES = 73
MAX_FILES_PER_BATCH = 50

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
SCRIPT_EXTS = {".txt", ".json", ".csv", ".md"}
KIND_EXTS = {"images": IMAGE_EXTS, "audio": AUDIO_EXTS, "script": SCRIPT_EXTS}
FULL_TRACK_HINTS = ("full", "master", "song", "track", "mix")

DEFAULT_NEGATIVE = (
    "oversaturated, overexposed, static, frozen face, closed mouth while singing, "
    "out of sync lips, blurry details, subtitles, text, watermark, grey overall tone, "
    "worst quality, low quality, jpeg artifacts, ugly, deformed, extra fingers, "
    "poorly drawn hands, poorly drawn face, disfigured, malformed limbs, fused fingers, "
    "cluttered background, three legs, crowd in background, walking backwards"
)
DEFAULT_STYLE = (
    "cinematic music video, dramatic lighting, shallow depth of field, "
    "35mm film look, rich color grade"
)


# --------------------------------------------------------------------------- paths

def safe_name(name, fallback="my_music_video"):
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", (name or "").strip()).strip("_")
    return name or fallback


def safe_filename(name):
    base = os.path.basename(name or "").strip()
    stem, ext = os.path.splitext(base)
    stem = re.sub(r"[^A-Za-z0-9_\-. ]+", "_", stem).strip(" .") or "file"
    return stem + ext.lower()


def project_input_dir(input_root, project):
    return os.path.join(input_root, "music_video", safe_name(project))


def project_output_dir(output_root, project):
    return os.path.join(output_root, "music_video", safe_name(project))


def kind_dir(input_root, project, kind):
    if kind not in KIND_EXTS:
        raise ValueError(f"Unknown upload kind: {kind}")
    return os.path.join(project_input_dir(input_root, project), kind)


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_files(input_root, project, kind):
    d = kind_dir(input_root, project, kind)
    if not os.path.isdir(d):
        return []
    exts = KIND_EXTS[kind]
    files = [f for f in os.listdir(d)
             if not f.startswith((".", "_")) and os.path.splitext(f)[1].lower() in exts]
    return sorted(files, key=natural_key)


def folder_fingerprint(input_root, project):
    parts = []
    for kind in KIND_EXTS:
        d = kind_dir(input_root, project, kind)
        for f in list_files(input_root, project, kind):
            st = os.stat(os.path.join(d, f))
            parts.append(f"{kind}/{f}:{st.st_size}:{st.st_mtime_ns}")
    return "|".join(parts)


# --------------------------------------------------------------------------- audio

def load_audio(path):
    """Decode any audio (or video) file to float32 stereo. Returns (waveform[2, N], sample_rate)."""
    with av.open(path) as container:
        stream = container.streams.audio[0]
        sr = stream.codec_context.sample_rate or stream.rate
        resampler = av.audio.resampler.AudioResampler(format="fltp", layout="stereo", rate=sr)
        chunks = []
        for frame in container.decode(stream):
            for rf in resampler.resample(frame):
                chunks.append(rf.to_ndarray())
        for rf in resampler.resample(None):
            chunks.append(rf.to_ndarray())
    if not chunks:
        raise ValueError(f"No audio samples decoded from {path}")
    return np.concatenate(chunks, axis=1).astype(np.float32), int(sr)


def pick_audio_mode(audio_files):
    """'full' = one master track sliced into segments, 'segments' = pre-split clips in order."""
    if not audio_files:
        raise ValueError("No audio uploaded. Upload your full song (recommended) or your 5-second clips.")
    if len(audio_files) == 1:
        return "full", audio_files[0]
    for f in audio_files:
        if any(h in os.path.splitext(f)[0].lower() for h in FULL_TRACK_HINTS):
            return "full", f
    return "segments", None


def build_timeline(input_root, project, segment_seconds):
    """Split the song into segments with frame-exact boundaries so the clips add up to the song."""
    audio_dir = kind_dir(input_root, project, "audio")
    files = list_files(input_root, project, "audio")
    mode, full_file = pick_audio_mode(files)
    segments = []
    if mode == "full":
        wav, sr = load_audio(os.path.join(audio_dir, full_file))
        total = wav.shape[1] / sr
        n = max(1, math.ceil(total / segment_seconds - 1e-6))
        # Fold a tiny tail (<1s) into the previous segment rather than render a 0.3s clip.
        if n > 1 and total - (n - 1) * segment_seconds < 1.0:
            n -= 1
        for i in range(n):
            start = i * segment_seconds
            end = total if i == n - 1 else (i + 1) * segment_seconds
            segments.append({"index": i, "start": start, "end": end, "audio_file": full_file})
    else:
        t = 0.0
        for i, f in enumerate(files):
            wav, sr = load_audio(os.path.join(audio_dir, f))
            dur = wav.shape[1] / sr
            segments.append({"index": i, "start": t, "end": t + dur, "audio_file": f})
            t += dur
        total = t
    for s in segments:
        # Frame boundaries rounded on the absolute timeline -> no cumulative drift.
        s["frames"] = int(round(s["end"] * S2V_FPS)) - int(round(s["start"] * S2V_FPS))
        s["start"] = round(s["start"], 4)
        s["end"] = round(s["end"], 4)
    return {"audio_mode": mode, "full_audio_file": full_file, "total_duration": round(total, 4),
            "segments": segments}


def segment_audio(input_root, script, seg):
    """Return a ComfyUI AUDIO-style tuple (waveform[2,N], sr) for one segment."""
    audio_dir = kind_dir(input_root, script["project"], "audio")
    wav, sr = load_audio(os.path.join(audio_dir, seg["audio_file"]))
    if script["audio_mode"] == "full":
        a = int(round(seg["start"] * sr))
        b = int(round(seg["end"] * sr))
        wav = wav[:, a:b]
    return wav, sr


def full_song_audio(input_root, script):
    audio_dir = kind_dir(input_root, script["project"], "audio")
    if script["audio_mode"] == "full":
        return load_audio(os.path.join(audio_dir, script["full_audio_file"]))
    parts, sr0 = [], None
    for seg in script["segments"]:
        wav, sr = load_audio(os.path.join(audio_dir, seg["audio_file"]))
        if sr0 is None:
            sr0 = sr
        elif sr != sr0:
            raise ValueError("All audio clips must share one sample rate when uploading pre-split clips.")
        parts.append(wav)
    return np.concatenate(parts, axis=1), sr0


# --------------------------------------------------------------------------- frames math

def s2v_length_for(frames):
    """WanSoundImageToVideo 'length' (4n+1) that yields >= `frames` after the VAE first-frame fix.

    With L latents the fixed decode yields 4*L+1 frames, minus the dropped ones.
    """
    latents = max(1, math.ceil((frames + VAE_FIX_DROPPED_FRAMES - 1) / 4))
    return (latents - 1) * 4 + 1


# --------------------------------------------------------------------------- script

def spread_index(i, n_items, n_slots):
    return min(n_items - 1, (i * n_items) // max(1, n_slots))


def split_lyrics(lyrics, n):
    lines = [l.strip() for l in (lyrics or "").splitlines() if l.strip()]
    if not lines:
        return [""] * n
    out = [[] for _ in range(n)]
    for k, line in enumerate(lines):
        out[spread_index(k, n, len(lines))].append(line)
    return [" / ".join(x) for x in out]


def template_prompt(style, scene="", lyric=""):
    p = []
    if style:
        p.append(style.strip().rstrip(".") + ".")
    if scene:
        p.append(scene.strip().rstrip(".") + ".")
    p.append("The singer performs this part of the song with clear, accurate lip movements "
             "perfectly matched to the vocals, expressive face and eyes, natural breathing, "
             "subtle head and shoulder movement on the beat. Consistent character, wardrobe and setting.")
    if lyric:
        p.append(f'Lyrics being sung: "{lyric}".')
    return " ".join(p)


def read_text(path):
    """Read a text file whatever program saved it: UTF-8, UTF-16 (Word/Notepad "Unicode") or Windows-1252."""
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("latin-1")


def parse_script_file(path):
    """Parse an uploaded script into a list of {image?, prompt, lyric?} entries (one per segment)."""
    ext = os.path.splitext(path)[1].lower()
    text = read_text(path)
    entries = []
    if ext == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("segments", [])
        for item in data:
            if isinstance(item, str):
                entries.append({"prompt": item})
            elif isinstance(item, dict):
                entries.append({
                    "segment": item.get("segment", item.get("index")),
                    "image": item.get("image"),
                    "prompt": item.get("prompt") or item.get("text") or item.get("description") or "",
                    "lyric": item.get("lyric") or item.get("lyrics") or "",
                })
    elif ext == ".csv":
        for row in csv.DictReader(io.StringIO(text)):
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            seg = row.get("segment") or row.get("index")
            entries.append({
                "segment": int(seg) if seg and seg.isdigit() else None,
                "image": row.get("image") or None,
                "prompt": row.get("prompt") or row.get("text") or row.get("description") or "",
                "lyric": row.get("lyric") or row.get("lyrics") or "",
            })
    else:  # .txt / .md: one line per segment, optionally "image.png | prompt"
        for line in text.splitlines():
            line = line.strip().lstrip("-*").strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                img, prompt = line.split("|", 1)
                entries.append({"image": img.strip(), "prompt": prompt.strip()})
            else:
                entries.append({"prompt": line})
    # Respect explicit segment numbers (1- or 0-based) when present.
    if entries and all(isinstance(e.get("segment"), int) for e in entries):
        base = min(e["segment"] for e in entries)
        ordered = {}
        for e in entries:
            ordered[e["segment"] - base] = e
        entries = [ordered.get(i, {}) for i in range(max(ordered) + 1)]
    return entries


def _resolve_image(name, images):
    if not name:
        return None
    if name in images:
        return name
    low = {i.lower(): i for i in images}
    if name.lower() in low:
        return low[name.lower()]
    stems = {os.path.splitext(i)[0].lower(): i for i in images}
    return stems.get(os.path.splitext(name)[0].lower())


def assemble_script(project, segment_seconds, timeline, images, entries, style, lyrics, source):
    n = len(timeline["segments"])
    lyric_parts = split_lyrics(lyrics, n)
    segs = []
    for i, t in enumerate(timeline["segments"]):
        e = entries[i] if i < len(entries) else {}
        image = _resolve_image(e.get("image"), images) or images[spread_index(i, len(images), n)]
        lyric = e.get("lyric") or lyric_parts[i]
        prompt = (e.get("prompt") or "").strip()
        if prompt and style and source == "uploaded_script":
            prompt = f"{style.strip().rstrip('.')}. {prompt}"
        if not prompt:
            prompt = template_prompt(style, "", lyric)
        segs.append(dict(t, image=image, prompt=prompt, lyric=lyric))
    return {
        "project": safe_name(project),
        "segment_seconds": segment_seconds,
        "fps": S2V_FPS,
        "source": source,
        "audio_mode": timeline["audio_mode"],
        "full_audio_file": timeline["full_audio_file"],
        "total_duration": timeline["total_duration"],
        "total_frames": sum(s["frames"] for s in segs),
        "segments": segs,
    }


def _image_b64(path, max_side=768):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


SCRIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "segment": {"type": "integer"},
                    "image": {"type": "string"},
                    "prompt": {"type": "string"},
                },
                "required": ["segment", "image", "prompt"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["segments"],
    "additionalProperties": False,
}


def claude_script_entries(image_dir, images, timeline, style, lyrics, model, api_key=None):
    """Ask Claude to look at the storyboard and write one video prompt per audio segment."""
    import anthropic  # optional dependency, only needed for this mode

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    n = len(timeline["segments"])
    lyric_parts = split_lyrics(lyrics, n)
    content = []
    for idx, name in enumerate(images):
        content.append({"type": "text", "text": f"Storyboard image {idx + 1}: {name}"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": _image_b64(os.path.join(image_dir, name))}})
    timeline_lines = "\n".join(
        f"segment {s['index']}: {s['start']:.2f}s-{s['end']:.2f}s"
        + (f' lyrics: "{lyric_parts[s["index"]]}"' if lyric_parts[s["index"]] else "")
        for s in timeline["segments"])
    content.append({"type": "text", "text": (
        f"These {len(images)} images are the storyboard for a music video, in story order. "
        f"The song is {timeline['total_duration']:.1f}s long and is rendered in {n} consecutive segments. "
        "Each segment is animated from exactly one storyboard image by an audio-driven, lip-synced "
        "video model (Wan2.2 Sound-to-Video), so the performer in the image sings the audio of that segment.\n\n"
        f"Segments:\n{timeline_lines}\n\n"
        f"Visual style for the whole video: {style or 'match the storyboard'}\n\n"
        "For every segment, choose the storyboard image (exact filename) and write the video prompt. "
        "Keep the storyboard order; spread the images across the song so the story reads naturally; "
        "reuse an image on consecutive segments when a shot should last longer than one segment. "
        "If there are more images than segments, drop the least essential ones. "
        "Each prompt (40-90 words) describes what is visible in the chosen image (subject, wardrobe, "
        "setting, lighting), the performance (singing with expressive, accurate lip movement, emotion "
        "matching the lyrics), and subtle camera and body motion. Do not describe anything that "
        "contradicts the image. Return one entry per segment, segment numbers 0 to "
        f"{n - 1}.")})

    with client.beta.messages.stream(
        model=model,
        max_tokens=64000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "medium",
                       "format": {"type": "json_schema", "schema": SCRIPT_SCHEMA}},
        messages=[{"role": "user", "content": content}],
    ) as stream:
        message = stream.get_final_message()
    if message.stop_reason == "refusal":
        raise RuntimeError("Claude declined to write the script; falling back to template prompts.")
    text = "".join(b.text for b in message.content if b.type == "text")
    data = json.loads(text)
    entries = [{} for _ in range(n)]
    for item in data.get("segments", []):
        i = item.get("segment")
        if isinstance(i, int) and 0 <= i < n:
            entries[i] = {"image": item.get("image"), "prompt": item.get("prompt", "")}
    return entries


# --------------------------------------------------------------------------- video I/O

def _to_uint8(frames):
    """Accept torch tensor or numpy [N,H,W,C] floats 0..1 -> uint8 numpy RGB."""
    if hasattr(frames, "detach"):
        frames = frames.detach().float().cpu().numpy()
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = (np.clip(frames, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return frames[..., :3]


def _add_audio(container, wav, sr):
    stream = container.add_stream("aac", rate=sr, layout="stereo")
    return stream


def _encode_audio(container, stream, wav, sr):
    frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(wav, dtype=np.float32), format="fltp", layout="stereo")
    frame.sample_rate = sr
    frame.pts = 0
    container.mux(stream.encode(frame))
    container.mux(stream.encode(None))


def write_video(path, frames, fps=S2V_FPS, audio=None, crf=12):
    """Write RGB frames (+ optional (wav[2,N], sr)) to an H.264 mp4."""
    frames = _to_uint8(frames)
    n, h, w = frames.shape[:3]
    # yuv420p needs even dimensions.
    h2, w2 = h - h % 2, w - w % 2
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.mp4"
    with av.open(tmp, mode="w") as out:
        vs = out.add_stream("libx264", rate=Fraction(fps))
        vs.width, vs.height, vs.pix_fmt = w2, h2, "yuv420p"
        vs.options = {"crf": str(crf), "preset": "medium"}
        astream = None
        if audio is not None:
            wav, sr = audio
            wav = wav[:, : int(math.ceil(n / fps * sr))]
            astream = _add_audio(out, wav, sr)
        for f in frames:
            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(f[:h2, :w2]), format="rgb24")
            out.mux(vs.encode(vf.reformat(format="yuv420p")))
        out.mux(vs.encode(None))
        if astream is not None:
            _encode_audio(out, astream, wav, sr)
    os.replace(tmp, path)


def read_video_frames(path, last_n=None):
    """Decode an mp4 to float32 numpy [N,H,W,3] in 0..1 (optionally only the last N frames)."""
    frames = []
    with av.open(path) as c:
        for f in c.decode(c.streams.video[0]):
            frames.append(f.to_ndarray(format="rgb24"))
            if last_n and len(frames) > last_n:
                frames.pop(0)
    if not frames:
        return None
    return np.stack(frames).astype(np.float32) / 255.0


def segment_path(output_root, project, index):
    return os.path.join(project_output_dir(output_root, project), "segments", f"seg_{index:03d}.mp4")


def next_unrendered(output_root, script):
    """Index of the first segment without a rendered clip, or None when all are done."""
    for s in script["segments"]:
        if not os.path.exists(segment_path(output_root, script["project"], s["index"])):
            return s["index"]
    return None


def assemble_final(input_root, output_root, script, sync_offset_ms=0, crf=17):
    """Concatenate every rendered segment frame-exactly and lay the original song underneath."""
    project = script["project"]
    missing = [s["index"] for s in script["segments"]
               if not os.path.exists(segment_path(output_root, project, s["index"]))]
    if missing:
        raise FileNotFoundError(f"Segments not rendered yet: {missing}")
    total_frames = sum(s["frames"] for s in script["segments"])
    fps = script.get("fps", S2V_FPS)
    wav, sr = full_song_audio(input_root, script)
    shift = int(round(sync_offset_ms / 1000.0 * sr))
    if shift > 0:  # delay audio
        wav = np.concatenate([np.zeros((2, shift), np.float32), wav], axis=1)
    elif shift < 0:  # advance audio
        wav = wav[:, -shift:]
    need = int(math.ceil(total_frames / fps * sr))
    if wav.shape[1] < need:
        wav = np.concatenate([wav, np.zeros((2, need - wav.shape[1]), np.float32)], axis=1)
    wav = wav[:, :need]

    out_path = os.path.join(project_output_dir(output_root, project), f"{project}_final.mp4")
    tmp = out_path + ".tmp.mp4"
    written = 0
    with av.open(tmp, mode="w") as out:
        vs = None
        size = None
        for s in script["segments"]:
            kept = 0
            last = None
            with av.open(segment_path(output_root, project, s["index"])) as c:
                for f in c.decode(c.streams.video[0]):
                    if kept >= s["frames"]:
                        break
                    rgb = f.to_ndarray(format="rgb24")
                    if vs is None:
                        size = (rgb.shape[1], rgb.shape[0])
                        vs = out.add_stream("libx264", rate=Fraction(fps))
                        vs.width, vs.height, vs.pix_fmt = size[0], size[1], "yuv420p"
                        vs.options = {"crf": str(crf), "preset": "slow"}
                        astream = _add_audio(out, wav, sr)
                    if (rgb.shape[1], rgb.shape[0]) != size:
                        rgb = np.asarray(Image.fromarray(rgb).resize(size, Image.LANCZOS))
                    last = rgb
                    vf = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(format="yuv420p")
                    out.mux(vs.encode(vf))
                    kept += 1
            # Never let a short segment shift the rest of the video out of sync.
            while kept < s["frames"] and last is not None:
                out.mux(vs.encode(av.VideoFrame.from_ndarray(last, format="rgb24").reformat(format="yuv420p")))
                kept += 1
            written += kept
        out.mux(vs.encode(None))
        _encode_audio(out, astream, wav, sr)
    os.replace(tmp, out_path)
    return out_path, written, written / fps
