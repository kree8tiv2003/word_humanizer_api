#!/usr/bin/env python3
"""Drive ComfyUI to render the "God Inside" music video, 5 seconds at a time.

Stages (run all with `all`, or one at a time):
  keyframes  - one cinematic keyframe per scene from your character image (Qwen-Image-Edit)
  segments   - one lip-synced 5 s clip per storyboard segment (Wan2.2 S2V)
  assemble   - stitch every frame together and mux the original song

Only needs Python 3.9+ and ffmpeg on PATH. ComfyUI must be running (default http://127.0.0.1:8188).
"""
import argparse
import json
import math
import mimetypes
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
FPS = 16  # Wan2.2 S2V native frame rate; 80 frames == 5.0 s
CLIENT_ID = str(uuid.uuid4())


# ---------------------------------------------------------------- ComfyUI API
class Comfy:
    def __init__(self, server):
        self.server = server.rstrip("/")

    def _req(self, path, data=None, headers=None):
        req = urllib.request.Request(self.server + path, data=data, headers=headers or {})
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.read()

    def upload(self, path, name=None):
        name = name or Path(path).name
        boundary = uuid.uuid4().hex
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        body = b"".join([
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n".encode(),
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n".encode(),
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n".encode(),
            Path(path).read_bytes(),
            f"\r\n--{boundary}--\r\n".encode(),
        ])
        res = json.loads(self._req("/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}))
        return res["name"] if not res.get("subfolder") else f"{res['subfolder']}/{res['name']}"

    def run(self, prompt, poll=3.0):
        payload = json.dumps({"prompt": prompt, "client_id": CLIENT_ID}).encode()
        try:
            res = json.loads(self._req("/prompt", payload, {"Content-Type": "application/json"}))
        except urllib.error.HTTPError as e:
            sys.exit(f"ComfyUI rejected the workflow:\n{e.read().decode(errors='replace')}")
        pid = res["prompt_id"]
        while True:
            hist = json.loads(self._req(f"/history/{pid}"))
            if pid in hist:
                entry = hist[pid]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                    sys.exit(f"ComfyUI execution error: {json.dumps(msgs, indent=1)[:3000]}")
                if status.get("completed", True):
                    return entry["outputs"]
            time.sleep(poll)

    def download_images(self, outputs, node_id, dest_dir):
        dest_dir.mkdir(parents=True, exist_ok=True)
        files = []
        for img in sorted(outputs[node_id]["images"], key=lambda i: i["filename"]):
            q = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img["subfolder"], "type": img["type"]})
            p = dest_dir / img["filename"]
            p.write_bytes(self._req(f"/view?{q}"))
            files.append(p)
        return files


# ---------------------------------------------------------------- helpers
def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *map(str, args)], check=True)


def load_json(p):
    return json.loads(Path(p).read_text())


def parse_range(spec, n):
    if not spec:
        return list(range(1, n + 1))
    ids = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        ids.update(range(int(a), int(b or a) + 1))
    return sorted(i for i in ids if 1 <= i <= n)


def seg_dir(out, sid):
    return out / "segments" / f"seg_{sid:02d}"


def seg_frames(out, sid):
    return sorted(seg_dir(out, sid).glob("f_*.png"))


# ---------------------------------------------------------------- stages
def stage_keyframes(args, sb, comfy):
    wf_t = load_json(HERE / "workflows" / "keyframe_qwen_edit_api.json")
    kf_dir = args.out / "keyframes"
    char_name = comfy.upload(args.image, "god_inside_character" + Path(args.image).suffix)
    sheet_name = comfy.upload(args.sheet, "god_inside_sheet" + Path(args.sheet).suffix) if args.sheet else None
    for sc_id, scene in sb["scenes"].items():
        if args.scenes and sc_id not in args.scenes:
            continue
        target = kf_dir / f"scene_{sc_id}.png"
        if target.exists() and not args.force:
            print(f"[keyframe {sc_id}] exists, skipping")
            continue
        wf = json.loads(json.dumps(wf_t))
        wf["6"]["inputs"]["image"] = char_name
        if sheet_name:
            wf["8"]["inputs"]["image"] = sheet_name
        else:
            del wf["8"]
            for n in ("9", "10"):
                del wf[n]["inputs"]["image2"]
        wf["9"]["inputs"]["prompt"] = f"The character is {sb['character']}. " + scene["keyframe_prompt"]
        wf["11"]["inputs"].update(width=args.kf_width, height=args.kf_height)
        wf["12"]["inputs"]["seed"] = args.seed + ord(sc_id)
        wf["14"]["inputs"]["filename_prefix"] = f"god_inside/keyframes/scene_{sc_id}"
        print(f"[keyframe {sc_id}] {scene['name']} ...", flush=True)
        files = comfy.download_images(comfy.run(wf), "14", kf_dir / "_raw")
        shutil.copy(files[0], target)
        print(f"[keyframe {sc_id}] -> {target}")


def stage_segments(args, sb, comfy):
    wf_t = load_json(HERE / "workflows" / "s2v_segment_api.json")
    segs = sb["segments"]
    todo = parse_range(args.segments, len(segs))
    lip_audio = args.vocals or args.audio
    wav = args.out / "_lip_audio.wav"
    args.out.mkdir(parents=True, exist_ok=True)
    ffmpeg("-i", lip_audio, "-ac", "1", "-ar", "16000", wav)
    audio_name = comfy.upload(wav, "god_inside_lip_audio.wav")
    uploaded_kf = {}

    for seg in segs:
        sid = seg["id"]
        if sid not in todo:
            continue
        d = seg_dir(args.out, sid)
        if seg_frames(args.out, sid) and not args.force:
            print(f"[seg {sid:02d}] done, skipping")
            continue
        dur = seg["end"] - seg["start"]
        need = round(dur * FPS)                 # frames that must exist for this window (80 for 5.0 s)
        latent_frames = math.ceil(need / 4) * 4  # S2V consumes audio in groups of 4 frames
        length = latent_frames - 3               # Wan VAE: 1 + 4*(n-1) decoded frames

        wf = json.loads(json.dumps(wf_t))
        # reference image: scene keyframe if generated/provided, else the character image
        kf = args.out / "keyframes" / f"scene_{seg['scene']}.png"
        ref = kf if kf.exists() else Path(args.image)
        if ref not in uploaded_kf:
            uploaded_kf[ref] = comfy.upload(ref, f"god_inside_ref_{ref.stem}{ref.suffix}")
        wf["10"]["inputs"]["image"] = uploaded_kf[ref]

        wf["7"]["inputs"]["audio"] = audio_name
        wf["8"]["inputs"].update(start_index=float(seg["start"]), duration=latent_frames / FPS)
        wf["11"]["inputs"]["text"] = seg["prompt"].replace("{CHARACTER}", sb["character"])
        wf["12"]["inputs"]["text"] = sb["negative_prompt"]
        wf["15"]["inputs"].update(width=args.width, height=args.height, length=length)
        wf["16"]["inputs"]["seed"] = args.seed + sid
        wf["18"]["inputs"]["filename_prefix"] = f"god_inside/seg_{sid:02d}/frame"

        if args.quality:
            del wf["2"]
            wf["3"]["inputs"]["model"] = ["1", 0]
            wf["16"]["inputs"].update(steps=args.steps or 20, cfg=args.cfg or 6.0)
        else:
            wf["16"]["inputs"].update(steps=args.steps or 4, cfg=args.cfg or 1.0)

        prev = seg_frames(args.out, sid - 1)
        if seg["continuity"] == "continue" and prev and not args.no_continuity:
            mp4 = args.out / "_motion" / f"prev_{sid:02d}.mp4"
            mp4.parent.mkdir(exist_ok=True)
            tail = prev[-73:]
            lst = mp4.with_suffix(".txt")
            lst.write_text("".join(f"file '{p.resolve()}'\nduration {1 / FPS}\n" for p in tail))
            ffmpeg("-f", "concat", "-safe", "0", "-i", lst, "-r", FPS, "-c:v", "libx264", "-crf", "12",
                   "-pix_fmt", "yuv420p", mp4)
            wf["13"]["inputs"]["file"] = comfy.upload(mp4, f"god_inside_prev_{sid:02d}.mp4")
        else:
            del wf["13"], wf["14"], wf["15"]["inputs"]["ref_motion"]

        print(f"[seg {sid:02d}] {seg['start']:6.1f}-{seg['end']:6.1f}s  {seg['section']:<18} \"{seg['lyric']}\"", flush=True)
        t0 = time.time()
        raw = comfy.download_images(comfy.run(wf), "18", d / "_raw")
        # Frame alignment: the first latent decodes to 1 frame but covers 4 audio frames,
        # so hold frame 0 for 4 frames. Then trim to exactly `need` frames (5.000 s).
        frames = [raw[0]] * 3 + raw
        frames = frames[:need]
        if len(frames) < need:
            frames += [frames[-1]] * (need - len(frames))
        for i, f in enumerate(frames):
            shutil.copy(f, d / f"f_{i:04d}.png")
        shutil.rmtree(d / "_raw", ignore_errors=True)
        # preview clip with the real song audio for quick review
        ffmpeg("-framerate", FPS, "-i", d / "f_%04d.png", "-ss", seg["start"], "-t", dur, "-i", args.audio,
               "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
               args.out / "segments" / f"seg_{sid:02d}_preview.mp4")
        print(f"[seg {sid:02d}] {len(frames)} frames in {time.time() - t0:.0f}s")


def stage_assemble(args, sb, _comfy=None):
    allf = args.out / "_all_frames"
    shutil.rmtree(allf, ignore_errors=True)
    allf.mkdir(parents=True)
    n = 0
    for seg in sb["segments"]:
        frames = seg_frames(args.out, seg["id"])
        if not frames:
            sys.exit(f"segment {seg['id']} has not been rendered yet")
        for f in frames:
            n += 1
            (allf / f"{n:06d}.png").symlink_to(f.resolve())
    out = args.out / "god_inside_music_video.mp4"
    vf = []
    if args.interp_fps:
        vf = ["-vf", f"minterpolate=fps={args.interp_fps}:mi_mode=mci:mc_mode=aobmc:vsbmc=1"]
    ffmpeg("-framerate", FPS, "-i", allf / "%06d.png", "-i", args.audio, *vf,
           "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", out)
    print(f"{n} frames ({n / FPS:.2f}s) -> {out}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["keyframes", "segments", "assemble", "all"])
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--storyboard", default=HERE / "storyboard.json", type=Path)
    ap.add_argument("--audio", required=True, help="full song mix (used for the final soundtrack)")
    ap.add_argument("--vocals", help="isolated vocal stem (strongly recommended for accurate lip sync)")
    ap.add_argument("--image", help="front-facing character portrait (identity reference)")
    ap.add_argument("--sheet", help="character sheet image (extra reference for keyframes)")
    ap.add_argument("--out", default=HERE / "render", type=Path)
    ap.add_argument("--segments", help="subset, e.g. '1-5,10'")
    ap.add_argument("--scenes", help="keyframe subset, e.g. 'A,D'")
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--kf-width", type=int, default=1344)
    ap.add_argument("--kf-height", type=int, default=768)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--quality", action="store_true", help="no speed LoRA: 20 steps, cfg 6 (much slower)")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--cfg", type=float)
    ap.add_argument("--no-continuity", action="store_true", help="never feed the previous clip as motion reference")
    ap.add_argument("--interp-fps", type=int, help="optional ffmpeg motion interpolation for the final video, e.g. 24 or 32")
    ap.add_argument("--force", action="store_true", help="re-render even if output exists")
    args = ap.parse_args()
    if args.scenes:
        args.scenes = set(args.scenes.split(","))

    sb = load_json(args.storyboard)
    if "REPLACE ME" in sb["character"]:
        sys.exit("Edit storyboard.json -> 'character' with a description from your character sheet first.")
    if args.stage in ("keyframes", "segments", "all") and not args.image:
        sys.exit("--image is required")
    comfy = Comfy(args.server)
    if args.stage in ("keyframes", "all"):
        stage_keyframes(args, sb, comfy)
    if args.stage in ("segments", "all"):
        stage_segments(args, sb, comfy)
    if args.stage in ("assemble", "all"):
        stage_assemble(args, sb)


if __name__ == "__main__":
    main()
