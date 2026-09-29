#!/usr/bin/env python3
"""Cut the 27 LTX-2 scene clips to the script timeline and lay the full song underneath.

  python3 tools/assemble_video.py --clips <ComfyUI>/output/god_inside [--out God_Inside_music_video.mp4]

A missing clip falls back to its start frame (input/god_inside_scene_XX.png) or the reference photo
with a slow push-in, so you can preview the edit (an animatic) before every scene is rendered.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ffmpeg_bin():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("ffmpeg not found: install ffmpeg or `pip install imageio-ffmpeg`")


def secs(t):
    m, s = t.split(":")
    return int(m) * 60 + float(s)


def find_clip(clips_dir, i):
    hits = sorted(glob.glob(os.path.join(clips_dir, f"clip_{i:02d}*.mp4")))
    return hits[-1] if hits else None


def timeline(scenes):
    t, out = 0.0, []
    for sc in scenes:
        dur = round(secs(sc["end"]) - secs(sc["start"]))
        start = sc.get("place_at", t)
        if abs(start - t) > 1e-6:
            raise SystemExit(f"scene {sc['id']} starts at {start}s but the previous scene ends at {t}s")
        out.append((sc, start, dur))
        t = start + dur
    return out, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=os.path.join(ROOT, "output", "clips"))
    ap.add_argument("--song", default=os.path.join(ROOT, "input", "God_Inside.mp3"))
    ap.add_argument("--stills", default=os.path.join(ROOT, "input"))
    ap.add_argument("--out", default=os.path.join(ROOT, "output", "God_Inside_music_video.mp4"))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--fps", type=int, default=24)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(ROOT, "scenes.json")))
    plan, total = timeline(cfg["scenes"])
    W, H, F = args.width, args.height, args.fps
    cmd, filters, labels = [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error"], [], []
    fit = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1"
    missing = []
    for k, (sc, start, dur) in enumerate(plan):
        i = sc["id"]
        clip = find_clip(args.clips, i)
        if clip:
            cmd += ["-i", clip]
            filters.append(f"[{k}:v]{fit},fps={F},trim=duration={dur},setpts=PTS-STARTPTS,"
                           f"tpad=stop_mode=clone:stop_duration={dur}[v{k}]")
        else:
            still = os.path.join(args.stills, f"god_inside_scene_{i:02d}.png")
            if not os.path.exists(still):
                still = os.path.join(ROOT, "input", "god_inside_ref_closeup.png")
            missing.append(i)
            cmd += ["-loop", "1", "-framerate", str(F), "-t", str(dur), "-i", still]
            frames = dur * F
            filters.append(f"[{k}:v]scale={W * 2}:{H * 2}:force_original_aspect_ratio=increase,crop={W * 2}:{H * 2},"
                           f"zoompan=z='1+0.08*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                           f"d=1:s={W}x{H}:fps={F},setsar=1,trim=duration={dur},setpts=PTS-STARTPTS[v{k}]")
        labels.append(f"[v{k}]")
    n = len(plan)
    cmd += ["-i", args.song]
    filters.append("".join(labels) + f"concat=n={n}:v=1:a=0,fade=t=in:st=0:d=1,"
                   f"fade=t=out:st={total - 2.5}:d=2.5,format=yuv420p[v]")
    filters.append(f"[{n}:a]apad,atrim=0:{total},afade=t=out:st={total - 1.5}:d=1.5[a]")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cmd += ["-filter_complex", ";".join(filters), "-map", "[v]", "-map", "[a]", "-r", str(F),
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-c:a", "aac", "-b:a", "256k",
            "-movflags", "+faststart", args.out]
    subprocess.run(cmd, check=True)
    print(f"wrote {args.out} ({total:.2f}s, {n} scenes)")
    if missing:
        print("scenes without a rendered clip (used their still instead):", ", ".join(map(str, missing)))


if __name__ == "__main__":
    main()
