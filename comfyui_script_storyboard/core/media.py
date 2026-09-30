"""ffmpeg helpers, dialogue TTS, lip-sync backends and final video assembly."""

from __future__ import annotations

import asyncio
import csv
import glob
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import wave
from typing import Callable, List, Optional

from .project import Project
from .shots import Shot


# ---------------------------------------------------------------- ffmpeg
def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError("ffmpeg not found. Install ffmpeg or `pip install imageio-ffmpeg`.")


def run(cmd: List[str], cwd: Optional[str] = None, timeout: Optional[float] = None):
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=timeout)
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="replace")[-2000:]
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd[:6])} ...\n{tail}")
    return proc


def wav_duration(path: str) -> Optional[float]:
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return None


def media_duration(path: str) -> Optional[float]:
    if path.lower().endswith(".wav"):
        d = wav_duration(path)
        if d:
            return d
    try:
        proc = subprocess.run([ffmpeg_exe(), "-i", path], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        import re
        m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", proc.stderr.decode(errors="replace"))
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    except Exception:
        pass
    return None


def to_wav(src: str, dst: str, sample_rate: int = 24000):
    run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", str(sample_rate), dst])


def write_wav(path: str, samples, sample_rate: int):
    """samples: float array [-1,1], shape (n,) or (channels, n)."""
    import numpy as np
    a = np.asarray(samples, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    a = np.clip(a, -1, 1)
    pcm = (a.T * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(a.shape[0])
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())


def read_wav(path: str):
    import numpy as np
    with wave.open(path, "rb") as w:
        n, ch, sw, sr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError("Only 16-bit wav supported")
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return a.reshape(-1, ch).T, sr


# ---------------------------------------------------------------- TTS
TTS_BACKENDS = ("edge-tts", "pyttsx3", "command", "existing-files-only")


def synthesize(text: str, voice: str, out_wav: str, backend: str = "edge-tts",
               command_template: str = "", rate: str = "+0%"):
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    if backend == "edge-tts":
        try:
            import edge_tts
        except ImportError as e:
            raise ImportError("edge-tts backend needs `pip install edge-tts` (online, free).") from e
        tmp = out_wav[:-4] + ".mp3"

        async def _go():
            await edge_tts.Communicate(text, voice or "en-US-GuyNeural", rate=rate).save(tmp)
        asyncio.run(_go())
        to_wav(tmp, out_wav)
        os.remove(tmp)
    elif backend == "pyttsx3":
        import pyttsx3
        eng = pyttsx3.init()
        if voice:
            for v in eng.getProperty("voices"):
                if voice.lower() in (v.id.lower(), (v.name or "").lower()):
                    eng.setProperty("voice", v.id)
        tmp = out_wav[:-4] + ".tmp.wav"
        eng.save_to_file(text, tmp)
        eng.runAndWait()
        to_wav(tmp, out_wav)
        os.remove(tmp)
    elif backend == "command":
        if not command_template:
            raise ValueError("command backend needs a command template")
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(text)
            text_file = f.name
        try:
            cmd = command_template.format(
                text=shlex.quote(text), text_file=shlex.quote(text_file),
                voice=shlex.quote(voice or ""), output=shlex.quote(out_wav))
            proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.decode(errors="replace")[-1500:])
        finally:
            os.remove(text_file)
        if not os.path.exists(out_wav):
            raise RuntimeError(f"TTS command did not produce {out_wav}")
    else:
        raise ValueError(f"Unknown TTS backend {backend}")


def import_existing_audio(project: Project, folder: str, shot: Shot) -> bool:
    """Looks for user-recorded dialogue named by shot id, shot index or
    <CHARACTER>_<n> (n = that character's nth line, 1-based)."""
    if not folder or not os.path.isdir(folder):
        return False
    cands = [shot.id, f"shot_{shot.index:06d}", str(shot.index)]
    for c in cands:
        for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
            p = os.path.join(folder, c + ext)
            if os.path.exists(p):
                to_wav(p, project.audio_path(shot.index))
                return True
    return False


def generate_dialogue_audio(project: Project, backend: str, start: int = 0, count: int = -1,
                            overwrite: bool = False, command_template: str = "",
                            existing_folder: str = "", voice_overrides: Optional[dict] = None,
                            progress: Optional[Callable[[int, int], None]] = None) -> dict:
    shots = [s for s in project.shots if s.dialogue]
    shots = [s for s in shots if s.index >= start]
    if count >= 0:
        shots = shots[:count]
    made = skipped = failed = 0
    errors = []
    for k, sh in enumerate(shots):
        out = project.audio_path(sh.index)
        if os.path.exists(out) and not overwrite:
            skipped += 1
        else:
            try:
                if import_existing_audio(project, existing_folder, sh):
                    made += 1
                elif backend != "existing-files-only":
                    ch = sh.dialogue["character"]
                    voice = (voice_overrides or {}).get(ch) or sh.dialogue.get("voice") or ""
                    c = project.bible.characters.get(ch)
                    if c and c.voice:
                        voice = (voice_overrides or {}).get(ch) or c.voice
                    synthesize(sh.dialogue["text"], voice, out, backend, command_template)
                    made += 1
                else:
                    skipped += 1
            except Exception as e:  # keep going, report at the end
                failed += 1
                errors.append(f"{sh.id}: {e}")
        if os.path.exists(out):
            d = wav_duration(out)
            if d:
                sh.duration = round(d + 0.35, 2)
        if progress:
            progress(k + 1, len(shots))
    project.save_shots(project.shots)
    return {"generated": made, "skipped": skipped, "failed": failed, "errors": errors[:20]}


# ---------------------------------------------------------------- lip sync
LIPSYNC_BACKENDS = ("wav2lip", "sadtalker", "custom-command")


def lipsync_one(image: str, audio: str, out_mp4: str, backend: str, repo_dir: str = "",
                checkpoint: str = "", command_template: str = "", python: str = "",
                extra_args: str = ""):
    py = python or sys.executable
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    if backend == "wav2lip":
        if not repo_dir or not os.path.exists(os.path.join(repo_dir, "inference.py")):
            raise FileNotFoundError("Set repo_dir to your Wav2Lip checkout (contains inference.py)")
        ckpt = checkpoint or os.path.join(repo_dir, "checkpoints", "wav2lip_gan.pth")
        cmd = [py, "inference.py", "--checkpoint_path", ckpt, "--face", image,
               "--audio", audio, "--outfile", out_mp4, "--static", "True", "--pads", "0", "15", "0", "0"]
        cmd += shlex.split(extra_args)
        run(cmd, cwd=repo_dir)
    elif backend == "sadtalker":
        if not repo_dir or not os.path.exists(os.path.join(repo_dir, "inference.py")):
            raise FileNotFoundError("Set repo_dir to your SadTalker checkout (contains inference.py)")
        rdir = tempfile.mkdtemp(prefix="sadtalker_")
        cmd = [py, "inference.py", "--driven_audio", audio, "--source_image", image,
               "--result_dir", rdir, "--still", "--preprocess", "full", "--enhancer", "gfpgan"]
        if checkpoint:
            cmd += ["--checkpoint_dir", checkpoint]
        cmd += shlex.split(extra_args)
        run(cmd, cwd=repo_dir)
        vids = sorted(glob.glob(os.path.join(rdir, "**", "*.mp4"), recursive=True), key=os.path.getmtime)
        if not vids:
            raise RuntimeError("SadTalker produced no video")
        shutil.move(vids[-1], out_mp4)
        shutil.rmtree(rdir, ignore_errors=True)
    elif backend == "custom-command":
        if not command_template:
            raise ValueError("custom-command backend needs a command template with {image} {audio} {output}")
        cmd = command_template.format(image=shlex.quote(image), audio=shlex.quote(audio),
                                      output=shlex.quote(out_mp4), python=shlex.quote(py))
        proc = subprocess.run(cmd, shell=True, cwd=repo_dir or None,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode(errors="replace")[-1500:])
    else:
        raise ValueError(f"Unknown lip-sync backend {backend}")
    if not os.path.exists(out_mp4):
        raise RuntimeError(f"Lip-sync backend did not produce {out_mp4}")


def lipsync_project(project: Project, backend: str, start: int = 0, count: int = -1,
                    overwrite: bool = False, only_on_screen: bool = True,
                    progress: Optional[Callable[[int, int], None]] = None, **kw) -> dict:
    shots = [s for s in project.shots if s.dialogue and s.index >= start]
    if only_on_screen:
        shots = [s for s in shots if s.dialogue.get("on_screen", True)]
    if count >= 0:
        shots = shots[:count]
    done = skipped = failed = 0
    errors = []
    for k, sh in enumerate(shots):
        out = project.clip_path(sh.index)
        img = project.frame_for_shot(sh)
        aud = project.audio_path(sh.index)
        if os.path.exists(out) and not overwrite:
            skipped += 1
        elif not img or not os.path.exists(aud):
            skipped += 1
            errors.append(f"{sh.id}: missing {'image' if not img else 'audio'}")
        else:
            try:
                lipsync_one(img, aud, out, backend, **kw)
                done += 1
            except Exception as e:
                failed += 1
                errors.append(f"{sh.id}: {e}")
        if progress:
            progress(k + 1, len(shots))
    return {"lipsynced": done, "skipped": skipped, "failed": failed, "errors": errors[:20]}


def frames_to_mp4(frames, fps: float, out_mp4: str, audio_wav: Optional[str] = None):
    """frames: uint8 numpy array (N,H,W,3)."""
    import numpy as np
    frames = np.ascontiguousarray(frames)
    n, h, w, _ = frames.shape
    w2, h2 = w - w % 2, h - h % 2
    frames = frames[:, :h2, :w2]
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w2}x{h2}", "-r", str(fps), "-i", "-"]
    if audio_wav:
        cmd += ["-i", audio_wav, "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out_mp4]
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = proc.communicate(frames.tobytes())
    if proc.returncode != 0:
        raise RuntimeError(err.decode(errors="replace")[-1500:])


# ---------------------------------------------------------------- assembly
def _srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def shot_duration(project: Project, sh: Shot, use_clips: bool) -> float:
    if use_clips and sh.dialogue and os.path.exists(project.clip_path(sh.index)):
        d = media_duration(project.clip_path(sh.index))
        if d:
            return d
    if sh.dialogue and os.path.exists(project.audio_path(sh.index)):
        d = wav_duration(project.audio_path(sh.index))
        if d:
            return round(d + 0.35, 2)
    return sh.duration


def assemble_video(project: Project, out_name: str = "storyboard", width: int = 1280, height: int = 720,
                   fps: int = 24, scene_start: int = 1, scene_end: int = -1,
                   include_audio: bool = True, use_lipsync_clips: bool = True,
                   ken_burns: bool = True, burn_subtitles: bool = False,
                   missing: str = "placeholder",
                   progress: Optional[Callable[[int, int], None]] = None) -> dict:
    """Render every shot to a normalized segment (cached by content), then
    concat. Segments are reused between runs, so re-assembling after fixing a
    few shots only re-encodes those shots."""
    ff = ffmpeg_exe()
    shots = [s for s in project.shots
             if s.scene_index + 1 >= scene_start and (scene_end < 1 or s.scene_index + 1 <= scene_end)]
    seg_dir = project.path("segments")
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(project.path("exports"), exist_ok=True)
    concat_lines, rows, srt, t = [], [], [], 0.0
    missing_count = 0
    scale = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
             f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1")
    for k, sh in enumerate(shots):
        dur = shot_duration(project, sh, use_lipsync_clips)
        clip = project.clip_path(sh.index) if use_lipsync_clips and sh.dialogue else None
        clip = clip if clip and os.path.exists(clip) else None
        img = project.frame_for_shot(sh)
        aud = project.audio_path(sh.index) if include_audio and sh.dialogue else None
        aud = aud if aud and os.path.exists(aud) else None
        src = clip or img
        if src is None:
            missing_count += 1
            if missing == "skip":
                continue
        key = f"{sh.index}_{width}x{height}_{fps}_{int(ken_burns)}_{int(include_audio)}_{dur:.2f}"
        if src:
            key += f"_{int(os.path.getmtime(src))}"
        if aud:
            key += f"_{int(os.path.getmtime(aud))}"
        seg = os.path.join(seg_dir, f"seg_{sh.index:06d}.mp4")
        stamp = seg + ".key"
        fresh = os.path.exists(seg) and os.path.exists(stamp) and open(stamp).read() == key
        if not fresh:
            cmd = [ff, "-y", "-loglevel", "error"]
            # input 0: video (lip-sync clip, still frame, or placeholder card)
            if clip:
                cmd += ["-i", clip]
                vf = f"{scale},fps={fps}"
            elif img and ken_burns and not sh.dialogue:
                frames = max(1, int(round(dur * fps)))
                vf = (f"scale={width * 2}:{height * 2}:force_original_aspect_ratio=decrease,"
                      f"pad={width * 2}:{height * 2}:(ow-iw)/2:(oh-ih)/2:color=black,"
                      f"zoompan=z='1.0+0.08*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                      f":d={frames}:s={width}x{height}:fps={fps},setsar=1")
                cmd += ["-i", img]
            elif img:
                cmd += ["-loop", "1", "-framerate", str(fps), "-t", f"{dur:.3f}", "-i", img]
                vf = f"{scale},fps={fps}"
            else:
                cmd += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i",
                        f"color=c=0x202020:s={width}x{height}:r={fps}"]
                label = f"{sh.id} {sh.shot_size} (not rendered)".replace(":", " ").replace("'", "")
                vf = (f"drawtext=text='{label}':fontcolor=white:fontsize={max(14, height // 30)}"
                      f":x=(w-tw)/2:y=(h-th)/2")
            # input 1: audio (dialogue wav or silence) - every segment gets identical stream layout
            if aud:
                cmd += ["-i", aud]
            else:
                cmd += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
            cmd += ["-map", "0:v", "-map", "1:a", "-vf", vf, "-af", "apad", "-t", f"{dur:.3f}",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-r", str(fps),
                    "-c:a", "aac", "-ar", "48000", "-ac", "2", seg]
            try:
                run(cmd)
            except RuntimeError:
                if "drawtext" in vf:  # ffmpeg built without freetype
                    cmd[cmd.index("-vf") + 1] = "null"
                    run(cmd)
                else:
                    raise
            with open(stamp, "w") as f:
                f.write(key)
        concat_lines.append(f"file '{seg}'")
        rows.append([sh.index, sh.id, sh.scene_index + 1, sh.heading, sh.kind, sh.shot_size,
                     f"{t:.3f}", f"{dur:.3f}", os.path.basename(src) if src else "",
                     sh.dialogue["character"] if sh.dialogue else "",
                     sh.dialogue["text"] if sh.dialogue else "", sh.prompt])
        if sh.dialogue:
            srt.append(f"{len(srt) + 1}\n{_srt_time(t)} --> {_srt_time(t + dur)}\n"
                       f"{sh.dialogue['character']}: {sh.dialogue['text']}\n")
        t += dur
        if progress:
            progress(k + 1, len(shots))
    if not concat_lines:
        raise RuntimeError("Nothing to assemble - no shots in range")
    base = project.path("exports", out_name)
    list_file = base + "_concat.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        f.write("\n".join(concat_lines))
    with open(base + ".srt", "w", encoding="utf-8") as f:
        f.write("\n".join(srt))
    with open(base + "_edit_list.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["shot_index", "shot_id", "scene", "heading", "kind", "size", "start_s",
                     "duration_s", "media", "character", "dialogue", "prompt"])
        wr.writerows(rows)
    out = base + ".mp4"
    cmd = [ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", list_file]
    if burn_subtitles and srt:
        sub = (base + ".srt").replace("\\", "/").replace(":", r"\:")
        cmd += ["-vf", f"subtitles='{sub}'", "-c:v", "libx264", "-crf", "20", "-c:a", "copy"]
    else:
        cmd += ["-c", "copy"]
    cmd += ["-movflags", "+faststart", out]
    run(cmd)
    return {"video": out, "srt": base + ".srt", "edit_list": base + "_edit_list.csv",
            "shots": len(rows), "missing_frames": missing_count, "duration_s": round(t, 2)}
