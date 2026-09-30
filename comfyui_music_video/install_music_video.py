"""One-command setup of the music-video workflow on a fresh ComfyUI install.

    python install_music_video.py --comfyui "C:/path/to/ComfyUI"

Downloads ONLY the 5 models this workflow uses (~23.5 GB), skips files you already
have, resumes interrupted downloads, and refuses to start if the drive is too full.
It also installs the MusicVideoKit custom nodes and puts the workflow in ComfyUI's
Workflows sidebar. Standard library only, so it runs with ComfyUI's embedded Python.

Options:
  --models-dir D:/AI/models   store models elsewhere (e.g. a second drive); writes extra_model_paths.yaml
  --no-lora                   skip the 1.2 GB speed LoRA (then bypass the LoRA node, steps 20 / cfg 6)
  --skip-models               only install nodes + workflow
  --with-claude               also pip-install `anthropic` (for Claude-written scripts)
  --dry-run                   show what would happen, change nothing
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
HF = "https://huggingface.co/Comfy-Org"
GB = 1024 ** 3

# key: (folder, filename, url, exact size in bytes)
MODELS = {
    "s2v": ("diffusion_models", "wan2.2_s2v_14B_fp8_scaled.safetensors",
            f"{HF}/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors", 16394832474),
    "text_encoder": ("text_encoders", "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                     f"{HF}/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", 6735906897),
    "lora": ("loras", "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
             f"{HF}/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors", 1226977424),
    "audio_encoder": ("audio_encoders", "wav2vec2_large_english_fp16.safetensors",
                      f"{HF}/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/audio_encoders/wav2vec2_large_english_fp16.safetensors", 630997322),
    "vae": ("vae", "wan_2.1_vae.safetensors",
            f"{HF}/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors", 253815318),
}
SAFETY_MARGIN = 3 * GB  # room left for renders + temp files


def find_comfyui(given):
    candidates = [given] if given else []
    home = os.path.expanduser("~")
    candidates += [os.getcwd(), os.path.join(os.getcwd(), "ComfyUI"),
                   os.path.join(home, "ComfyUI"), os.path.join(home, "Documents", "ComfyUI"),
                   os.path.join(home, "ComfyUI_windows_portable", "ComfyUI")]
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "custom_nodes")) and os.path.isdir(os.path.join(c, "models")):
            return os.path.abspath(c)
        # user passed the portable root instead of the ComfyUI folder inside it
        if c and os.path.isdir(os.path.join(c, "ComfyUI", "custom_nodes")):
            return os.path.abspath(os.path.join(c, "ComfyUI"))
    return None


def human(n):
    return f"{n / GB:.1f} GB" if n >= GB else f"{n / 1024 ** 2:.0f} MB"


def download(url, dest, size, dry):
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        print(f"  ✓ already have {os.path.basename(dest)}")
        return
    if dry:
        print(f"  would download {os.path.basename(dest)} ({human(size)})")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    for attempt in range(1, 6):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(url, headers={"User-Agent": "MusicVideoKit-installer"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:  # server ignored the resume request
                    have = 0
                mode = "ab" if have else "wb"
                done, t0, last = have, time.time(), 0
                with open(part, mode) as f:
                    while True:
                        chunk = r.read(8 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 1:
                            speed = (done - have) / max(time.time() - t0, 1e-6)
                            print(f"\r  ↓ {os.path.basename(dest)}: {done * 100 // size:3d}%  "
                                  f"{human(done)}/{human(size)}  {speed / 1024 ** 2:.1f} MB/s   ", end="", flush=True)
                            last = time.time()
            print()
            if os.path.getsize(part) == size:
                os.replace(part, dest)
                print(f"  ✓ {os.path.basename(dest)}")
                return
            print(f"  size mismatch ({os.path.getsize(part)} != {size}), retrying…")
            if os.path.getsize(part) > size:
                os.remove(part)
        except Exception as e:
            print(f"\n  network error ({e}); retry {attempt}/5 in {2 ** attempt}s (download resumes)")
            time.sleep(2 ** attempt)
    raise SystemExit(f"Could not download {url}. Re-run the installer to resume.")


def write_extra_model_paths(comfy, models_dir, dry):
    path = os.path.join(comfy, "extra_model_paths.yaml")
    block = ("music_video_models:\n"
             f"    base_path: {models_dir.replace(os.sep, '/')}\n"
             + "".join(f"    {folder}: {folder}/\n" for folder in sorted({m[0] for m in MODELS.values()})))
    existing = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
    if "music_video_models:" in existing:
        print(f"  ✓ {path} already points at the models folder")
        return
    if dry:
        print(f"  would add to {path}:\n{block}")
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(("\n" if existing and not existing.endswith("\n") else "") + block)
    print(f"  ✓ added models folder to {path}")
    print("    (ComfyUI Desktop app: add the same folder under Settings → Server-Config → model paths instead)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comfyui", help="your ComfyUI folder (the one containing custom_nodes and models)")
    ap.add_argument("--models-dir", help="download models here instead of ComfyUI/models")
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--with-claude", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    comfy = find_comfyui(args.comfyui)
    if not comfy:
        raise SystemExit("Couldn't find ComfyUI. Pass --comfyui \"path/to/ComfyUI\" "
                         "(the folder that contains custom_nodes/ and models/).")
    print(f"ComfyUI: {comfy}")

    # 1) custom nodes
    src = os.path.join(HERE, "ComfyUI-MusicVideoKit")
    dst = os.path.join(comfy, "custom_nodes", "ComfyUI-MusicVideoKit")
    print("\n[1/4] Custom nodes")
    if args.dry_run:
        print(f"  would copy {src} → {dst}")
    else:
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        print(f"  ✓ {dst}")

    # 2) workflow into the Workflows sidebar
    print("\n[2/4] Workflow")
    wf_dir = os.path.join(comfy, "user", "default", "workflows")
    for name in ("music_video_lipsync.json",):
        s = os.path.join(HERE, "workflows", name)
        if args.dry_run:
            print(f"  would copy {name} → {wf_dir}")
        else:
            os.makedirs(wf_dir, exist_ok=True)
            shutil.copy2(s, os.path.join(wf_dir, name))
            print(f"  ✓ {os.path.join(wf_dir, name)}  (open it from the Workflows sidebar)")

    # 3) models
    print("\n[3/4] Models")
    if args.skip_models:
        print("  skipped (--skip-models)")
    else:
        models_root = os.path.abspath(args.models_dir) if args.models_dir else os.path.join(comfy, "models")
        wanted = {k: v for k, v in MODELS.items() if not (args.no_lora and k == "lora")}
        todo = 0
        for folder, fname, _, size in wanted.values():
            p = os.path.join(models_root, folder, fname)
            have = os.path.getsize(p) if os.path.exists(p) else (
                os.path.getsize(p + ".part") if os.path.exists(p + ".part") else 0)
            todo += max(0, size - have) if have != size else 0
        if not args.dry_run:
            os.makedirs(models_root, exist_ok=True)
        free = shutil.disk_usage(models_root if os.path.exists(models_root) else comfy).free
        print(f"  folder: {models_root}")
        print(f"  total for this workflow: {human(sum(v[3] for v in wanted.values()))}; "
              f"still to download: {human(todo)}; free space: {human(free)}")
        if todo + SAFETY_MARGIN > free and args.dry_run:
            print(f"  ⚠ this drive is too small: need {human(todo + SAFETY_MARGIN)} incl. working room")
        if todo + SAFETY_MARGIN > free and not args.dry_run:
            raise SystemExit(f"  ✗ Not enough space: need {human(todo + SAFETY_MARGIN)} "
                             f"(incl. {human(SAFETY_MARGIN)} working room). Free up space or use --models-dir on another drive.")
        for folder, fname, url, size in wanted.values():
            download(url, os.path.join(models_root, folder, fname), size, args.dry_run)
        if args.models_dir:
            write_extra_model_paths(comfy, models_root, args.dry_run)

    # 4) optional python package
    print("\n[4/4] Python packages")
    if args.with_claude:
        py = sys.executable
        portable = os.path.join(os.path.dirname(comfy), "python_embeded", "python.exe")
        if os.path.exists(portable):
            py = portable
        cmd = [py, "-m", "pip", "install", "anthropic"]
        print("  " + " ".join(cmd))
        if not args.dry_run:
            subprocess.check_call(cmd)
    else:
        print("  nothing needed (add --with-claude to let Claude write scripts from your storyboard)")

    print("\nDone. Restart ComfyUI, open 'music_video_lipsync' from the Workflows sidebar, and follow the READ ME note.")
    if args.no_lora:
        print("You skipped the speed LoRA: bypass the LoRA node (Ctrl+B) and set KSampler steps 20, cfg 6.")


if __name__ == "__main__":
    main()
