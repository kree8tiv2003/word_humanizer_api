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


SKIP_DIRS = {"windows", "$recycle.bin", "system volume information", "node_modules", ".git",
             "site-packages", "__pycache__", "programdata", "recovery", "perflogs", "models",
             "input", "output", "temp", "tmp", "cache", ".cache"}


def is_install(d):
    """A ComfyUI install = a folder with custom_nodes/ plus ComfyUI's own files or user data."""
    if not os.path.isdir(os.path.join(d, "custom_nodes")):
        return False
    # The Desktop app ships a read-only copy of ComfyUI inside its program files; never install there.
    if "resources" in os.path.normpath(d).lower().split(os.sep):
        return False
    return any(os.path.exists(os.path.join(d, x)) for x in ("main.py", "comfy", "user", "models", ".venv"))


def search_installs(roots, max_depth=5, time_budget=45):
    """Look for ComfyUI installs under the given folders (depth-limited, time-limited)."""
    found, t0, seen = [], time.time(), set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        base_depth = os.path.abspath(root).rstrip(os.sep).count(os.sep)
        for cur, dirs, _ in os.walk(root):
            if time.time() - t0 > time_budget:
                break
            key = os.path.normcase(os.path.abspath(cur))
            if key in seen:
                dirs[:] = []
                continue
            seen.add(key)
            if is_install(cur):
                found.append(os.path.abspath(cur))
                dirs[:] = []
                continue
            if cur.count(os.sep) - base_depth >= max_depth:
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIRS and not d.startswith(".")]
    # Most recently used first (custom_nodes changes when you install nodes / update ComfyUI).
    found.sort(key=lambda d: os.path.getmtime(os.path.join(d, "custom_nodes")), reverse=True)
    return found


def default_search_roots():
    home = os.path.expanduser("~")
    roots = [os.getcwd(), os.path.join(home, "Documents"), os.path.join(home, "Desktop"),
             os.path.join(home, "Downloads"), home,
             os.path.join(home, "AppData", "Roaming"), os.path.join(home, "AppData", "Local")]
    if os.name == "nt":
        import string
        roots += [f"{c}:\\" for c in string.ascii_uppercase if os.path.isdir(f"{c}:\\")]
    return roots


def classify_given(path):
    """Returns (install_dir or None, shared_models_dir or None) for whatever folder the user dropped."""
    if not path:
        return None, None
    p = os.path.abspath(path.strip().strip('"'))
    if is_install(p):
        return p, None
    if is_install(os.path.join(p, "ComfyUI")):  # portable root
        return os.path.join(p, "ComfyUI"), None
    if os.path.isdir(os.path.join(p, "models")):  # shared folder: input / models / output
        return None, os.path.join(p, "models")
    if os.path.basename(p).lower() == "models" and os.path.isdir(p):
        return None, p
    return None, None


def choose_install(candidates, interactive):
    if len(candidates) == 1:
        return candidates[0]
    print("  Found more than one ComfyUI install:")
    for i, c in enumerate(candidates, 1):
        print(f"    {i}. {c}")
    if not interactive:
        print("  Using #1 (most recently used). Pass --comfyui to choose another.")
        return candidates[0]
    ans = input("  Type the number to install into [1]: ").strip() or "1"
    return candidates[int(ans) - 1] if ans.isdigit() and 1 <= int(ans) <= len(candidates) else candidates[0]


def ask_for_install(interactive):
    if not interactive:
        return None
    print("\n  I couldn't find the folder that holds ComfyUI's 'custom_nodes' folder.")
    print("  Open File Explorer, find the folder that contains 'custom_nodes', drag it into this")
    print("  window and press Enter. (Press Enter on its own to skip installing the nodes.)")
    while True:
        ans = input("  > ").strip().strip('"')
        if not ans:
            return None
        if os.path.basename(ans.rstrip("\\/")).lower() == "custom_nodes":
            ans = os.path.dirname(ans.rstrip("\\/"))
        if os.path.isdir(os.path.join(ans, "custom_nodes")):
            return os.path.abspath(ans)
        print("  That folder has no 'custom_nodes' inside. Try again, or press Enter to skip.")


def models_configured(install, models_root):
    """True if ComfyUI already loads models from models_root (its own models folder or a configured path)."""
    norm = lambda s: os.path.normcase(os.path.abspath(s)).replace("\\", "/").rstrip("/")
    target = norm(models_root)
    if install and norm(os.path.join(install, "models")) == target:
        return True
    configs = []
    if install:
        configs.append(os.path.join(install, "extra_model_paths.yaml"))
    appdata = os.environ.get("APPDATA")
    if appdata:
        configs.append(os.path.join(appdata, "ComfyUI", "extra_models_config.yaml"))
    for cfg in configs:
        if os.path.exists(cfg):
            text = open(cfg, encoding="utf-8", errors="ignore").read().replace("\\\\", "/").replace("\\", "/")
            if target.lower() in text.lower() or norm(os.path.dirname(models_root)).lower() in text.lower():
                return True
    return False


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
    desktop_cfg = os.path.join(os.environ.get("APPDATA", ""), "ComfyUI", "extra_models_config.yaml")
    # The Desktop app has no main.py in its data folder and reads its own config file instead.
    if os.environ.get("APPDATA") and os.path.exists(desktop_cfg) and not os.path.exists(os.path.join(comfy, "main.py")):
        path = desktop_cfg
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comfyui", help="your ComfyUI folder, the portable root, or a shared folder (input/models/output)")
    ap.add_argument("--models-dir", help="download models here (default: the shared models folder or ComfyUI/models)")
    ap.add_argument("--search", nargs="*", help="folders/drives to search for the ComfyUI install")
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--with-claude", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):  # never crash on ✓/↓ in an old Windows console
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    interactive = sys.stdin is not None and sys.stdin.isatty()

    comfy, shared_models = classify_given(args.comfyui)
    if args.comfyui and not comfy and not shared_models:
        print(f"Note: '{args.comfyui}' has no custom_nodes or models folder inside; searching instead.")
    if shared_models:
        print(f"Shared models folder: {shared_models}")
    if not comfy:
        print("Looking for your ComfyUI install (the folder with custom_nodes)… this can take up to a minute.")
        roots = args.search or default_search_roots()
        if shared_models:
            # launchers usually keep the install next to (or just above) the shared folder
            parent = os.path.dirname(os.path.dirname(shared_models))
            roots = [parent, os.path.dirname(parent)] + roots
        found = search_installs(roots)
        comfy = choose_install(found, interactive) if found else ask_for_install(interactive)
    if comfy:
        print(f"ComfyUI install: {comfy}")
    else:
        print("ComfyUI install: not found - nodes and workflow will be skipped (see the end of this output).")

    # 1) custom nodes
    src = os.path.join(HERE, "ComfyUI-MusicVideoKit")
    print("\n[1/4] Custom nodes")
    if comfy:
        dst = os.path.join(comfy, "custom_nodes", "ComfyUI-MusicVideoKit")
        if args.dry_run:
            print(f"  would copy {src} → {dst}")
        else:
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
            print(f"  ✓ {dst}")
    else:
        print("  skipped")

    # 2) workflow into the Workflows sidebar
    print("\n[2/4] Workflow")
    wf_src = os.path.join(HERE, "workflows", "music_video_lipsync.json")
    if comfy:
        wf_dir = os.path.join(comfy, "user", "default", "workflows")
        if args.dry_run:
            print(f"  would copy music_video_lipsync.json → {wf_dir}")
        else:
            os.makedirs(wf_dir, exist_ok=True)
            shutil.copy2(wf_src, os.path.join(wf_dir, "music_video_lipsync.json"))
            print(f"  ✓ {os.path.join(wf_dir, 'music_video_lipsync.json')}  (open it from the Workflows sidebar)")
    else:
        print(f"  skipped - you can drag {wf_src} onto the ComfyUI canvas instead")

    # 3) models
    print("\n[3/4] Models")
    if args.skip_models:
        print("  skipped (--skip-models)")
    else:
        if args.models_dir:
            models_root = os.path.abspath(args.models_dir)
        elif shared_models:
            models_root = shared_models
        elif comfy:
            models_root = os.path.join(comfy, "models")
        else:
            raise SystemExit("  ✗ Don't know where to put the models: pass --models-dir or --comfyui.")
        wanted = {k: v for k, v in MODELS.items() if not (args.no_lora and k == "lora")}
        todo = 0
        for folder, fname, _, size in wanted.values():
            p = os.path.join(models_root, folder, fname)
            have = os.path.getsize(p) if os.path.exists(p) else (
                os.path.getsize(p + ".part") if os.path.exists(p + ".part") else 0)
            todo += max(0, size - have) if have != size else 0
        if not args.dry_run:
            os.makedirs(models_root, exist_ok=True)
        probe = models_root
        while not os.path.exists(probe):
            probe = os.path.dirname(probe)
        free = shutil.disk_usage(probe).free
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
        if comfy and not models_configured(comfy, models_root):
            write_extra_model_paths(comfy, models_root, args.dry_run)
        elif comfy:
            print("  ✓ ComfyUI already loads models from this folder")

    # 4) optional python package
    print("\n[4/4] Python packages")
    if args.with_claude:
        py = sys.executable
        for cand in ([os.path.join(os.path.dirname(comfy), "python_embeded", "python.exe"),
                      os.path.join(comfy, ".venv", "Scripts", "python.exe"),
                      os.path.join(comfy, ".venv", "bin", "python")] if comfy else []):
            if os.path.exists(cand):
                py = cand
                break
        cmd = [py, "-m", "pip", "install", "anthropic"]
        print("  " + " ".join(cmd))
        if not args.dry_run:
            subprocess.check_call(cmd)
    else:
        print("  nothing needed (add --with-claude to let Claude write scripts from your storyboard)")

    if comfy:
        print("\nDone. Restart ComfyUI, open 'music_video_lipsync' from the Workflows sidebar, and follow the READ ME note.")
    else:
        print("\nModels are done, but the custom nodes still need installing: copy the folder\n"
              f"  {src}\ninto ComfyUI's custom_nodes folder, restart ComfyUI, then drag\n  {wf_src}\nonto the canvas.")
    if args.no_lora:
        print("You skipped the speed LoRA: bypass the LoRA node (Ctrl+B) and set KSampler steps 20, cfg 6.")


if __name__ == "__main__":
    main()
