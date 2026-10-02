#!/usr/bin/env python3
"""Download the models these workflows need into a ComfyUI install (stdlib only, resumable).

  python3 tools/download_models.py /path/to/ComfyUI                 # everything (~95 GB)
  python3 tools/download_models.py /path/to/ComfyUI --only upscaler # just the LTX-2 spatial upscaler (~1 GB)
  python3 tools/download_models.py /path/to/ComfyUI --only ltx,qwen

Gated or private repos: set HF_TOKEN in the environment.
"""
import argparse
import os
import sys
import time
import urllib.request

HF = "https://huggingface.co"
MODELS = [
    # group, ComfyUI/models subfolder, repo, path in repo, size in bytes
    ("upscaler", "latent_upscale_models", "Lightricks/LTX-2", "ltx-2-spatial-upscaler-x2-1.0.safetensors", 995765578),
    ("ltx", "diffusion_models", "Kijai/LTXV2_comfy",
     "diffusion_models/ltx-2-19b-distilled_transformer_only_bf16.safetensors", 37759394560),
    ("ltx", "vae", "Kijai/LTXV2_comfy", "VAE/LTX2_video_vae_bf16.safetensors", 2445008522),
    ("ltx", "vae", "Kijai/LTXV2_comfy", "VAE/LTX2_audio_vae_bf16.safetensors", 217740136),
    ("ltx", "text_encoders", "Comfy-Org/ltx-2", "split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors",
     13205434827),
    ("ltx", "text_encoders", "Kijai/LTXV2_comfy",
     "text_encoders/ltx-2-19b-embeddings_connector_distill_bf16.safetensors", 2862983784),
    ("qwen", "diffusion_models", "Comfy-Org/Qwen-Image-Edit_ComfyUI",
     "split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors", 20430698424),
    ("qwen", "loras", "lightx2v/Qwen-Image-Lightning",
     "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors", 849608296),
    ("qwen", "text_encoders", "Comfy-Org/Qwen-Image_ComfyUI",
     "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", 9384670680),
    ("qwen", "vae", "Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors", 253806246),
]


def download(url, dest, size):
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        print(f"  ok      {dest}")
        return
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"User-Agent": "comfyui-music-video/1.0"}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:  # server ignored the range: start over
            have = 0
        mode = "ab" if have else "wb"
        done, t0 = have, time.time()
        with open(part, mode) as f:
            while True:
                chunk = r.read(8 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                rate = (done - have) / max(time.time() - t0, 1e-6) / 1e6
                sys.stdout.write(f"\r  {done / 1e9:6.2f}/{size / 1e9:.2f} GB  {rate:6.1f} MB/s  {os.path.basename(dest)}")
                sys.stdout.flush()
    print()
    if os.path.getsize(part) != size:
        raise SystemExit(f"size mismatch for {dest}: got {os.path.getsize(part)}, expected {size} (re-run to resume)")
    os.replace(part, dest)
    print(f"  saved   {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("comfyui", help="path to your ComfyUI folder (the one containing models/)")
    ap.add_argument("--only", help="comma list of groups: upscaler, ltx, qwen (default: all)")
    args = ap.parse_args()
    groups = set(args.only.split(",")) if args.only else {"upscaler", "ltx", "qwen"}
    models_dir = os.path.join(os.path.abspath(args.comfyui), "models")
    if not os.path.isdir(models_dir):
        raise SystemExit(f"{models_dir} not found: pass the ComfyUI folder that contains models/")
    for group, sub, repo, path, size in MODELS:
        if group not in groups:
            continue
        folder = os.path.join(models_dir, sub)
        os.makedirs(folder, exist_ok=True)
        download(f"{HF}/{repo}/resolve/main/{path}", os.path.join(folder, os.path.basename(path)), size)


if __name__ == "__main__":
    main()
