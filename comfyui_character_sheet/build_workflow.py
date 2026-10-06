"""Build (and optionally run) a ComfyUI workflow: 1 photo -> 8K character sheet.

    python build_workflow.py --photo me.jpg            # writes character_sheet_8k_api.json
    python build_workflow.py --photo me.jpg --submit   # also uploads + queues on a local ComfyUI

Pipeline
  1. IPAdapter FaceID Plus v2 locks the identity from the photo.
  2. OpenPose ControlNet (pose_sheet.png) lays out 4 turnarounds + 3 busts.
  3. SDXL renders the sheet at 1344x768.
  4. 4x model upscale -> 3840x2160 (4K).
  5. UltimateSDUpscale (tiled img2img, low denoise) 2x -> 7680x4320 (8K).
"""
import argparse
import json
import os
import sys
import time

import requests

POSITIVE = (
    "character design reference sheet, turnaround model sheet, same character in every panel, "
    "full body front view, three-quarter view, side view and back view, plus three head-and-shoulders "
    "close-ups with different expressions, consistent outfit and proportions, neutral plain light grey "
    "background, clean layout, {style}, highly detailed face, detailed fabric and accessories, "
    "sharp focus, professional concept art{extra}"
)
NEGATIVE = (
    "different person, inconsistent outfit, extra limbs, extra fingers, deformed hands, bad anatomy, "
    "blurry, lowres, jpeg artifacts, watermark, text, logo, cropped, busy background, duplicate panels"
)
TILE_PROMPT = "highly detailed, sharp focus, fine skin and fabric texture, clean lines, {style}"


def build(a):
    n = {}
    n["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": a.ckpt}}
    n["2"] = {"class_type": "LoadImage", "inputs": {"image": a.photo_name}}
    n["3"] = {"class_type": "LoadImage", "inputs": {"image": "pose_sheet.png"}}
    n["4"] = {"class_type": "IPAdapterUnifiedLoaderFaceID", "inputs": {
        "model": ["1", 0], "preset": "FACEID PLUS V2", "lora_strength": 0.6, "provider": "CUDA"}}
    n["5"] = {"class_type": "IPAdapterFaceID", "inputs": {
        "model": ["4", 0], "ipadapter": ["4", 1], "image": ["2", 0], "weight": a.identity,
        "weight_faceidv2": 1.0, "weight_type": "linear", "combine_embeds": "concat",
        "start_at": 0.0, "end_at": 1.0, "embeds_scaling": "V only"}}
    n["6"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": POSITIVE.format(
        style=a.style, extra=(", " + a.extra) if a.extra else "")}}
    n["7"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": NEGATIVE}}
    n["8"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": a.controlnet}}
    n["9"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
        "positive": ["6", 0], "negative": ["7", 0], "control_net": ["8", 0], "image": ["3", 0],
        "strength": 0.8, "start_percent": 0.0, "end_percent": 0.85, "vae": ["1", 2]}}
    n["10"] = {"class_type": "EmptyLatentImage", "inputs": {"width": 1344, "height": 768, "batch_size": 1}}
    n["11"] = {"class_type": "KSampler", "inputs": {
        "model": ["5", 0], "positive": ["9", 0], "negative": ["9", 1], "latent_image": ["10", 0],
        "seed": a.seed, "steps": 35, "cfg": 5.0, "sampler_name": "dpmpp_2m_sde",
        "scheduler": "karras", "denoise": 1.0}}
    n["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}}
    n["13"] = {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": "charsheet/base_1344"}}
    n["14"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": a.upscaler}}
    n["15"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["14", 0], "image": ["12", 0]}}
    n["16"] = {"class_type": "ImageScale", "inputs": {
        "image": ["15", 0], "upscale_method": "lanczos", "width": 3840, "height": 2160, "crop": "disabled"}}
    n["17"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": TILE_PROMPT.format(style=a.style)}}
    n["18"] = {"class_type": "UltimateSDUpscale", "inputs": {
        "image": ["16", 0], "model": ["1", 0], "positive": ["17", 0], "negative": ["7", 0], "vae": ["1", 2],
        "upscale_by": 2.0, "seed": a.seed, "steps": 20, "cfg": 5.0, "sampler_name": "dpmpp_2m_sde",
        "scheduler": "karras", "denoise": 0.22, "upscale_model": ["14", 0], "mode_type": "Linear",
        "tile_width": 1024, "tile_height": 1024, "mask_blur": 16, "tile_padding": 64,
        "seam_fix_mode": "Half Tile", "seam_fix_denoise": 0.3, "seam_fix_width": 64,
        "seam_fix_mask_blur": 8, "seam_fix_padding": 32, "force_uniform_tiles": True, "tiled_decode": True}}
    n["19"] = {"class_type": "SaveImage", "inputs": {"images": ["18", 0], "filename_prefix": "charsheet/sheet_8k_7680x4320"}}
    return n


def submit(url, photo, workflow, pose):
    for path in (photo, pose):
        with open(path, "rb") as f:
            r = requests.post(f"{url}/upload/image", files={"image": (os.path.basename(path), f)},
                              data={"overwrite": "true"})
        r.raise_for_status()
    pid = requests.post(f"{url}/prompt", json={"prompt": workflow}).json()
    if "prompt_id" not in pid:
        sys.exit(f"ComfyUI rejected the workflow: {json.dumps(pid, indent=2)}")
    pid = pid["prompt_id"]
    print("queued", pid)
    while True:
        h = requests.get(f"{url}/history/{pid}").json()
        if pid in h:
            for out in h[pid]["outputs"].values():
                for im in out.get("images", []):
                    print("output:", im["subfolder"], im["filename"])
            return
        time.sleep(5)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--photo", required=True, help="path to the single source photo (clear, front-facing face)")
    p.add_argument("--out", default="character_sheet_8k_api.json")
    p.add_argument("--ckpt", default="juggernautXL_ragnarokBy.safetensors")
    p.add_argument("--controlnet", default="OpenPoseXL2.safetensors")
    p.add_argument("--upscaler", default="4x-UltraSharp.pth")
    p.add_argument("--style", default="realistic photographic character design")
    p.add_argument("--extra", default="", help="extra prompt text, e.g. outfit description")
    p.add_argument("--identity", type=float, default=0.85, help="IPAdapter FaceID weight")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--server", default="http://127.0.0.1:8188")
    a = p.parse_args()
    a.photo_name = os.path.basename(a.photo)
    wf = build(a)
    with open(a.out, "w") as f:
        json.dump(wf, f, indent=2)
    print("wrote", a.out)
    if a.submit:
        submit(a.server, a.photo, wf, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_sheet.png"))


if __name__ == "__main__":
    main()
