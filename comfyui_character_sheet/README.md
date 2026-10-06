# ComfyUI: photo → 8K character sheet

One photo in, one 7680×4320 character sheet out (4 full-body turnarounds + 3 expression busts).

## Files
- `character_sheet_8k_api.json` – workflow in ComfyUI **API format** (generated for photo name `x.jpg`; regenerate with your photo name).
- `build_workflow.py` – regenerates the workflow with your options; `--submit` uploads and queues it on a running ComfyUI.
- `make_pose_sheet.py` / `pose_sheet.png` – the OpenPose layout guide that fixes the panel layout.

## Requirements
Custom nodes: [ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus) (+ `insightface`),
[ComfyUI_UltimateSDUpscale](https://github.com/ssitu/ComfyUI_UltimateSDUpscale).

Models:
| Type | Default name (change via flags) |
|---|---|
| SDXL checkpoint | `juggernautXL_ragnarokBy.safetensors` |
| OpenPose ControlNet (SDXL) | `OpenPoseXL2.safetensors` |
| Upscaler | `4x-UltraSharp.pth` |
| IPAdapter FaceID Plus v2 SDXL + its LoRA | `ip-adapter-faceid-plusv2_sdxl.bin` / `..._lora.safetensors` (see IPAdapter_plus README for folders) |
| CLIP Vision | `CLIP-ViT-H-14-laion2B-s32B-b79K` |

## Use
```bash
python build_workflow.py --photo me.jpg --extra "red leather jacket, silver rings" --style "anime concept art" --submit
```
Without `--submit`, copy `me.jpg` and `pose_sheet.png` into ComfyUI's `input/` folder and queue the JSON (API-format loading, or Save (API) menu).
Flags: `--ckpt --controlnet --upscaler --identity --seed --server`.

## Notes
- Needs roughly 16 GB VRAM for the 8K stage (tiled, 1024 px tiles; lower `tile_width/height` if you hit OOM). 8K PNG is large (~100+ MB).
- Use a clear, well-lit, front-facing photo; the face identity comes from it. Outfit/body come from your prompt (`--extra`) since one photo can't define the back/sides.
- 8K is reached by upscaling (1344×768 → 4K → 8K with low-denoise tiles), not native generation. Raise `--identity` for likeness, lower it if the face looks pasted.
- Untested here (no GPU/ComfyUI in this environment): node and input names follow current IPAdapter_plus / UltimateSDUpscale; if your versions differ, ComfyUI's error will name the field.
