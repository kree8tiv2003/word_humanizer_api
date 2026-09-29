# God Inside: ComfyUI music video (LTX-2 audio to video)

This folder turns the uploaded LTX-2 audio-to-video workflow, the song `God_Inside.mp3` (3:03.84), the two character reference images and the 27-scene script into a full music video pipeline in ComfyUI.

| File | What it does |
|---|---|
| `workflows/01_ltx2_audio_to_video_FIXED.json` | Your original workflow with every error repaired (see below). It makes one continuous ~46 s performance take. |
| `workflows/02_god_inside_scene_stills.json` | Makes 27 character-consistent start frames with **Qwen-Image-Edit-2509** (+ Lightning 4-step). The singer comes from your references. The venue and the crowd of patrons are generated. Scene 01's empty-stage plate is reused as the venue reference for every later shot. |
| `workflows/03_god_inside_music_video_ltx2.json` | Renders 27 lip-synced LTX-2 clips, one per scene. Each clip gets its own start frame, motion prompt and exact song slice (vocals isolated for lip-sync). Output: `output/god_inside/clip_XX.mp4`. |
| `tools/run_music_video.py` | Headless: runs stills, then clips, then the final MP4 against a running ComfyUI (`--server`). You can pick a stage or scenes, for example `--stage clips --scenes 9-12`. |
| `tools/assemble_video.py` | Cuts clips to the script timecodes and lays the full, untouched song underneath. It adds a 1 s fade-in and fades to black after the last note. Missing clips fall back to their still with a slow push-in (animatic preview). |
| `tools/collect_stills.py` | Copies workflow-02 outputs to `input/god_inside_scene_XX.png` for workflow 03 (UI route). |
| `tools/build_workflows.py` | Regenerates everything from `source/` + `scenes.json`. Edit prompts or timings there, then rebuild. |
| `tools/validate_workflow.py` | Link/slot/subgraph consistency checker. The original fails it; all three built workflows pass. |
| `scenes.json` | Timing map + still prompts + LTX motion prompts for all 27 scenes. |

## Errors fixed in the uploaded workflow
1. **Main "Audio to Video (LTX-2.0)" node was missing its `text`, `start_time` and `end_time` slots.** Its prompt, width, height, crop and FPS were quarantined (`proxyWidgetErrorQuarantine`), so the settings you saw were not the ones being used. All 8 slots are restored and driven by Primitive nodes.
2. **Basic Sampling + the 4 Video Extension subgraphs** declared `noise_seed` / `start_index` / `sampler_name` inputs that the parent nodes never exposed. The parent widget values shifted (`sampler_name = 9 / 18 / 27 / 36`, which is not a sampler name → "value not in list"). The inner TrimAudioDuration / KSamplerSelect / RandomNoise inputs were also left with no source. Those values now live on the inner nodes (`lcm`, seeds 44/42, audio offsets 9/18/27/36 s).
3. CreateVideo FPS changed from `24.2421875` to 24. The preview writers were 7.75 fps GIF; they are now 24 fps MP4. The 4th preview was also missing its FPS link.
4. `divisible_by` changed from 2 to 32 (LTX latent grid). The prompt said "the man"; it now describes the singer. The stale embedded API prompt from an unrelated checkpoint was removed, along with duplicate output link ids. The extension subgraph name typo ("VIdeo Extrndion") is fixed.

## Models (ComfyUI/models/...)
- LTX-2: `diffusion_models/ltx-2-19b-distilled_transformer_only_bf16.safetensors`, `vae/LTX2_video_vae_bf16.safetensors`, `vae/LTX2_audio_vae_bf16.safetensors` (Kijai/LTXV2_comfy), `text_encoders/gemma_3_12B_it_fp8_scaled.safetensors` (Comfy-Org/ltx-2), `text_encoders/ltx-2-19b-embeddings_connector_distill_bf16.safetensors` (Kijai/LTXV2_comfy)
- Qwen: `diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors` (Comfy-Org/Qwen-Image-Edit_ComfyUI), `loras/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors` (lightx2v/Qwen-Image-Lightning), `text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors`, `vae/qwen_image_vae.safetensors` (Comfy-Org/Qwen-Image_ComfyUI)
- Custom nodes: ComfyUI-KJNodes (LTX2_NAG, LTXVChunkFeedForward, ImageResizeKJv2, VAELoaderKJ, LTXVImgToVideoInplaceKJ...). ComfyUI-VideoHelperSuite is optional (only the bypassed previews in workflow 01). Use a ComfyUI build with LTX-2 audio support (LTXVAudioVAEEncode, AudioSeparation, AudioCrop).

## Run it
```bash
# copy input/* into ComfyUI/input, start ComfyUI, then:
python3 tools/run_music_video.py --server http://127.0.0.1:8188
# -> output/stills/*.png, output/clips/clip_XX.mp4, output/God_Inside_music_video.mp4 (3:06, 1280x704, 24 fps)
```
Defaults: 1280x704. Drop Width/Height to 960x544 if VRAM runs out, or if the output is black. Re-roll a single shot by changing its seed (stills) or re-running `--stage clips --scenes N`, then `--stage assemble`.

## Notes
- The script's text describes wavy hair, gold hoops and a satin slip dress. The prompts follow **your uploaded images** instead: black side-parted bob, silver drop earrings, red beaded mermaid gown.
- The scene cut points are the script's estimated timing map (no beat grid was in the file). Nudge `start`/`end` in `scenes.json` and rebuild if a cut should land on a different phrase.
