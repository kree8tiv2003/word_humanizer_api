# ComfyUI Music Video: lip-synced storyboard to video

This project turns a song, a storyboard (about 40 images) and an optional script into a lip-synced music video. It uses **Wan2.2-S2V 14B** (sound-to-video), which ComfyUI supports natively. The song is split into 5-second segments. Each segment is animated from a storyboard image and driven by that segment's audio. When the last segment finishes, the clips are joined frame-exactly and laid over your original song. The song itself is never cut or re-encoded per segment, so there are no clicks or drift at segment boundaries.

```
comfyui_music_video/
├── workflows/
│   ├── music_video_lipsync.json       ← drag this into ComfyUI
│   └── music_video_lipsync_api.json   ← API format (used by tools/queue_all.py)
├── ComfyUI-MusicVideoKit/             ← custom nodes: copy to ComfyUI/custom_nodes/
└── tools/
    ├── queue_all.py                   ← optional headless upload + queue everything
    └── build_workflows.py             ← regenerates both workflow JSONs
```

## 1. Install (fresh ComfyUI)

The installer sets everything up in one step. It downloads **only the 5 models this workflow needs, about 23.5 GB total**. Files you already have are skipped, an interrupted download picks up where it stopped when you re-run the installer, and it checks free space **before** downloading anything.

**Windows (portable or git install):** download this `comfyui_music_video` folder, then drag your ComfyUI folder onto **`install_windows.bat`**. You can also run it from a command prompt:

```bat
install_windows.bat "C:\ComfyUI_windows_portable\ComfyUI"
install_windows.bat "C:\ComfyUI_windows_portable\ComfyUI" --models-dir "D:\AI\models"   :: models on another drive
```

**Any OS:**

```bash
python install_music_video.py --comfyui /path/to/ComfyUI --dry-run   # preview: sizes and free space
python install_music_video.py --comfyui /path/to/ComfyUI
```

| Option | What it does |
|---|---|
| `--models-dir D:\AI\models` | Keeps the models on another drive and adds that folder to `extra_model_paths.yaml`. For the ComfyUI Desktop app, add the folder in its settings instead. |
| `--no-lora` | Skips the 1.1 GB speed LoRA. Bypass the LoRA node and use steps 20 / cfg 6. |
| `--with-claude` | Installs `anthropic` so Claude can write the script from your storyboard. You also need `ANTHROPIC_API_KEY` set. |
| `--skip-models` | Installs only the nodes and the workflow. |

After it finishes, restart ComfyUI and open **music_video_lipsync** from the **Workflows** sidebar.

<details><summary>Manual install / model list</summary>

1. Use a recent ComfyUI. It needs the built-in `WanSoundImageToVideo` and `AudioEncoderLoader` nodes.
2. Copy `ComfyUI-MusicVideoKit` into `ComfyUI/custom_nodes/`.
3. Download the models:

| File | Folder | Size | Source |
|---|---|---|---|
| `wan2.2_s2v_14B_fp8_scaled.safetensors` | `models/diffusion_models` | 15.3 GB | [Comfy-Org/Wan_2.2_ComfyUI_Repackaged](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/diffusion_models) |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `models/text_encoders` | 6.3 GB | [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/text_encoders) |
| `wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors` | `models/loras` | 1.1 GB | Wan_2.2 repo, `split_files/loras` |
| `wav2vec2_large_english_fp16.safetensors` | `models/audio_encoders` | 0.6 GB | Wan_2.2 repo, `split_files/audio_encoders` |
| `wan_2.1_vae.safetensors` | `models/vae` | 0.2 GB | Wan_2.2 repo, `split_files/vae` |

</details>

**Disk space:** besides the models, keep a few GB free for renders. A finished 3-minute video plus its 37 clips is well under 1 GB. The installer won't run if less than 3 GB would be left after downloading. A 24 GB GPU is comfortable at the default 832×480; lower the resolution on the Segment Loader if you have less VRAM.

## 2. Use it in ComfyUI

1. **Load** `workflows/music_video_lipsync.json` by dragging it onto the canvas.
2. **Upload** your files with the buttons on the **🎬 1. Project, uploads & script** node. You can also drag files onto that node.
   - **🎵 Audio.** Upload the **full 3:03 song** (recommended). It gets cut into frame-exact 5-second segments, and the last one is 3 seconds. You can upload your 5-second clips instead. They're used in filename order (`001.wav`, `002.wav`, …), so zero-pad the numbers.
   - **📤 Storyboard images.** Name them in story order (`01.png` … `40.png`).
   - **📝 Script (optional).** See the formats below.
   - You can select more than 50 files at once. They're sent in batches of 50, and the server rejects any single batch larger than 50.
   - Files go to `ComfyUI/input/music_video/<project>/{images,audio,script}`. Change `project` to keep several videos apart.
3. **Choose where the prompts come from** with `script_source`:
   - `auto` (default): your uploaded script if there is one; otherwise **Claude** reads all the storyboard images and writes one prompt per segment, choosing which image goes where (only if `ANTHROPIC_API_KEY` is set); otherwise template prompts.
   - `claude_vision` / `uploaded_script` / `template` force one mode.
   - Paste your **lyrics** in the `lyrics` box. They're spread across the segments and written into the prompts.
   - The Claude call uses `claude-opus-5-5` with the API's server-side refusal fallback enabled. The result is cached, so you're charged once, not once per segment. Tick `regenerate` to get a new version.
4. **Render.** On **🎬 2. Segment loader**, leave `segment_index = 0` with **increment**. Set the queue count to the number of segments, which is **37** for 3:03 (the exact number is in the console log and in `script_used.json`), then press **Queue**.
5. **Result.** After the last segment, **🎬 3. Save segment** writes
   `ComfyUI/output/music_video/<project>/<project>_final.mp4` (16 fps, H.264 + AAC). The individual clips are in `…/segments/seg_000.mp4`, and so on.

To re-render one bad segment, set `segment_index` to that number, set the control to `fixed`, and queue once. The final video is rebuilt automatically. You can also unmute (Ctrl+M) the **Re-stitch final video** node and queue.

### Script formats

The script has one entry per segment, in order. The storyboard image is optional; if you leave it out, images are spread evenly across the song.

```text
# script.txt: one line per segment, "image | prompt" or just the prompt
01.png | Close-up of the singer at dawn on the rooftop, singing softly, wind in her hair
01.png | Same shot, camera slowly pushes in as she opens her eyes
02.png | Wide shot of the band in the warehouse, chorus hits, confident energy
```

- `.csv` needs a header row with columns `segment,image,prompt,lyric`. Only `prompt` is required.
- `.json` can be a list of strings, or `[{"segment": 0, "image": "01.png", "prompt": "…"}]`.

Each run also writes `output/music_video/<project>/script_used.json`, the full plan with the image, prompt, timings and frame counts for each segment. You can edit it and upload it as your script.

## 3. How the lip-sync and seamless timing work

- **Audio-driven generation.** Each segment's own audio is encoded with wav2vec2 and drives Wan2.2-S2V. The mouth, face and body move to the vocals, not to a guessed animation.
- **Frame-exact timeline.** Segment boundaries are rounded on the song's absolute timeline at 16 fps (the model's native rate). The segments add up to exactly `song_length × 16` frames (183 s → 2928 frames), so there's no cumulative drift.
- **Exact trimming.** The sampler renders a few frames more than needed, using a `4n+1` length and the official "first-frame VAE fix". The save node then keeps exactly the frames for that segment.
- **One uncut audio track.** The final video uses your original song, not re-joined clips.
- **Motion continuity.** When consecutive segments use the same storyboard image, the last 73 frames of the previous clip are fed as reference motion (`continue_motion`). A shot held for 10 or 15 seconds then continues smoothly instead of resetting every 5 seconds. Queue the segments in order for this to work; the increment control does that.
- **Sync nudge.** `sync_offset_ms` on the save node shifts the final audio by ±ms if you ever see a constant offset.

### Quality tips

- **Maximum lip-sync quality.** Bypass the LightX2V LoRA node (Ctrl+B) and set the KSampler to **steps 20, cfg 6**. It's slower but gives crisper mouth shapes. The default of 4 steps with cfg 1 is roughly 5× faster.
- Clear, front- or three-quarter-facing faces in the storyboard give the best lip-sync. Shots without a face still animate, driven by the music.
- Use an isolated vocal stem as the audio if you have one. Wav2vec2 follows voice more precisely without drums and bass. The final video still uses whatever song file you uploaded.
- Match the resolution to your storyboard's aspect ratio. Images are center-cropped to width×height.
- 16 fps is the model's native rate. For a smoother look, put a frame-interpolation node (for example RIFE ×2) between *ImageFromBatch* and the save node, and change `fps` in `script_used.json` to 32 before assembling.

## 4. Headless (optional)

With ComfyUI running:

```bash
python tools/queue_all.py --project my_video --audio song.mp3 --images "storyboard/*.png" \
    --lyrics lyrics.txt [--script script.txt] [--max-quality]
```

This uploads everything in batches of 50, queues segment 0, reads how many segments the song has, and then queues the rest.

## Custom nodes reference

| Node | Purpose |
|---|---|
| **MV Script Builder** | Upload buttons, drag-and-drop, song timeline, script from a file, Claude vision or templates |
| **MV Segment Loader** | Loads the reference image, audio slice, prompts, `length`, exact frame count, reference motion and seed for one segment |
| **MV Save Segment** | Trims to exact frames, saves the clip, and builds the final video after the last segment |
| **MV Assemble Final Video** | Re-stitches manually, with sync offset and CRF |

HTTP routes: `POST /musicvideo/upload` (multipart with `project`, `kind` = images|audio|script, and up to 50 `files`), `GET /musicvideo/status?project=`, `POST /musicvideo/clear`.
