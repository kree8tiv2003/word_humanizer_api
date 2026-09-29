# "God Inside" — ComfyUI lip-sync music video pipeline

This folder turns **your character image + the song** into a full music video. The song is
split into 5-second segments, and the character lip-syncs the vocal in every shot.

| File | What it is |
|---|---|
| `storyboard.json` | Machine-readable storyboard: 37 × 5 s segments with the lyric heard in each window, section, energy, scene, camera move, performance direction and the final video prompt. |
| `storyboard.md` | The same storyboard as a readable table. |
| `workflows/s2v_segment_api.json` | ComfyUI workflow (API format) that renders **one** 5 s lip-synced segment with **Wan2.2-S2V-14B**. |
| `workflows/keyframe_qwen_edit_api.json` | ComfyUI workflow (API format) that places your character into each scene with **Qwen-Image-Edit-2509** (one keyframe per scene). |
| `make_video.py` | Driver script. It queues every segment into ComfyUI, keeps each clip at exactly 5.000 s, and assembles the final MP4 with the original song. |

## What the song analysis found

* **Length** 3:03.8 → **37 segments** (36 × 5 s + a final 3.8 s).
* **Tempo** ≈ 96 BPM, key estimate G♯/A♭.
* **Arc**: quiet intro (energy 0.31) → verse 1 → chorus 1 (0:45) → verse 2 → chorus 2 (1:45) → bridge (2:10) → final chorus peak (2:35–2:55) → soft outro.
* **Theme**: a seeker who looks for God across forests, mountains and seas, then finds the divine "deep within the temple of a heart". The visuals follow that path, from outer landscapes to an inner temple of light.

Lyrics were transcribed automatically with Whisper. **Check `storyboard.md` against your real lyrics**, especially segment 37 (the final line was unclear) and segments 2–3 (ad-libs).

### Scenes (one keyframe each)

| Scene | Name | Used in |
|---|---|---|
| A | The Void (single shaft of light) | Intro 0:00–0:15 |
| B | Whisper in the Wind (moonlit misty forest) | Verse 1a |
| C | Mountains and the Silver Sea (cliff at dawn) | Verse 1b |
| D | The Temple Within (temple + river of light) | Choruses 1 & 2 |
| E | Dawn of Inner Light (sunrise meadow) | Verse 2a |
| F | Stars and Battle Scars (starfield, glowing kintsugi scars) | Verse 2b |
| G | Walking in Grace (rain turning to light) | Bridge a |
| H | Heartbeat (glow from the chest) | Bridge b |
| I | Temple in Full Bloom | Final chorus |
| J | Return to Stillness | Outro |

## How it works

```
character.png ─┐
character sheet┴─► [Qwen-Image-Edit] ─► scene keyframes A…J (1344×768)
                                               │
song ─► vocal stem ─► TrimAudio(start, 5 s) ─► wav2vec2 ─► [Wan2.2 S2V 14B] ◄─ keyframe + segment prompt
                                                                │   ◄─ previous clip (motion continuity)
                                                                ▼
                                           77 frames ─► hold frame 0 ─► 80 frames = 5.000 s @ 16 fps
                                                                ▼
                                     all 37 segments ─► ffmpeg + original full mix ─► final MP4
```

**Why the timing stays exact:** in ComfyUI's `WanSoundImageToVideo` node, `length = 77` produces 20
latent frames, and those latents take in exactly **80 audio frames at 16 fps = 5.0 s**. The Wan VAE
decodes the first latent as a single frame, so the script repeats frame 0 three times to get back
to 80 frames. Every segment therefore starts exactly on its 5-second boundary, and lip-sync drift
can't build up over the 3-minute song.

**Continuity:** segments marked `"continuity": "continue"` pass the previous clip's last 73 frames
into `ref_motion`, so motion flows through a shot. Segments marked `"cut"` start a new shot on a
new keyframe, which gives the edit its music-video cutting rhythm.

## Setup

1. **ComfyUI**: install a recent version (the Wan2.2 S2V, TrimAudioDuration and Qwen Edit Plus nodes are core, so no custom nodes are needed).
2. **Models**: download into `ComfyUI/models/…`:

| Folder | File | Source (Hugging Face) |
|---|---|---|
| `diffusion_models` | `wan2.2_s2v_14B_fp8_scaled.safetensors` (or `_bf16`) | `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` → `split_files/diffusion_models` |
| `audio_encoders` | `wav2vec2_large_english_fp16.safetensors` | `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` → `split_files/audio_encoders` |
| `text_encoders` | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `Comfy-Org/Wan_2.1_ComfyUI_repackaged` → `split_files/text_encoders` |
| `vae` | `wan_2.1_vae.safetensors` | `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` → `split_files/vae` |
| `loras` | `wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors` | `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` → `split_files/loras` |
| `diffusion_models` | `qwen_image_edit_2509_fp8_e4m3fn.safetensors` | `Comfy-Org/Qwen-Image-Edit_ComfyUI` |
| `text_encoders` | `qwen_2.5_vl_7b_fp8_scaled.safetensors` | `Comfy-Org/Qwen-Image_ComfyUI` |
| `vae` | `qwen_image_vae.safetensors` | `Comfy-Org/Qwen-Image_ComfyUI` |
| `loras` | `Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors` | `lightx2v/Qwen-Image-Lightning` |

   If you already have different versions (e.g. a newer Qwen-Image-Edit), change the filenames in the loader nodes of the two JSON files.

3. **Isolate the vocals** (the biggest lip-sync quality win). With the instruments mixed in,
   wav2vec2 "hears" drums and synths and the mouth moves on beats instead of words. Split
   `God_Inside.mp3` with UVR5 / Demucs (`demucs --two-stems=vocals God_Inside.mp3`) and pass the
   vocal stem with `--vocals`. The final video still uses the full mix.
4. **Describe your character**: open `storyboard.json` and replace the `"character"` value with one
   sentence from your character sheet (hair, eyes, skin tone, outfit, distinguishing marks). It is
   injected into every prompt.
5. **Images**: use a clean, front-facing, well-lit **portrait** as `--image`. The multi-view
   character sheet goes in `--sheet`; it only feeds the keyframe stage. Don't use the sheet itself as
   the S2V reference, or the model will try to animate the whole sheet layout.

## Run

```bash
# ComfyUI running on :8188, ffmpeg on PATH
cd music_video/god_inside

# 1) test one scene keyframe + two segments first
python make_video.py keyframes --audio God_Inside.mp3 --image character.png --sheet sheet.png --scenes B
python make_video.py segments  --audio God_Inside.mp3 --vocals vocals.wav --image character.png --segments 4-5
#    -> check render/segments/seg_04_preview.mp4 (has the real audio)

# 2) everything
python make_video.py all --audio God_Inside.mp3 --vocals vocals.wav --image character.png --sheet sheet.png

# final: render/god_inside_music_video.mp4
```

Useful flags: `--quality` (no speed LoRA: 20 steps, cfg 6, about 5× slower and sharper),
`--width 1280 --height 720` (needs roughly 24 GB+ VRAM), `--segments 10-14 --force` (re-roll
shots you don't like), `--seed N`, `--interp-fps 24` (smooths 16 → 24 fps in ffmpeg; RIFE/GIMM in
ComfyUI gives better results), `--no-continuity`.

Finished segments are skipped on re-run, so you can stop and resume at any point. To swap in a
hand-made keyframe, drop it in at `render/keyframes/scene_X.png`.

### Using the workflow by hand in the ComfyUI UI

Drag `workflows/s2v_segment_api.json` onto the ComfyUI canvas (the UI can open API-format files).
For a single segment, set **Trim Audio Duration → start_index** to the segment start (e.g. `15.0`)
with **duration** `5.0`, load the scene keyframe, and paste the segment prompt from
`storyboard.md`. Delete the two "Previous clip" nodes when there is no previous clip. The frames
come out as 77 PNGs; repeat the first frame 3 times when you edit them together at 16 fps.

## Tips for the best lip sync

* Keep faces at **medium close-up or closer** in vocal segments. The storyboard already does this.
  Wide shots are only used where the voice is soft or absent.
* Keep each keyframe's face **large, front-lit and mouth relaxed/closed**.
* For a shot that goes wrong, re-roll only that segment with another seed:
  `--segments 23 --force --seed 1234`.
