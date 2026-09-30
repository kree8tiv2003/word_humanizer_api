# ComfyUI Script-to-Storyboard

A ComfyUI custom node pack. It takes a screenplay (up to 1000 pages) and turns it into:

1. **Scenes**: the script is parsed into scenes, action, dialogue and transitions.
2. **A character bible**: every character gets one fixed description, seed and voice. Their look stays the same **unless the script says otherwise**.
3. **Shot prompts**: each scene becomes an establishing shot, action beats and one shot per dialogue line, with consistent character descriptions in every prompt.
4. **Images**: each prompt is rendered with any model (SDXL, Flux, …). Shots render one per queued run, can be resumed, and are guided by up to **5 reference images** you upload.
5. **A sequenced, lip-synced video**: dialogue is voiced (TTS or your own recordings) and speakers are lip-synced (Wav2Lip, SadTalker, any lip-sync CLI, or any ComfyUI lip-sync node). Everything is then assembled in storyboard order into an MP4, with an `.srt` subtitle file and a CSV edit list.

## Install

```bash
cd ComfyUI/custom_nodes
git clone <this repo> && ln -s <this repo>/comfyui_script_storyboard .   # or copy the folder
pip install -r comfyui_script_storyboard/requirements.txt
```

Restart ComfyUI. The nodes appear under **Script2Storyboard**. Example workflows are in `workflows/`; drag them onto the canvas.

## Workflow order

| # | Workflow | What it does |
|---|----------|--------------|
| 01 | `01_plan_storyboard.json` | Upload script → parse → character bible → shot prompts → save project |
| 02 | `02_reference_images.json` | Upload **up to 5 reference images** and bind them to characters, locations, scenes, shots or everything |
| 03 | `03_character_sheets.json` | *(optional)* Generate a reference sheet per character for identity consistency |
| 04 | `04_render_shots_sdxl.json` | Render every shot in order, resumable (SDXL; `reference_images` output for IP-Adapter/PuLID) |
| 05 | `05_render_shots_flux_redux.json` | Same, with Flux and reference images applied automatically through Flux Redux |
| 06 | `06_audio_lipsync_assemble.json` | Dialogue TTS → lip-sync → final video + subtitles + edit list |
| 07 | `07_lipsync_with_comfy_nodes.json` | Lip-sync through any ComfyUI lip-sync node pack instead of a CLI |

Use the **same `project_name`** in every workflow. All stages meet at the project folder `ComfyUI/output/storyboards/<project_name>/`.

### Rendering thousands of shots
A 1000-page script produces roughly 8–15k shots. The **Shot Iterator** in `next_missing` mode renders the next shot that has no image each time the workflow runs. Set **Batch count** to the `shots_remaining` number from **Project Status**, or turn on Auto Queue. You can stop at any time and resume later, even days later. Use `mode = index` to re-render a single shot.

You can also split the work: set `scene_start`/`scene_end` on **Plan Shots** (and on Enhance or Assemble) to handle the script in chunks.

## Script formats
`.fountain`, `.txt`, `.fdx` (Final Draft), `.pdf` (needs `pypdf`), `.docx`, `.md`. Upload with the **upload script** button on the Load Script node, or paste text into `script_text`. Text without `INT./EXT.` headings (treatments, prose) is split into pseudo-scenes.

## Character consistency: "consistent unless otherwise stated"
- Descriptions are pulled from screenplay introductions, e.g. `MARA OKAFOR (30s), a lanky mechanic with a shaved head…`. A cue like `MARA` is linked to her full-name introduction.
- Each character gets a fixed seed and voice, injected into **every** shot they appear in.
- Any field can be overridden in the Character Bible `overrides` box, either as JSON or as one `NAME: description` per line:
  ```json
  {"MARA": {"description": "…", "gender": "female", "age": "35", "voice": "en-GB-SoniaNeural"}}
  ```
- A character's look changes only when the script says so:
  - Prose: `Mara, now wearing a borrowed raincoat, …` or `John changes into a tux.` (lasts until the next change)
  - Fountain notes (most reliable):
    - `[[LOOK JOHN: torn tux, bloodied]]` lasts until changed
    - `[[SCENE LOOK JOHN: soaking wet]]` applies to this scene only
    - `[[LOOK JOHN: reset]]` returns to the canonical look
    - `[[AGE JOHN: 70]]` and `[[DESC JOHN: full new description]]`
  - `YOUNG JOHN` and `OLDER JOHN` cues are linked to `JOHN` and rendered with an age modifier.
- Turn off `consistent_characters` to let each shot describe characters only from its own text.
- Character names are replaced with "the man" / "the woman" in prompts, so image models never draw names as text.

## Reference images (up to 5)
Connect up to five **Load Image** nodes to **Reference Images**. Each slot has:

- **bind**: `all` · `style` · `character:NAME` · `location:DINER` · `scenes:3-10` · `shot:SC0003_SH002` / `shot:42`
- **usage**:
  - `reference`: steers generation for every matching shot. It's applied automatically with Flux Redux (workflow 05), and it's also available as an IMAGE batch from the Shot Iterator for IP-Adapter, PuLID, InstantID or Kontext.
  - `init_image`: img2img starting frame for matching shots, with its own denoise.
  - `use_as_frame`: the uploaded image **is** the storyboard frame for that shot (or for the first shot of each scene in a `scenes:` range). It is placed in the sequence and the video instead of a generated image.
- A `character:` reference also becomes that character's identity reference in every shot they appear in. References persist in the project and survive re-planning.

## Dialogue and lip-sync
- **Dialogue Audio** generates one WAV per dialogue shot with `edge-tts` (free, online), `pyttsx3` (offline) or any CLI (`command`, e.g. Piper). You can also drop in real recordings named `SC0002_SH003.wav` or `shot_000005.wav`. Shot durations then follow the real audio length.
- **Lip-Sync (batch)** supports three backends:
  - `wav2lip`: `repo_dir` = Wav2Lip checkout; uses `checkpoints/wav2lip_gan.pth`
  - `sadtalker`: `repo_dir` = SadTalker checkout
  - `custom-command`: any tool, e.g. `{python} -m scripts.inference --video_path {image} --audio_path {audio} --video_out_path {output}`
- **Lip-Sync Shot Inputs → [your lip-sync node] → Save Lip-Sync Clip**: works with LatentSync, MuseTalk, Sonic, InfiniteTalk and similar node packs.
- Only on-screen speakers are lip-synced. `V.O.` and `O.S.` lines are voiced over a listener or wide shot.

## Final assembly
**Assemble Video** renders every shot to a cached segment and concatenates them in storyboard order:

- Lip-synced clips replace their stills.
- Non-dialogue stills get a slow Ken Burns push-in.
- Missing shots appear as labeled placeholder cards.

Outputs go to `exports/<name>.mp4`, `.srt` and `_edit_list.csv`. Re-assembling after fixing a few shots only re-encodes those shots.

## Optional LLM prompt pass
**Enhance Prompts** rewrites prompts one scene per request, keeping character descriptions word for word. Backends:

- `anthropic`: Claude via the `anthropic` SDK, reading `ANTHROPIC_API_KEY`. The default model is `claude-opus-5-5` at low effort. Refusals fall back to the original prompt. Server-side model fallback is enabled.
- `ollama`: a local model.

Responses are cached per scene, so very long scripts can be enhanced over several sessions.

## Tests
```bash
pip install pytest numpy pillow imageio-ffmpeg torch
python -m pytest comfyui_script_storyboard/tests -q
python comfyui_script_storyboard/tools/build_workflows.py   # regenerate example workflows
```
