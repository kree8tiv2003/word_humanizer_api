# Script Studio

Script Studio turns a story, recording, web page or song into a timed shooting script. Every shot comes with a ready-to-paste AI video prompt. Prompts are written with the Director's Manual method from the six prompt manuals in this repository.

## What it does

**Inputs**
- **Documents:** Word (`.docx`), PDF, text, Markdown, HTML, RTF, Fountain, SRT and LRC. Several files are joined in upload order.
- **Pasted text.**
- **Links:** web pages (the article text is extracted), linked PDFs and Word files, and direct audio links.
- **Audio:** narration, audiobook or podcast recordings. The audio is transcribed with timestamps, so the script follows the recording's real pacing.
- **Music video mode:** a song, plus optional lyrics, a script or a concept as a document, link or pasted text. The app analyses tempo, beats, downbeats, sections and energy. Cuts land on downbeats, and the visuals follow the song's structure (intro, verse, chorus, bridge, outro) and its energy.

**Output length:** 5, 10, 15, 30 or 60-second segments, each one a clip span you can generate, or a complete full-length script with scenes of natural length. Shots always fill each segment exactly.

**Every shot includes:**
- the camera technique (from the 94-move camera and advertising library)
- a 40–90 word video prompt: camera first, character looks repeated word for word, motion verbs, setting and light, a layered effect, and the look
- an action line, dialogue with delivery notes, voiceover and lyrics
- sound, VFX and a negative prompt

Every segment ends with a continuity state and a transition that carries it into the next segment.

**Editing:** you can edit any field in place, rewrite a single segment with a note (for example "more aerial shots, add rain"), and resume a job that was interrupted.

**Exports:** PDF screenplay, Fountain (opens in Highland, Fade In, WriterSolo and Final Draft via import), Markdown, shot-list CSV, a prompts-only text file for batch generation, and JSON.

## How it works

1. **Ingest:** the source becomes numbered units: paragraphs, lyric lines or timed transcript segments.
2. **Story bible** (Claude): title, logline, tone and visual style; characters and locations with fixed visual signatures; and story beats. Each beat maps to a range of source units and has an emotion, an intensity (1–5) and a screen-time weight. Sources longer than about 150,000 characters are read in parts.
3. **Timed plan** (deterministic):
   - Audio sources keep their real timestamps.
   - Text sources are paced by how many words the source spends on each beat, scaled by the beat's dramatic weight.
   - The plan is cut into the chosen increment. Each segment gets the exact source text it must cover and a suggested shot count based on intensity and energy. In music mode it also gets the section, energy level and downbeat cut points.
4. **Writing** (Claude): segments are written in batches with the full method, the camera library and the story bible as a cached system prompt. The previous segments are passed in for continuity and the next segment's material is passed in for the lead-out. Output is structured JSON, and the shot timings are re-tiled in code so they always add up.

The method itself, distilled from the manuals, is in [`principles.py`](principles.py). It covers prompt anatomy, continuity, transitions, pacing, movement, action and looks, the 8-step VFX method with layering and stacking, conditions and abilities, music-video structure, screenplay craft and negative prompts.

## Run it

```bash
pip install -r script_studio/requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn script_studio.app:app --port 8000
# open http://localhost:8000
```

Run these commands from the repository root.

### Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required for writing. |
| `SCRIPT_MODEL` | `claude-opus-5-5` | Claude model used for the bible and the script. |
| `OPENAI_API_KEY` | — | Turns on transcription with the OpenAI Whisper API (files up to 25 MB). |
| `TRANSCRIBE_BACKEND` | auto | `openai`, `faster-whisper` or `none`. |
| `WHISPER_MODEL` | `small` | Model size for local faster-whisper (`pip install faster-whisper`). |
| `SCRIPT_JOBS_DIR` | `./script_jobs` | Where finished and in-progress scripts are saved. |
| `MAX_UPLOAD_MB` | `200` | Upload size limit. |
| `ALLOW_PRIVATE_URLS` | — | Set to `1` to allow links to local or private network addresses (blocked by default). |

**Transcription:**
- **Story mode with audio** needs a transcription backend, either `OPENAI_API_KEY` or `faster-whisper`.
- **Music mode** works without one: uploaded lyrics are spread across the vocal sections by word count, and an instrumental track gets a concept built from its sections. With a backend, the transcribed vocals give exact lyric timing, and any lyrics you upload are used for the correct wording.

## API

| Method | Path | |
|---|---|---|
| `POST` | `/api/jobs` | multipart: `settings` (JSON), `files[]`, `audio`, `url`, `text` → `{id}` |
| `GET` | `/api/jobs/{id}` | status, progress, story bible, plan and the segments written so far |
| `POST` | `/api/jobs/{id}/resume` | continue a job that stopped |
| `POST` | `/api/jobs/{id}/segments/{i}/rewrite` | `{"note": "..."}` |
| `PUT` | `/api/jobs/{id}/segments/{i}` | save an edited segment |
| `GET` | `/api/jobs/{id}/export/{fmt}` | `pdf`, `fountain`, `md`, `csv`, `prompts`, `json` |

The settings JSON looks like this:

```json
{"mode": "story|music", "increment": "5|10|15|30|60|full",
 "target_seconds": 180, "visual_style": "...", "tone": "...",
 "aspect_ratio": "16:9", "target_generator": "any|veo|sora|kling|runway|luma",
 "include_dialogue": true, "direction": "..."}
```

## Tests

```bash
python -m pytest script_studio/tests -q
```

The tests cover ingestion of every format, audio analysis on a synthetic song, planning and timing, chunked long sources, recovery from truncated output, the story, audio and music pipelines, every export, and the web API including editing and rewriting. They use a fake Claude client, so no API key is needed.
