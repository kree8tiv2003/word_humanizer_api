"""End-to-end pipeline: uploads -> SourceDoc -> story bible -> timed plan -> script segments."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import audio as audio_mod
from . import transcribe
from .ingest import IngestError, fetch_url, from_html, guess_name, is_audio, load_document, merge_docs, text_to_units
from .models import ScriptResult, Settings, SourceDoc, Unit
from .planner import build_plan, time_units_over_audio, total_seconds, units_from_sections
from .writer import Writer


@dataclass
class Inputs:
    documents: list[tuple[str, bytes]] = field(default_factory=list)   # (filename, data)
    audio: Optional[tuple[str, bytes]] = None
    url: str = ''
    text: str = ''


def _save_temp(name: str, data: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=os.path.splitext(name)[1] or '.audio')
    with os.fdopen(fd, 'wb') as f:
        f.write(data)
    return path


def prepare_source(inp: Inputs, settings: Settings, progress: Callable[[str], None] = lambda m: None) -> SourceDoc:
    docs: list[SourceDoc] = []
    audio_in = inp.audio
    notes: list[str] = []

    for name, data in inp.documents:
        if is_audio(name):
            if audio_in is None:
                audio_in = (name, data)
            else:
                notes.append(f'Ignored extra audio file {name}; only one audio track is used.')
            continue
        progress(f'Reading {name}')
        docs.append(load_document(name, data))

    if inp.url.strip():
        progress('Fetching the link')
        data, ctype, final = fetch_url(inp.url.strip())
        name = guess_name(final, ctype)
        if is_audio(name, ctype):
            if audio_in is None:
                audio_in = (name, data)
        elif name.endswith(('.html', '.htm')) or 'html' in ctype:
            title, text = from_html(data.decode('utf-8', errors='replace'), final)
            docs.append(SourceDoc(kind='url', title=title or final, units=text_to_units(text)))
        else:
            d = load_document(name, data)
            d.kind = 'url'
            docs.append(d)

    if inp.text.strip():
        docs.append(SourceDoc(kind='text', title='Pasted text', units=text_to_units(inp.text)))

    text_doc = merge_docs(docs) if docs else None

    if settings.mode == 'music' and audio_in is None:
        raise IngestError('Music video mode needs an audio track. Upload the song (MP3, WAV, FLAC, M4A or OGG).')
    if audio_in is None:
        if text_doc is None:
            raise IngestError('Upload a document or audio file, paste text, or add a link.')
        text_doc.notes += notes
        return text_doc

    path = _save_temp(*audio_in)
    try:
        title = os.path.splitext(os.path.basename(audio_in[0]))[0]
        analysis = None
        if settings.mode == 'music':
            progress('Analysing the music: tempo, beats, sections and energy')
            analysis = audio_mod.analyze(path)

        transcript: list[Unit] = []
        if transcribe.available():
            progress('Transcribing the audio')
            try:
                transcript = transcribe.transcribe(path)
            except Exception as e:  # keep going in music mode; story mode needs words
                if settings.mode != 'music':
                    raise IngestError(f'Transcription failed: {e}')
                notes.append(f'Transcription failed ({e}); lyric timing is approximate.')
        elif settings.mode == 'story':
            raise IngestError('Turning speech into a script needs a transcription backend. Set OPENAI_API_KEY '
                              'or install faster-whisper (see README).')
        else:
            notes.append('No transcription backend configured: lyrics are spread across the vocal sections by word count.')

        words = sum(u.words for u in transcript)
        if settings.mode == 'story':
            doc = SourceDoc(kind='audio', title=title, units=transcript,
                            reference=text_doc.text if text_doc else '', notes=notes)
            if not transcript:
                raise IngestError('No speech was found in the audio.')
            return doc

        # music
        if transcript and words >= 12:
            units = transcript
            if text_doc:
                notes.append('Timing comes from the transcribed vocals; your uploaded text is used as the lyric/script reference.')
        elif text_doc:
            lines = [l.strip() for u in text_doc.units for l in u.text.split('\n') if l.strip()]
            units = time_units_over_audio([Unit(idx=i, text=l) for i, l in enumerate(lines)], analysis)
        else:
            units = units_from_sections(analysis)
            notes.append('No lyrics found: the concept follows the music\'s sections and energy.')
        return SourceDoc(kind='music', title=(text_doc.title if text_doc else title), units=units,
                         reference=text_doc.text if (text_doc and units is transcript) else '', audio=analysis, notes=notes)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def target_beats(doc: SourceDoc, settings: Settings) -> int:
    total = total_seconds(doc, settings)
    if settings.mode == 'music' and doc.audio:
        n = max(len(doc.audio.sections) * 2, round(total / 12))
    elif settings.increment == 'full':
        n = round(doc.word_count / 250)
    else:
        n = round(total / (float(settings.increment) * 1.5))
    return int(min(150, max(6, n, min(len(doc.units), 6))))


def run(doc: SourceDoc, settings: Settings, writer: Writer,
        progress: Callable[[str, float], None] = lambda m, p: None,
        on_segments: Callable[[ScriptResult], None] = lambda r: None) -> ScriptResult:
    progress('Building the story bible', 0.08)
    bible = writer.build_bible(doc, settings, target_beats(doc, settings), lambda m: progress(m, 0.1))
    plan = build_plan(doc, bible.beats, settings)
    result = ScriptResult(settings=settings, source_kind=doc.kind, bible=bible, plan=plan, notes=list(doc.notes))
    progress(f'Planned {len(plan.segments)} segments over {plan.total_seconds:.0f} s', 0.2)
    on_segments(result)
    n = len(plan.segments)
    for part in writer.write_segments(result, doc, lambda m: progress(m, 0.2 + 0.8 * len(result.segments) / max(n, 1))):
        result.segments.extend(part)
        progress(f'Wrote {len(result.segments)} of {n} segments', 0.2 + 0.8 * len(result.segments) / n)
        on_segments(result)
    return result
