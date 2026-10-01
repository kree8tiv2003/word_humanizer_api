"""Claude calls: story bible, segment-by-segment script writing and single-segment rewrites."""
from __future__ import annotations

import json
import os
from typing import Callable, Iterator, Optional

from .models import (BeatsOut, PlannedSegment, Plan, ScriptResult, SegmentOut, SegmentsOut, Settings,
                     SourceDoc, StoryBibleOut)
from .planner import fmt_time, normalize_beats, tile_shots
from .principles import GENERATOR_NOTES, system_blocks

DEFAULT_MODEL = 'claude-opus-5-5'


def current_model() -> str:
    return os.getenv('SCRIPT_MODEL') or DEFAULT_MODEL
BIBLE_CHUNK_CHARS = 150_000
MAX_TOKENS = int(os.getenv('SCRIPT_MAX_TOKENS', '16000'))
BATCH_TOKEN_BUDGET = 8000


class WriterError(RuntimeError):
    pass


def make_client():
    import anthropic
    if not os.getenv('ANTHROPIC_API_KEY'):
        raise WriterError('No Anthropic API key yet. Open Settings (the gear, top right) and paste your key.')
    return anthropic.Anthropic(max_retries=4)


class Writer:
    def __init__(self, client=None, model: str | None = None):
        self._client = client
        self.model = model or current_model()

    @property
    def client(self):
        if self._client is None:
            self._client = make_client()
        return self._client

    # ------------------------------------------------------------------ low level
    def _call(self, system: list[dict], prompt: str, schema, max_tokens: int = MAX_TOKENS):
        resp = self.client.messages.parse(model=self.model, max_tokens=max_tokens, system=system,
                                          messages=[{'role': 'user', 'content': prompt}], output_format=schema)
        if resp.stop_reason == 'max_tokens':
            raise _Truncated()
        if resp.stop_reason == 'refusal':
            raise WriterError('The model declined to write this material.')
        out = resp.parsed_output
        if out is None:
            raise WriterError('The model returned no structured output.')
        return out

    # ------------------------------------------------------------------ story bible
    def build_bible(self, doc: SourceDoc, settings: Settings, target_beats: int,
                    progress: Callable[[str], None] = lambda m: None) -> StoryBibleOut:
        units = doc.units
        chunks, cur, size = [], [], 0
        for u in units:
            line = _unit_line(u)
            if cur and size + len(line) > BIBLE_CHUNK_CHARS:
                chunks.append(cur); cur, size = [], 0
            cur.append(u); size += len(line)
        if cur:
            chunks.append(cur)
        total_words = doc.word_count or 1
        def beats_for(chunk):
            w = sum(u.words for u in chunk)
            return max(3, round(target_beats * w / total_words))

        progress('Reading the source and building the story bible')
        bible = self._call(system_blocks(), _bible_prompt(doc, settings, chunks[0], beats_for(chunks[0]), len(chunks) > 1),
                           StoryBibleOut)
        for ci, chunk in enumerate(chunks[1:], start=2):
            progress(f'Reading part {ci} of {len(chunks)} of the source')
            more = self._call(system_blocks(_bible_json(bible, beats=False)),
                              _beats_prompt(chunk, beats_for(chunk), bible.beats[-1].summary if bible.beats else ''), BeatsOut)
            bible.beats += more.beats
            known = {c.name.lower() for c in bible.characters}
            bible.characters += [c for c in more.new_characters if c.name.lower() not in known]
            known = {l.name.lower() for l in bible.locations}
            bible.locations += [l for l in more.new_locations if l.name.lower() not in known]
        bible.beats = normalize_beats(bible.beats, len(units))
        return bible

    # ------------------------------------------------------------------ segments
    def write_segments(self, result: ScriptResult, doc: SourceDoc,
                       progress: Callable[[str], None] = lambda m: None) -> Iterator[list[SegmentOut]]:
        plan, settings = result.plan, result.settings
        system = system_blocks(_bible_json(result.bible))
        batches = _batches(plan.segments)
        written: list[SegmentOut] = list(result.segments)
        for batch in batches:
            batch = [s for s in batch if s.index >= len(written)]   # resume: skip what is already written
            if not batch:
                continue
            progress(f'Writing segments {batch[0].index + 1}-{batch[-1].index + 1} of {len(plan.segments)}')
            for part in self._write_batch(system, result, doc, batch, written):
                written.extend(part)
                yield part

    def _write_batch(self, system, result, doc, batch: list[PlannedSegment], written) -> Iterator[list[SegmentOut]]:
        prompt = _segments_prompt(result, doc, batch, written)
        try:
            out = self._call(system, prompt, SegmentsOut)
        except _Truncated:
            if len(batch) == 1:
                raise WriterError(f'Segment {batch[0].index + 1} is too long to write in one call. Choose a shorter increment.')
            mid = len(batch) // 2
            first = []
            for part in self._write_batch(system, result, doc, batch[:mid], written):
                first.extend(part); yield part
            yield from self._write_batch(system, result, doc, batch[mid:], written + first)
            return
        yield _fit(batch, out.segments)

    def rewrite_segment(self, result: ScriptResult, doc: SourceDoc, index: int, note: str) -> SegmentOut:
        seg = result.plan.segments[index]
        system = system_blocks(_bible_json(result.bible))
        prompt = _segments_prompt(result, doc, [seg], result.segments[:index], following=result.segments[index + 1:index + 2],
                                  current=result.segments[index] if index < len(result.segments) else None, note=note)
        return _fit([seg], self._call(system, prompt, SegmentsOut).segments)[0]


class _Truncated(Exception):
    pass


# --------------------------------------------------------------------------- helpers
def _unit_line(u) -> str:
    stamp = f' {fmt_time(u.start)}-{fmt_time(u.end)}' if u.start is not None else ''
    return f'[{u.idx}{stamp}] {u.text}'


def _settings_block(s: Settings) -> str:
    inc = 'a complete, full-length script (scenes of natural length)' if s.increment == 'full' else f'{s.increment}-second segments'
    lines = [f'Mode: {"music video / musical production" if s.mode == "music" else "story to script"}',
             f'Output: {inc}', f'Visual style: {s.visual_style or "choose what suits the source"}',
             f'Aspect ratio: {s.aspect_ratio}', GENERATOR_NOTES.get(s.target_generator, GENERATOR_NOTES['any'])]
    if s.tone:
        lines.append(f'Tone: {s.tone}')
    if not s.include_dialogue:
        lines.append('Dialogue: none. Tell the story visually, with voiceover only if essential.')
    if s.direction:
        lines.append(f'Director\'s notes from the user (follow them): {s.direction}')
    return '\n'.join(lines)


def _audio_block(doc: SourceDoc) -> str:
    a = doc.audio
    if not a:
        return ''
    secs = '\n'.join(f'  {fmt_time(s.start)}-{fmt_time(s.end)}  section {s.label} ({s.role}), {s.energy_level} energy'
                     for s in a.sections)
    return f'AUDIO: {fmt_time(a.duration)} long, about {a.tempo:.0f} BPM ({60 / max(a.tempo, 1):.2f} s per beat).\nSections:\n{secs}\n'


def _bible_prompt(doc, settings, units, n_beats, partial) -> str:
    src = '\n'.join(_unit_line(u) for u in units)
    ref = f'\nREFERENCE MATERIAL (lyrics sheet or script supplied with the audio; use it for exact wording):\n{doc.reference[:20000]}\n' if doc.reference else ''
    part = ('\nThis is the FIRST PART of a longer source; cover only these units with beats, but set up characters, '
            'locations and style for the whole work.' if partial else '')
    music = ('\nThis is a music video. Build a visual concept that fits the song: performance setting(s) for the artist, '
             'a narrative or metaphor that follows the lyrics and emotional arc, and a visual motif for the chorus. '
             'If the source is only instrumental section markers, invent a concept that follows the energy of each section.'
             if settings.mode == 'music' else '')
    return f"""{_settings_block(settings)}
{_audio_block(doc)}
SOURCE ({doc.kind}, title "{doc.title}"), numbered units:
{src}
{ref}
TASK: Build the story bible for adapting this source into a script.{part}{music}
- Characters: everyone who appears on screen (for a song: the artist/performer plus story characters). Give each a precise visual_signature that will be repeated in every prompt.
- Locations: each distinct place with a visual_signature including light and time of day.
- Beats: about {n_beats} beats, in source order, each covering a contiguous range of unit numbers (first_unit..last_unit), together covering every unit. Intensity 1-5 follows the source's energy; weight is relative screen time (1.0 average; >1 for moments the story dwells on or that need visual space, <1 for summary or connective material).
- visual_style, tone and pacing_notes should follow the user's settings and the source.
Write in the language of the source unless the director's notes say otherwise."""


def _beats_prompt(units, n_beats, last_summary) -> str:
    src = '\n'.join(_unit_line(u) for u in units)
    return f"""Continue the story bible for the next part of the source. The previous beat was: "{last_summary}".
SOURCE units:
{src}

TASK: about {n_beats} beats in order covering every unit above (first_unit..last_unit use the unit numbers shown). List any characters or locations not already in the bible in new_characters / new_locations; otherwise leave those lists empty."""


def _bible_json(bible: StoryBibleOut, beats: bool = True) -> str:
    d = bible.model_dump()
    if beats:
        d['beats'] = [{'beat': i, 'summary': b['summary'], 'emotion': b['emotion'], 'intensity': b['intensity'],
                       'characters': b['characters'], 'location': b['location']} for i, b in enumerate(d['beats'])]
    else:
        d.pop('beats')
    return json.dumps(d, ensure_ascii=False, indent=1)


def _seg_tokens(s: PlannedSegment) -> int:
    n = int(s.shot_hint.split()[0]) if s.shot_hint else 2
    return 300 + (n + 1) * 260


def _batches(segs: list[PlannedSegment]) -> list[list[PlannedSegment]]:
    out, cur, tok = [], [], 0
    for s in segs:
        t = _seg_tokens(s)
        if cur and (tok + t > BATCH_TOKEN_BUDGET or len(cur) >= 12):
            out.append(cur); cur, tok = [], 0
        cur.append(s); tok += t
    if cur:
        out.append(cur)
    return out


def _describe_segment(result: ScriptResult, s: PlannedSegment) -> str:
    beats = result.bible.beats
    bl = '; '.join(f'#{i} {beats[i].summary} ({beats[i].emotion}, intensity {beats[i].intensity}, at {beats[i].location})'
                   for i in s.beat_ids if i < len(beats))
    lines = [f'SEGMENT index={s.index} | {fmt_time(s.start)}-{fmt_time(s.end)} (start={s.start:.2f}s, end={s.end:.2f}s, '
             f'{s.duration:.1f}s) | intensity {s.intensity} | aim for {s.shot_hint}',
             f'Story beats: {bl or "continuation"}']
    if s.section:
        lines.append(f'Music: {s.section}; {s.energy_level} energy')
    if s.downbeat_times:
        lines.append('Downbeats (good cut points): ' + ', '.join(f'{t:.2f}' for t in s.downbeat_times[:24]))
    elif s.beat_times:
        lines.append('Beats: ' + ', '.join(f'{t:.2f}' for t in s.beat_times[:32]))
    label = ('Lyrics and material in this window (timestamps show when each line is sung)' if s.section
             else 'Source to cover in this segment')
    lines.append(f'{label}:\n"""\n{s.source}\n"""')
    return '\n'.join(lines)


def _prev_block(prev: list[SegmentOut]) -> str:
    if not prev:
        return 'This is the opening of the script: start with a strong hook image.'
    out = ['PREVIOUS SEGMENTS (continue seamlessly from these):']
    for p in prev[-2:]:
        last = p.shots[-1] if p.shots else None
        out.append(f'- Segment {p.index}: {p.summary}')
        if last:
            out.append(f'  Last shot ({last.camera}): {last.prompt}')
        out.append(f'  Continuity: {p.continuity}\n  Transition out: {p.transition_out}')
    return '\n'.join(out)


def _segments_prompt(result: ScriptResult, doc: SourceDoc, batch: list[PlannedSegment], written: list[SegmentOut],
                     following: Optional[list[SegmentOut]] = None, current: Optional[SegmentOut] = None, note: str = '') -> str:
    s = result.settings
    plan = result.plan
    nxt = plan.segments[batch[-1].index + 1] if batch[-1].index + 1 < len(plan.segments) else None
    lead = ''
    if following:
        f = following[0]
        lead = f'NEXT SEGMENT ALREADY WRITTEN (end so it flows into this): {f.summary}. Its first shot: {f.shots[0].prompt if f.shots else ""}'
    elif nxt is not None:
        lead = f'NEXT SEGMENT (for your lead-out only, do not write it): {nxt.source[:300]}'
    else:
        lead = 'This batch ends the script: land the ending and resolve the final image.'
    redo = ''
    if current is not None:
        redo = ('\nREWRITE REQUEST. Current version of this segment:\n' + current.model_dump_json(indent=1) +
                f'\nUser note for the rewrite: {note or "make it more dynamic and cinematic"}\nKeep what works, apply the note.\n')
    full = s.increment == 'full'
    rules = f"""RULES
- Return exactly {len(batch)} segment(s), with index values {', '.join(str(b.index) for b in batch)} in that order.
- Shots inside each segment must tile it exactly: first shot starts at the segment start, each shot starts where the previous ends, last shot ends at the segment end. Times are absolute seconds.
- {"A full scene: as many shots as the drama needs, complete dialogue." if full else "Each segment is one generated clip span; keep each shot's action achievable within its length."}
- Every shot prompt follows the anatomy: camera first, character visual signatures verbatim, motion verbs, setting and light, layered effect, look. 40-90 words.
- Cover the source text given for each segment, in order, at the pace it sets. Do not skip story events or invent ones that contradict the source.
- {"Put the exact sung words for each shot in its lyrics field (from the timestamps)." if s.mode == "music" else "Use voiceover only where the source's narration carries meaning the images cannot."}
- Fields with nothing to say are empty strings or empty lists."""
    return f"""{_settings_block(s)}
{_audio_block(doc)}
{_prev_block(written)}
{redo}
WRITE THESE SEGMENTS:
""" + '\n\n'.join(_describe_segment(result, b) for b in batch) + f"""

{lead}

{rules}"""


def _fit(batch: list[PlannedSegment], segs: list[SegmentOut]) -> list[SegmentOut]:
    """Match model output to the planned segments and enforce exact timing."""
    by_idx = {s.index: s for s in segs}
    out = []
    for i, p in enumerate(batch):
        seg = by_idx.get(p.index) or (segs[i] if i < len(segs) else None)
        if seg is None:
            raise WriterError(f'The model skipped segment {p.index + 1}.')
        seg.index = p.index
        if not seg.shots:
            raise WriterError(f'Segment {p.index + 1} came back with no shots.')
        seg.shots.sort(key=lambda x: x.start)
        tile_shots(p.start, p.end, seg.shots)
        out.append(seg)
    return out
