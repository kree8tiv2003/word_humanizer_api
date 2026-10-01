"""Turn the story beats and the source's own pacing into a timed segment plan."""
from __future__ import annotations

import math

from .audio import mean_energy, section_at
from .models import AudioAnalysis, BeatOut, Plan, PlannedSegment, Settings, SourceDoc, Unit

NARRATION_WPS = 2.6        # ~155 words per minute: screen time for untimed text
MAX_SCENE_SECONDS = 120    # "full" mode: split longer scenes
SOURCE_CHARS = 1600        # source text handed to the writer per segment


def fmt_time(t: float) -> str:
    t = max(0.0, t)
    m, s = divmod(t, 60)
    h, m = divmod(int(m), 60)
    return f'{h}:{m:02d}:{s:04.1f}' if h else f'{m}:{s:04.1f}'


# --------------------------------------------------------------------------- source timing helpers
def units_from_sections(audio: AudioAnalysis) -> list[Unit]:
    """Music with no lyrics or concept: one unit per musical section, so beats can still map to time."""
    return [Unit(idx=i, start=s.start, end=s.end,
                 text=f'[Instrumental section {s.label}, likely {s.role}, {s.energy_level} energy, {fmt_time(s.start)}-{fmt_time(s.end)}]')
            for i, s in enumerate(audio.sections)] or [Unit(idx=0, start=0, end=audio.duration, text='[Instrumental]')]


def time_units_over_audio(units: list[Unit], audio: AudioAnalysis) -> list[Unit]:
    """Spread untimed lyric/script units across the vocal part of a song, proportional to word count."""
    start, end = 0.0, audio.duration
    secs = audio.sections
    if len(secs) > 2:
        if secs[0].role == 'intro':
            start = secs[0].end
        if secs[-1].role == 'outro':
            end = secs[-1].start
    total_words = sum(u.words for u in units) or 1
    t = start
    out = []
    for u in units:
        d = (end - start) * u.words / total_words
        out.append(Unit(idx=u.idx, text=u.text, start=round(t, 2), end=round(t + d, 2)))
        t += d
    return out


def total_seconds(doc: SourceDoc, settings: Settings) -> float:
    if doc.audio:
        return doc.audio.duration
    if doc.timed:
        return float(doc.units[-1].end)
    if settings.target_seconds:
        return float(settings.target_seconds)
    return max(15.0, doc.word_count / NARRATION_WPS)


def normalize_beats(beats: list[BeatOut], n_units: int) -> list[BeatOut]:
    """Sort beats, clamp them to the source and make them cover every unit with no gaps or overlaps."""
    if n_units == 0:
        return beats
    bs = sorted(beats, key=lambda b: (b.first_unit, b.last_unit))
    clean: list[BeatOut] = []
    for b in bs:
        b.first_unit = min(max(0, b.first_unit), n_units - 1)
        b.last_unit = min(max(b.first_unit, b.last_unit), n_units - 1)
        if clean and b.first_unit <= clean[-1].last_unit:
            if b.last_unit <= clean[-1].last_unit:
                continue  # fully inside the previous beat
            b.first_unit = clean[-1].last_unit + 1
        b.intensity = int(min(5, max(1, b.intensity)))
        b.weight = float(min(5.0, max(0.2, b.weight or 1.0)))
        clean.append(b)
    if not clean:
        return [BeatOut(first_unit=0, last_unit=n_units - 1, summary='The whole story', emotion='', intensity=3,
                        weight=1.0, characters=[], location='')]
    clean[0].first_unit = 0
    for a, b in zip(clean, clean[1:]):
        a.last_unit = b.first_unit - 1
    clean[-1].last_unit = n_units - 1
    return clean


def beat_windows(doc: SourceDoc, beats: list[BeatOut], total: float) -> list[tuple[float, float]]:
    """Screen-time window for each beat. Timed sources keep their real timing; text is paced by
    how much the source spends on the beat (word share) blended with its dramatic weight."""
    if doc.timed:
        starts = [0.0] + [float(doc.units[b.first_unit].start) for b in beats[1:]]
        ends = starts[1:] + [total]
        return [(round(s, 3), round(max(e, s), 3)) for s, e in zip(starts, ends)]
    words = [sum(doc.units[i].words for i in range(b.first_unit, b.last_unit + 1)) for b in beats]
    # The source's own pace (words spent) scaled by the beat's dramatic weight, with a small floor so a
    # one-line moment ("The door slammed.") still gets a shot.
    raw = [(w ** 0.85) * b.weight for w, b in zip(words, beats)]
    floor = min(2.0, total / (len(beats) * 3))
    rest = max(0.0, total - floor * len(beats))
    tr = sum(raw) or 1
    durs = [floor + rest * r / tr for r in raw]
    out, t = [], 0.0
    for d in durs:
        out.append((round(t, 3), round(t + d, 3)))
        t += d
    out[-1] = (out[-1][0], round(total, 3))
    return out


def _slice_units(doc: SourceDoc, beat: BeatOut, win: tuple[float, float], a: float, b: float) -> list[Unit]:
    units = doc.units[beat.first_unit:beat.last_unit + 1]
    if doc.timed:
        hit = [u for u in units if u.end > a and u.start < b]
        return hit or units[:1]
    span = (win[1] - win[0]) or 1
    f0, f1 = (max(a, win[0]) - win[0]) / span, (min(b, win[1]) - win[0]) / span
    tw = sum(u.words for u in units) or 1
    out, acc = [], 0
    for u in units:
        u0, u1 = acc / tw, (acc + u.words) / tw
        acc += u.words
        if u1 > f0 and u0 < f1:
            out.append(u)
    return out or units[:1]


def _shot_len(intensity: float, energy_level: str, music: bool) -> float:
    base = {1: 8.0, 2: 6.5, 3: 4.5, 4: 3.0, 5: 2.2}[int(round(min(5, max(1, intensity))))]
    if music:
        base *= {'low': 1.3, 'medium': 0.9, 'high': 0.6}.get(energy_level, 1.0)
    return base


def build_plan(doc: SourceDoc, beats: list[BeatOut], settings: Settings) -> Plan:
    total = total_seconds(doc, settings)
    music = settings.mode == 'music' and doc.audio is not None
    windows = beat_windows(doc, beats, total)

    # segment boundaries
    if settings.increment == 'full':
        bounds, cur = [], None
        for i, (b, w) in enumerate(zip(beats, windows)):
            same = cur is not None and b.location.strip().lower() == beats[cur[0]].location.strip().lower()
            if same and w[1] - windows[cur[0]][0] <= MAX_SCENE_SECONDS:
                cur = (cur[0], i)
            else:
                if cur:
                    bounds.append((windows[cur[0]][0], windows[cur[1]][1]))
                cur = (i, i)
        if cur:
            bounds.append((windows[cur[0]][0], windows[cur[1]][1]))
        # music sections are natural scene breaks too
        if music:
            cuts = sorted({round(s.start, 2) for s in doc.audio.sections} | {round(a, 2) for a, _ in bounds})
            bounds = [(a, b) for a, b in zip(cuts, cuts[1:] + [total]) if b - a > 0.5]
    else:
        n = float(settings.increment)
        k = max(1, math.ceil(total / n - 1e-6))
        bounds = [(i * n, min((i + 1) * n, total)) for i in range(k)]
        if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < 1.0:   # fold a sliver into the previous clip
            a, _ = bounds.pop()
            bounds[-1] = (bounds[-1][0], total)

    segments = []
    for si, (a, b) in enumerate(bounds):
        seg = PlannedSegment(index=si, start=round(a, 3), end=round(b, 3))
        parts, weights, texts, used = [], [], [], set()
        for bi, (beat, w) in enumerate(zip(beats, windows)):
            ov = min(b, w[1]) - max(a, w[0])
            if ov <= 1e-6 and not (w[0] == w[1] and a <= w[0] < b):
                continue
            seg.beat_ids.append(bi)
            weights.append((max(ov, 0.01), beat.intensity))
            for u in _slice_units(doc, beat, w, a, b):
                if u.idx not in used:
                    used.add(u.idx)
                    stamp = f'[{fmt_time(u.start)}] ' if u.start is not None else ''
                    texts.append(stamp + u.text)
        tot = sum(o for o, _ in weights) or 1
        seg.intensity = round(sum(o * i for o, i in weights) / tot, 2) if weights else 3.0
        src = '\n'.join(texts)
        seg.source = src if len(src) <= SOURCE_CHARS else src[:SOURCE_CHARS].rsplit(' ', 1)[0] + ' …'

        if doc.audio:
            secs = [s for s in doc.audio.sections if s.end > a and s.start < b]
            seg.section = ' -> '.join(f'{s.role} ({s.label}) from {fmt_time(max(s.start, a))}' for s in secs)
            e = mean_energy(doc.audio, a, b)
            seg.energy_level = 'high' if e > 0.66 else 'low' if e < 0.33 else 'medium'
            seg.beat_times = [t for t in doc.audio.beats if a <= t < b]
            seg.downbeat_times = [t for t in doc.audio.downbeats if a <= t < b]
            mid = section_at(doc.audio, (a + b) / 2)
            if mid and not seg.energy_level:
                seg.energy_level = mid.energy_level

        n_shots = max(1, round(seg.duration / _shot_len(seg.intensity, seg.energy_level, music)))
        if settings.increment == '5':
            n_shots = min(n_shots, 2)
        lo, hi = max(1, n_shots - 1), n_shots + 1
        seg.shot_hint = f'{n_shots} shot' + ('s' if n_shots > 1 else '') + (f' (between {lo} and {hi})' if hi - lo > 1 else '')
        segments.append(seg)
    return Plan(increment=settings.increment, total_seconds=round(total, 3), segments=segments, beat_times=windows)


def tile_shots(seg_start: float, seg_end: float, shots: list) -> None:
    """Force shots to tile the segment exactly: ordered, contiguous, inside [start, end]."""
    if not shots:
        return
    n = len(shots)
    raw = []
    for s in shots:
        a, b = float(s.start), float(s.end)
        raw.append(max(0.05, b - a) if b > a else None)
    known = [d for d in raw if d]
    default = (sum(known) / len(known)) if known else (seg_end - seg_start) / n
    durs = [d if d else default for d in raw]
    scale = (seg_end - seg_start) / sum(durs)
    t = seg_start
    for s, d in zip(shots, durs):
        s.start = round(t, 2)
        t += d * scale
        s.end = round(t, 2)
    shots[-1].end = round(seg_end, 2)
