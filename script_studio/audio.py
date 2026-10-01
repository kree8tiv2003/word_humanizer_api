"""Music and voice-track analysis: tempo, beats, downbeats, sections and energy."""
from __future__ import annotations

import numpy as np

from .models import AudioAnalysis, Section

SR = 22050


class AudioError(ValueError):
    pass


def load(path: str) -> tuple[np.ndarray, int]:
    import librosa
    try:
        y, sr = librosa.load(path, sr=SR, mono=True)
    except Exception as e:
        try:   # M4A/AAC/MP4/WMA etc.: decode with PyAV (bundled with faster-whisper) when available
            from faster_whisper.audio import decode_audio
            y, sr = decode_audio(path, sampling_rate=SR), SR
        except Exception:
            raise AudioError(f'Could not decode the audio file ({e}). Try MP3, WAV, FLAC or OGG.')
    if y.size < sr:
        raise AudioError('The audio is shorter than one second.')
    return y, sr


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = np.percentile(x, 5), np.percentile(x, 98)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def _downbeats(beats: np.ndarray, onset_at_beats: np.ndarray, per_bar: int = 4) -> np.ndarray:
    if len(beats) < per_bar:
        return beats[:1]
    scores = [onset_at_beats[k::per_bar].mean() for k in range(per_bar)]
    return beats[int(np.argmax(scores))::per_bar]


def _sections(y, sr, beats_frames, duration, rms_sec) -> list[Section]:
    import librosa
    hop = 512
    if len(beats_frames) < 16 or duration < 25:
        e = float(np.mean(rms_sec)) if len(rms_sec) else 0.5
        return [Section(start=0, end=duration, label='A', role='verse', energy=e, energy_level=_level(e, 0.33, 0.66))]
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
    feat = np.vstack([librosa.util.normalize(chroma, axis=1), librosa.util.normalize(mfcc, axis=1)])
    sync = librosa.util.sync(feat, beats_frames, aggregate=np.median)
    k = int(np.clip(round(duration / 22), 3, 14))
    k = min(k, sync.shape[1] - 1)
    bounds = librosa.segment.agglomerative(sync, k)           # indices into beat-synced frames
    bound_frames = librosa.util.fix_frames(np.asarray(beats_frames)[np.clip(bounds, 0, len(beats_frames) - 1)],
                                           x_min=0)
    times = sorted(set([0.0] + [float(t) for t in librosa.frames_to_time(bound_frames, sr=sr, hop_length=hop)] + [duration]))
    # merge tiny sections (< 6 s) into neighbours
    merged = [times[0]]
    for t in times[1:]:
        if t - merged[-1] < 6 and t != duration:
            continue
        merged.append(t)
    if len(merged) > 2 and merged[-1] - merged[-2] < 6:
        merged.pop(-2)
    spans = list(zip(merged[:-1], merged[1:]))

    # describe each section by mean chroma and energy
    frame_t = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    descs, energies = [], []
    for a, b in spans:
        m = (frame_t >= a) & (frame_t < b)
        descs.append(chroma[:, m].mean(axis=1) if m.any() else chroma.mean(axis=1))
        sec = rms_sec[int(a):max(int(a) + 1, int(b))]
        energies.append(float(np.mean(sec)) if len(sec) else 0.0)

    labels: list[str] = []
    protos: list[np.ndarray] = []
    for d in descs:
        best, best_sim = None, 0.0
        for li, p in enumerate(protos):
            sim = float(np.dot(d, p) / (np.linalg.norm(d) * np.linalg.norm(p) + 1e-9))
            if sim > best_sim:
                best, best_sim = li, sim
        if best is not None and best_sim > 0.97:
            labels.append(chr(65 + best))
        else:
            protos.append(d)
            labels.append(chr(65 + min(len(protos) - 1, 25)))

    q1, q2 = np.percentile(energies, [33, 66]) if len(energies) > 2 else (0.33, 0.66)
    counts = {l: labels.count(l) for l in labels}
    # the repeated label with the highest average energy is the likely chorus
    rep = [l for l in counts if counts[l] > 1]
    chorus = max(rep, key=lambda l: np.mean([e for e, x in zip(energies, labels) if x == l])) if rep else None
    out = []
    for i, ((a, b), lab, e) in enumerate(zip(spans, labels, energies)):
        lvl = _level(e, q1, q2)
        if i == 0 and lvl != 'high' and len(spans) > 2:
            role = 'intro'
        elif i == len(spans) - 1 and lvl != 'high' and len(spans) > 2:
            role = 'outro'
        elif lab == chorus:
            role = 'chorus'
        elif counts[lab] == 1 and i > len(spans) / 2:
            role = 'bridge'
        else:
            role = 'verse'
        out.append(Section(start=round(a, 2), end=round(b, 2), label=lab, role=role, energy=round(e, 3), energy_level=lvl))
    return out


def _level(e: float, q1: float, q2: float) -> str:
    return 'low' if e <= q1 else 'high' if e > q2 else 'medium'


def analyze(path: str) -> AudioAnalysis:
    import librosa
    y, sr = load(path)
    duration = float(len(y) / sr)
    hop = 512
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=hop)
    tempo = float(np.atleast_1d(tempo)[0])
    beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
    onset_at = onset[np.clip(beat_frames, 0, len(onset) - 1)] if len(beat_frames) else np.array([])
    down = _downbeats(beats, onset_at) if len(beats) else beats

    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    per_sec = np.array([rms[(rms_t >= s) & (rms_t < s + 1)].mean() if ((rms_t >= s) & (rms_t < s + 1)).any() else 0
                        for s in range(int(np.ceil(duration)))])
    per_sec = _norm(per_sec) if per_sec.size else per_sec
    sections = _sections(y, sr, beat_frames, duration, per_sec)
    return AudioAnalysis(duration=round(duration, 3), tempo=round(tempo, 1),
                         beats=[round(float(b), 3) for b in beats], downbeats=[round(float(b), 3) for b in down],
                         sections=sections, energy_curve=[round(float(v), 3) for v in per_sec])


def section_at(a: AudioAnalysis, t: float) -> Section | None:
    for s in a.sections:
        if s.start <= t < s.end:
            return s
    return a.sections[-1] if a.sections else None


def mean_energy(a: AudioAnalysis, start: float, end: float) -> float:
    vals = a.energy_curve[int(start):max(int(start) + 1, int(np.ceil(end)))]
    return float(np.mean(vals)) if vals else 0.5
