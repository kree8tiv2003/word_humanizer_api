"""Speech and vocal transcription with timestamps.

Backends, picked in this order unless TRANSCRIBE_BACKEND is set:
  openai          OpenAI Whisper API (needs OPENAI_API_KEY; files up to 25 MB)
  faster-whisper  local model via the faster-whisper package (downloads a model on first use)
"""
from __future__ import annotations

import os

from .models import Unit


class TranscriptionUnavailable(RuntimeError):
    pass


def _has_faster_whisper() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def backend() -> str | None:
    want = os.getenv('TRANSCRIBE_BACKEND', '').strip().lower()
    if want in ('openai', 'faster-whisper'):
        return want
    if want == 'none':
        return None
    if os.getenv('OPENAI_API_KEY'):
        return 'openai'
    if _has_faster_whisper():
        return 'faster-whisper'
    return None


def available() -> bool:
    return backend() is not None


def _openai(path: str) -> list[Unit]:
    import httpx
    if os.path.getsize(path) > 25 * 1024 * 1024:
        raise TranscriptionUnavailable('The OpenAI transcription API accepts files up to 25 MB. Compress the audio or use the local faster-whisper backend.')
    with open(path, 'rb') as f:
        r = httpx.post('https://api.openai.com/v1/audio/transcriptions',
                       headers={'Authorization': f'Bearer {os.environ["OPENAI_API_KEY"]}'},
                       data={'model': os.getenv('OPENAI_TRANSCRIBE_MODEL', 'whisper-1'), 'response_format': 'verbose_json',
                             'timestamp_granularities[]': 'segment'},
                       files={'file': (os.path.basename(path), f)}, timeout=600)
    if r.status_code >= 400:
        raise RuntimeError(f'Transcription failed: HTTP {r.status_code} {r.text[:300]}')
    segs = r.json().get('segments') or []
    return [Unit(idx=i, text=s['text'].strip(), start=float(s['start']), end=float(s['end']))
            for i, s in enumerate(x for x in segs if x.get('text', '').strip())]


_FW_MODEL = None
_FW_DEVICE = None
GPU_ERROR = ('cublas', 'cudnn', 'cuda', 'cudart', '.dll', 'libcu', 'gpu')


def _load_fw(device: str):
    from faster_whisper import WhisperModel
    return WhisperModel(os.getenv('WHISPER_MODEL', 'small'), device=device,
                        compute_type=os.getenv('WHISPER_COMPUTE', 'int8'))


def _run_fw(model, path: str) -> list[Unit]:
    segments, _info = model.transcribe(path, vad_filter=True)
    out = []
    for s in segments:   # segments is lazy: GPU library errors surface here
        t = s.text.strip()
        if t:
            out.append(Unit(idx=len(out), text=t, start=float(s.start), end=float(s.end)))
    return out


def _faster_whisper(path: str) -> list[Unit]:
    """Run on the CPU by default. The packaged app does not ship NVIDIA CUDA libraries, so a
    GPU is only used when WHISPER_DEVICE asks for it, and any GPU failure falls back to the CPU."""
    global _FW_MODEL, _FW_DEVICE
    want = os.getenv('WHISPER_DEVICE', 'cpu')
    if _FW_MODEL is None:
        try:
            _FW_MODEL, _FW_DEVICE = _load_fw(want), want
        except Exception as e:
            if want == 'cpu' or not any(k in str(e).lower() for k in GPU_ERROR):
                raise
            _FW_MODEL, _FW_DEVICE = _load_fw('cpu'), 'cpu'
    try:
        return _run_fw(_FW_MODEL, path)
    except Exception as e:
        if _FW_DEVICE == 'cpu' or not any(k in str(e).lower() for k in GPU_ERROR):
            raise
        _FW_MODEL, _FW_DEVICE = _load_fw('cpu'), 'cpu'
        return _run_fw(_FW_MODEL, path)


def transcribe(path: str) -> list[Unit]:
    b = backend()
    if b is None:
        raise TranscriptionUnavailable('No transcription backend is configured. Set OPENAI_API_KEY or install faster-whisper.')
    units = _openai(path) if b == 'openai' else _faster_whisper(path)
    # make segments contiguous and ordered
    units.sort(key=lambda u: u.start)
    for i, u in enumerate(units):
        u.idx = i
    return units
