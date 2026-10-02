import io
import json
import os
import time

import numpy as np
import pytest
import soundfile as sf

from script_studio import exporters, ingest, planner, transcribe
from script_studio.audio import analyze
from script_studio.models import BeatOut, Settings, SourceDoc, Unit
from script_studio.pipeline import Inputs, prepare_source, run, target_beats
from script_studio.principles import camera_library, system_blocks
from script_studio.writer import Writer

from .fakes import FakeClient

STORY = '\n\n'.join(
    f'Paragraph {i}. ' + ' '.join(['The storm grew and Mara climbed the stairs to the lamp.'] * (2 + i % 4)) for i in range(12))


# --------------------------------------------------------------------------- fixtures
def make_song(path, seconds=48, bpm=120, sr=22050):
    """Click track with a quiet first half and a loud, brighter second half."""
    t = np.arange(int(seconds * sr)) / sr
    y = np.zeros_like(t)
    beat = 60 / bpm
    for k in range(int(seconds / beat)):
        i = int(k * beat * sr)
        n = min(len(y) - i, int(0.05 * sr))
        amp = 0.3 if k * beat < seconds / 2 else 0.9
        y[i:i + n] += amp * np.sin(2 * np.pi * (220 if k * beat < seconds / 2 else 660) * t[:n]) * np.exp(-t[:n] * 60)
    pad = np.where(t < seconds / 2, 0.05, 0.25) * np.sin(2 * np.pi * np.where(t < seconds / 2, 110, 330) * t)
    sf.write(path, (y + pad).astype(np.float32), sr)


@pytest.fixture
def song(tmp_path):
    p = tmp_path / 'song.wav'
    make_song(str(p))
    return p


def docx_bytes(text):
    import docx
    d = docx.Document()
    for par in text.split('\n\n'):
        d.add_paragraph(par)
    b = io.BytesIO(); d.save(b)
    return b.getvalue()


def pdf_bytes(text):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import Paragraph, SimpleDocTemplate
    from reportlab.lib.styles import getSampleStyleSheet
    b = io.BytesIO()
    from reportlab.lib.styles import ParagraphStyle
    st = ParagraphStyle('p', parent=getSampleStyleSheet()['Normal'], spaceAfter=8)
    SimpleDocTemplate(b, pagesize=letter).build([Paragraph(p, st) for p in text.split('\n\n')])
    return b.getvalue()


# --------------------------------------------------------------------------- ingestion
def test_ingest_formats():
    for name, data in [('s.txt', STORY.encode()), ('s.docx', docx_bytes(STORY)), ('s.pdf', pdf_bytes(STORY)),
                       ('s.html', f'<html><title>Keeper</title><body><nav>menu</nav><article>{"".join(f"<p>{p}</p>" for p in STORY.split(chr(10) * 2))}</article></body></html>'.encode())]:
        doc = ingest.load_document(name, data)
        assert len(doc.units) >= 10, name
        assert 'Mara climbed' in doc.text, name
        assert 'menu' not in doc.text
    assert ingest.load_document('s.html', b'<html><title>Keeper</title><body><p>' + b'x ' * 200 + b'</p></body></html>').title == 'Keeper'


def test_lyrics_keep_lines_and_rtf_srt():
    lyr = 'I walk the line\nunder neon light\nhold on tonight\nwe were young\nwe were bright\nnever say goodbye\nrun run run\nall night'
    assert len(ingest.load_document('l.txt', lyr.encode()).units) == 8
    rtf = r'{\rtf1\ansi{\fonttbl\f0 Arial;}\f0 Hello storm.\par Second line.\par}'
    assert 'Hello storm.' in ingest.load_document('a.rtf', rtf.encode()).text
    srt = '1\n00:00:01,000 --> 00:00:02,000\nFirst line\n\n2\n00:00:03,000 --> 00:00:04,000\nSecond line\n'
    t = ingest.load_document('a.srt', srt.encode()).text
    assert 'First line' in t and '-->' not in t


def test_url_guard_blocks_private():
    with pytest.raises(ingest.IngestError):
        ingest._check_host('http://127.0.0.1/x')
    with pytest.raises(ingest.IngestError):
        ingest._check_host('file:///etc/passwd')


def test_bad_inputs():
    with pytest.raises(ingest.IngestError):
        ingest.load_document('x.doc', b'\xd0\xcf\x11\xe0')
    with pytest.raises(ingest.IngestError):
        prepare_source(Inputs(), Settings())
    with pytest.raises(ingest.IngestError):
        prepare_source(Inputs(text='hello there'), Settings(mode='music'))


# --------------------------------------------------------------------------- audio
def test_audio_analysis(song):
    a = analyze(str(song))
    assert abs(a.duration - 48) < 0.1
    assert 110 <= a.tempo <= 130 or 55 <= a.tempo <= 65 or 230 <= a.tempo <= 250
    assert len(a.beats) > 40 and len(a.downbeats) > 8
    assert len(a.sections) >= 2 and a.sections[0].start == 0 and abs(a.sections[-1].end - a.duration) < 0.1
    first_half = np.mean(a.energy_curve[:20]); second_half = np.mean(a.energy_curve[28:])
    assert second_half > first_half


# --------------------------------------------------------------------------- planning
def _doc():
    return SourceDoc(kind='text', title='t', units=ingest.text_to_units(STORY))


def _beats(n):
    return [BeatOut(first_unit=i, last_unit=i, summary=f'b{i}', emotion='', intensity=1 + i % 5, weight=1.0,
                    characters=[], location='A' if i < n // 2 else 'B') for i in range(n)]


@pytest.mark.parametrize('inc', ['5', '10', '15', '30', '60'])
def test_plan_increments_tile_exactly(inc):
    doc = _doc()
    beats = planner.normalize_beats(_beats(len(doc.units)), len(doc.units))
    plan = planner.build_plan(doc, beats, Settings(increment=inc, target_seconds=125))
    assert plan.segments[0].start == 0 and abs(plan.segments[-1].end - 125) < 1e-6
    for a, b in zip(plan.segments, plan.segments[1:]):
        assert a.end == b.start
    assert all(abs(s.duration - float(inc)) < 1e-6 for s in plan.segments[:-1])
    assert all(s.source for s in plan.segments)


def test_plan_full_groups_scenes_by_location():
    doc = _doc()
    beats = planner.normalize_beats(_beats(len(doc.units)), len(doc.units))
    plan = planner.build_plan(doc, beats, Settings(increment='full', target_seconds=200))
    assert len(plan.segments) == 2


def test_normalize_beats_covers_source():
    bs = [BeatOut(first_unit=5, last_unit=3, summary='', emotion='', intensity=9, weight=0, characters=[], location=''),
          BeatOut(first_unit=0, last_unit=2, summary='', emotion='', intensity=0, weight=1, characters=[], location=''),
          BeatOut(first_unit=1, last_unit=1, summary='', emotion='', intensity=3, weight=1, characters=[], location=''),
          BeatOut(first_unit=8, last_unit=99, summary='', emotion='', intensity=3, weight=1, characters=[], location='')]
    out = planner.normalize_beats(bs, 10)
    assert out[0].first_unit == 0 and out[-1].last_unit == 9
    assert all(a.last_unit + 1 == b.first_unit for a, b in zip(out, out[1:]))
    assert all(1 <= b.intensity <= 5 for b in out)


def test_text_pacing_follows_word_share_and_weight():
    units = [Unit(idx=0, text='short'), Unit(idx=1, text=' '.join(['long'] * 99))]
    doc = SourceDoc(kind='text', units=units)
    beats = [BeatOut(first_unit=i, last_unit=i, summary='', emotion='', intensity=3, weight=1, characters=[], location='') for i in range(2)]
    w = planner.beat_windows(doc, beats, 100)
    assert w[1][1] - w[1][0] > 3 * (w[0][1] - w[0][0])


def test_timed_source_keeps_real_timing():
    units = [Unit(idx=i, text=f'line {i}', start=i * 7.0, end=i * 7.0 + 6) for i in range(6)]
    doc = SourceDoc(kind='audio', units=units)
    beats = planner.normalize_beats([BeatOut(first_unit=i, last_unit=i + 1, summary='', emotion='', intensity=3, weight=1,
                                             characters=[], location='') for i in (0, 2, 4)], 6)
    w = planner.beat_windows(doc, beats, planner.total_seconds(doc, Settings()))
    assert w == [(0.0, 14.0), (14.0, 28.0), (28.0, 41.0)]


def test_tile_shots():
    from script_studio.models import ShotOut
    shots = [ShotOut(start=0, end=0, camera='', prompt='', action='', dialogue=[], voiceover='', lyrics='', sound='', vfx='', negative=''),
             ShotOut(start=3, end=9, camera='', prompt='', action='', dialogue=[], voiceover='', lyrics='', sound='', vfx='', negative='')]
    planner.tile_shots(10, 20, shots)
    assert shots[0].start == 10 and shots[-1].end == 20 and shots[0].end == shots[1].start


# --------------------------------------------------------------------------- principles
def test_principles_and_camera_library():
    assert len(camera_library()) == 94
    blocks = system_blocks('{"title": "x"}')
    text = ' '.join(b['text'] for b in blocks)
    for must in ('Anatomy of every shot prompt', 'visual signature', 'Match cut', '8-step VFX', 'Cut on beats', 'Dolly in / push in'):
        assert must in text
    assert blocks[-1]['cache_control'] == {'type': 'ephemeral'}


# --------------------------------------------------------------------------- end to end (fake model)
@pytest.mark.parametrize('inc', ['5', '10', '60', 'full'])
def test_story_pipeline(inc):
    client = FakeClient()
    doc = prepare_source(Inputs(documents=[('story.docx', docx_bytes(STORY))]), Settings(increment=inc))
    st = Settings(increment=inc, target_seconds=90)
    res = run(doc, st, Writer(client=client, model='m'))
    assert len(res.segments) == len(res.plan.segments)
    for seg, p in zip(res.segments, res.plan.segments):
        assert seg.index == p.index
        assert seg.shots[0].start == round(p.start, 2) and seg.shots[-1].end == round(p.end, 2)
        assert all(a.end == b.start for a, b in zip(seg.shots, seg.shots[1:]))
    assert client.calls[0]['schema'] == 'StoryBibleOut'
    assert 'Anatomy of every shot prompt' in client.calls[0]['system'][0]['text']
    seg_prompts = [c['prompt'] for c in client.calls if c['schema'] == 'SegmentsOut']
    assert 'PREVIOUS SEGMENTS' in seg_prompts[-1] or len(seg_prompts) == 1
    for fmt, (_ct, _ext, fn) in exporters.FORMATS.items():
        out = fn(res)
        assert out and len(out) > 100, fmt
    assert exporters.to_pdf(res)[:5] == b'%PDF-'


def test_truncation_splits_batch():
    client = FakeClient(truncate_once=True)
    doc = SourceDoc(kind='text', units=ingest.text_to_units(STORY))
    res = run(doc, Settings(increment='5', target_seconds=40), Writer(client=client))
    assert [s.index for s in res.segments] == list(range(len(res.plan.segments)))


def test_long_source_is_chunked(monkeypatch):
    import script_studio.writer as w
    monkeypatch.setattr(w, 'BIBLE_CHUNK_CHARS', 800)
    client = FakeClient()
    doc = SourceDoc(kind='text', units=ingest.text_to_units(STORY))
    bible = Writer(client=client).build_bible(doc, Settings(), 12)
    assert [c['schema'] for c in client.calls].count('BeatsOut') >= 1
    assert bible.beats[0].first_unit == 0 and bible.beats[-1].last_unit == len(doc.units) - 1


def test_music_pipeline_with_lyrics_no_transcription(song, monkeypatch):
    monkeypatch.setenv('TRANSCRIBE_BACKEND', 'none')
    lyrics = '\n'.join(['I see the light', 'over the water', 'calling me home', 'through the storm'] * 3)
    st = Settings(mode='music', increment='10')
    doc = prepare_source(Inputs(audio=('song.wav', song.read_bytes()), text=lyrics), st)
    assert doc.audio and doc.timed and doc.units[-1].end <= doc.audio.duration + 0.01
    assert any('approximate' in n or 'spread' in n for n in doc.notes)
    client = FakeClient()
    res = run(doc, st, Writer(client=client))
    assert len(res.segments) == 5
    seg_prompt = next(c['prompt'] for c in client.calls if c['schema'] == 'SegmentsOut')
    assert 'BPM' in seg_prompt and 'Downbeats' in seg_prompt and 'Music:' in seg_prompt
    assert target_beats(doc, st) >= 6


def test_music_without_lyrics_uses_sections(song, monkeypatch):
    monkeypatch.setenv('TRANSCRIBE_BACKEND', 'none')
    doc = prepare_source(Inputs(audio=('song.wav', song.read_bytes())), Settings(mode='music', increment='full'))
    assert doc.units[0].text.startswith('[Instrumental')
    res = run(doc, Settings(mode='music', increment='full'), Writer(client=FakeClient()))
    assert res.plan.segments[-1].end == pytest.approx(doc.audio.duration)


def test_story_audio_uses_transcript(song, monkeypatch):
    monkeypatch.setattr(transcribe, 'backend', lambda: 'openai')
    monkeypatch.setattr(transcribe, 'transcribe', lambda p: [Unit(idx=i, text=f'Spoken line {i} about the storm.', start=i * 6.0, end=i * 6.0 + 5.5) for i in range(8)])
    doc = prepare_source(Inputs(audio=('talk.wav', song.read_bytes())), Settings(mode='story'))
    assert doc.kind == 'audio' and doc.timed and doc.audio is None
    res = run(doc, Settings(increment='15'), Writer(client=FakeClient()))
    assert res.plan.total_seconds == pytest.approx(47.5)


def test_story_audio_without_backend_errors(song, monkeypatch):
    monkeypatch.setenv('TRANSCRIBE_BACKEND', 'none')
    with pytest.raises(ingest.IngestError):
        prepare_source(Inputs(audio=('talk.wav', song.read_bytes())), Settings(mode='story'))


# --------------------------------------------------------------------------- web app
def test_web_app_flow(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import script_studio.app as appmod
    from script_studio.jobs import JobStore
    monkeypatch.setattr(appmod, 'store', JobStore(str(tmp_path)))
    fake = FakeClient()
    monkeypatch.setattr(appmod, 'writer_factory', lambda model='': Writer(client=fake))
    c = TestClient(appmod.app)
    assert c.get('/').status_code == 200 and 'Script Studio' in c.get('/').text
    assert c.get('/api/config').json()['increments'][-1] == 'full'
    assert c.post('/api/jobs', data={'settings': '{}'}).status_code == 400
    r = c.post('/api/jobs', data={'settings': json.dumps({'increment': '15', 'target_seconds': 60})},
               files=[('files', ('story.pdf', pdf_bytes(STORY), 'application/pdf'))])
    jid = r.json()['id']
    for _ in range(100):
        job = c.get(f'/api/jobs/{jid}').json()
        if job['status'] in ('done', 'error'):
            break
        time.sleep(0.05)
    assert job['status'] == 'done', job.get('error')
    assert len(job['result']['segments']) == 4 and job['source']['kind'] == 'pdf'
    for fmt in ('pdf', 'fountain', 'md', 'csv', 'prompts', 'json'):
        resp = c.get(f'/api/jobs/{jid}/export/{fmt}')
        assert resp.status_code == 200 and 'attachment' in resp.headers['content-disposition']
    seg = job['result']['segments'][1]
    seg['shots'][0]['prompt'] = 'Edited prompt'
    seg['shots'][0]['end'] = seg['shots'][0]['start']  # broken timing gets re-tiled
    e = c.put(f'/api/jobs/{jid}/segments/1', json=seg).json()
    assert e['shots'][0]['prompt'] == 'Edited prompt' and e['shots'][-1]['end'] == 30.0
    rw = c.post(f'/api/jobs/{jid}/segments/2/rewrite', json={'note': 'more rain'})
    assert rw.status_code == 200 and 'more rain' in fake.calls[-1]['prompt']
    assert any(j['id'] == jid for j in c.get('/api/jobs').json())
    # persisted to disk and reloadable
    appmod.store.jobs.clear()
    assert c.get(f'/api/jobs/{jid}').json()['result']['segments'][1]['shots'][0]['prompt'] == 'Edited prompt'


def test_transcription_falls_back_to_cpu(monkeypatch):
    from script_studio import transcribe as tr
    calls = []

    class Seg:
        def __init__(self, t): self.text, self.start, self.end = t, 0.0, 1.0

    class Model:
        def __init__(self, device): self.device = device

        def transcribe(self, path, vad_filter=True):
            def gen():
                if self.device != 'cpu':
                    raise RuntimeError('Library cublas64_12.dll is not found or cannot be loaded')
                yield Seg(' hello ')
            return gen(), None
    monkeypatch.setattr(tr, '_FW_MODEL', None)
    monkeypatch.setattr(tr, '_load_fw', lambda d: calls.append(d) or Model(d))
    monkeypatch.setenv('WHISPER_DEVICE', 'auto')
    out = tr._faster_whisper('x.wav')
    assert [u.text for u in out] == ['hello'] and calls == ['auto', 'cpu']
    monkeypatch.setattr(tr, '_FW_MODEL', None)
    monkeypatch.delenv('WHISPER_DEVICE')
    calls.clear()
    tr._faster_whisper('x.wav')
    assert calls == ['cpu']


def test_health_version_and_shutdown_guard(monkeypatch):
    from fastapi.testclient import TestClient
    import script_studio.app as appmod
    c = TestClient(appmod.app)
    assert c.get('/api/health').json()['version']
    assert c.get('/api/config').json()['version']
    assert c.post('/api/shutdown', headers={'X-Script-Studio': 'replace'}).status_code == 403   # only in the desktop app
    monkeypatch.setattr(appmod.userconfig, 'is_frozen', lambda: True)
    assert c.post('/api/shutdown').status_code == 403                                             # needs the header


def test_stale_running_job_is_listed_as_stopped(tmp_path):
    from script_studio.jobs import JobStore
    old = JobStore(str(tmp_path))
    job = old.create(Settings())
    job.status, job.title = 'running', 'Saved mid-run'
    old.save(job)
    assert old.list()[0]['status'] == 'running'          # still running in this server
    fresh = JobStore(str(tmp_path))                     # the server stopped and started again
    assert fresh.list()[0]['status'] == 'error'
    assert fresh.get(job.id).status == 'error'
