"""Script Studio web app.

Run:  uvicorn script_studio.app:app --reload
"""
from __future__ import annotations

import json
import os
import re

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from . import userconfig

userconfig.apply()   # saved API keys -> environment, before anything reads them

from . import transcribe  # noqa: E402
from .exporters import FORMATS
from .ingest import IngestError
from .jobs import Job, JobStore
from .models import SegmentOut, Settings
from .pipeline import Inputs, prepare_source, run
from .planner import tile_shots
from ._version import VERSION
from .writer import Writer, WriterError, current_model

STATIC = os.path.join(os.path.dirname(__file__), 'static')
MAX_UPLOAD = int(os.getenv('MAX_UPLOAD_MB', '200')) * 1024 * 1024

app = FastAPI(title='Script Studio', version='1.0')
app.mount('/static', StaticFiles(directory=STATIC), name='static')
store = JobStore()


@app.middleware('http')
async def same_origin_only(request: Request, call_next):
    """The app listens on this computer only; refuse write requests sent by other websites."""
    if request.method not in ('GET', 'HEAD', 'OPTIONS'):
        origin = request.headers.get('origin')
        if origin and origin.split('://', 1)[-1] != request.headers.get('host', ''):
            return JSONResponse({'detail': 'Cross-site request refused.'}, status_code=403)
    return await call_next(request)


def writer_factory(model: str = '') -> Writer:
    """Swapped out in tests."""
    return Writer(model=model or None)


@app.get('/')
def index():
    return FileResponse(os.path.join(STATIC, 'index.html'))


@app.get('/api/config')
def config():
    return {'version': VERSION, 'model': current_model(), 'has_api_key': bool(os.getenv('ANTHROPIC_API_KEY')), 'desktop': userconfig.is_frozen(),
            'transcription': transcribe.backend(), 'max_upload_mb': MAX_UPLOAD // (1024 * 1024),
            'increments': ['5', '10', '15', '30', '60', 'full'],
            'generators': ['any', 'veo', 'sora', 'kling', 'runway', 'luma']}


@app.get('/api/health')
def health():
    return {'app': 'script-studio', 'ok': True, 'version': VERSION}


@app.post('/api/shutdown')
def shutdown(request: Request):
    """Lets a newer copy of the desktop app replace an older one that is still running."""
    if not userconfig.is_frozen() or request.headers.get('x-script-studio') != 'replace':
        raise HTTPException(403, 'Not allowed.')
    import threading
    threading.Timer(0.5, lambda: os._exit(0)).start()
    return {'ok': True}


@app.get('/api/settings')
def get_settings():
    return userconfig.status()


@app.post('/api/settings')
def set_settings(body: dict = Body(...)):
    userconfig.save({k: str(v) for k, v in body.items() if isinstance(v, (str, int, float))})
    return userconfig.status()


@app.post('/api/settings/test')
def test_key():
    """Check the Anthropic key with a free request (lists models; no tokens used)."""
    if not os.getenv('ANTHROPIC_API_KEY'):
        return {'ok': False, 'message': 'No Anthropic key saved yet.'}
    try:
        import anthropic
        anthropic.Anthropic(max_retries=1, timeout=20).models.list(limit=1)
        return {'ok': True, 'message': 'Your Anthropic key works.'}
    except Exception as e:
        msg = str(e)
        if 'authentication' in msg.lower() or '401' in msg:
            msg = 'That key was rejected. Check that you copied the whole key.'
        return {'ok': False, 'message': msg[:300]}


async def _read(f: UploadFile) -> bytes:
    data = await f.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, f'{f.filename} is larger than {MAX_UPLOAD // (1024 * 1024)} MB.')
    return data


def _job_view(job: Job) -> dict:
    d = job.model_dump(exclude={'doc'})
    if job.doc:
        d['source'] = {'kind': job.doc.kind, 'title': job.doc.title, 'units': len(job.doc.units),
                       'words': job.doc.word_count, 'audio': job.doc.audio.model_dump() if job.doc.audio else None}
    return d


@app.post('/api/jobs')
async def create_job(settings: str = Form('{}'), url: str = Form(''), text: str = Form(''),
                     files: list[UploadFile] = File(default=[]), audio: UploadFile | None = File(default=None)):
    try:
        st = Settings.model_validate(json.loads(settings or '{}'))
    except (ValidationError, json.JSONDecodeError) as e:
        raise HTTPException(422, f'Invalid settings: {e}')
    docs = [(f.filename or 'upload', await _read(f)) for f in files if f and f.filename]
    aud = (audio.filename or 'audio', await _read(audio)) if audio and audio.filename else None
    if not (docs or aud or url.strip() or text.strip()):
        raise HTTPException(400, 'Upload a document or audio file, paste text, or add a link.')
    inp = Inputs(documents=docs, audio=aud, url=url, text=text)
    job = store.create(st)

    def work(job: Job):
        job.doc = prepare_source(inp, st, lambda m: store.progress(job, m, 0.03))
        job.title = job.doc.title
        w = writer_factory(st.model)

        def on_seg(result):
            job.result = result
            job.title = result.bible.title or job.title
            store.save(job)
        run(job.doc, st, w, lambda m, p: store.progress(job, m, p), on_seg)

    store.submit(job, work)
    return {'id': job.id}


@app.get('/api/jobs')
def list_jobs():
    return store.list()


def _get(jid: str) -> Job:
    job = store.get(jid)
    if not job:
        raise HTTPException(404, 'Job not found.')
    return job


@app.get('/api/jobs/{jid}')
def get_job(jid: str):
    return JSONResponse(_job_view(_get(jid)))


@app.post('/api/jobs/{jid}/resume')
def resume(jid: str):
    job = _get(jid)
    if job.status in ('queued', 'running'):
        raise HTTPException(409, 'The job is still running.')
    if not job.doc:
        raise HTTPException(409, 'This job failed before its source was read; start a new one.')
    job.status, job.error = 'queued', ''

    def work(job: Job):
        w = writer_factory(job.settings.model)
        if job.result is None:
            run(job.doc, job.settings, w, lambda m, p: store.progress(job, m, p),
                lambda r: (setattr(job, 'result', r), store.save(job)))
            return
        n = len(job.result.plan.segments)
        for part in w.write_segments(job.result, job.doc, lambda m: store.progress(job, m)):
            job.result.segments.extend(part)
            store.progress(job, f'Wrote {len(job.result.segments)} of {n} segments', 0.2 + 0.8 * len(job.result.segments) / n)
            store.save(job)

    store.submit(job, work)
    return {'id': job.id}


@app.post('/api/jobs/{jid}/segments/{index}/rewrite')
def rewrite(jid: str, index: int, body: dict = Body(default={})):
    job = _get(jid)
    if not job.result or index >= len(job.result.segments) or index < 0:
        raise HTTPException(404, 'Segment not found.')
    if job.status == 'running':
        raise HTTPException(409, 'Wait for the script to finish before rewriting segments.')
    try:
        seg = writer_factory(job.settings.model).rewrite_segment(job.result, job.doc, index, str(body.get('note', ''))[:2000])
    except WriterError as e:
        raise HTTPException(502, str(e))
    job.result.segments[index] = seg
    store.save(job)
    return seg.model_dump()


@app.put('/api/jobs/{jid}/segments/{index}')
def edit_segment(jid: str, index: int, seg: SegmentOut):
    job = _get(jid)
    if not job.result or index >= len(job.result.segments) or index < 0:
        raise HTTPException(404, 'Segment not found.')
    if not seg.shots:
        raise HTTPException(422, 'A segment needs at least one shot.')
    p = job.result.plan.segments[index]
    seg.index = index
    seg.shots.sort(key=lambda s: s.start)
    tile_shots(p.start, p.end, seg.shots)
    job.result.segments[index] = seg
    store.save(job)
    return seg.model_dump()


@app.get('/api/jobs/{jid}/export/{fmt}')
def export(jid: str, fmt: str):
    job = _get(jid)
    if fmt not in FORMATS:
        raise HTTPException(404, 'Unknown format.')
    if not job.result or not job.result.segments:
        raise HTTPException(409, 'Nothing to export yet.')
    ctype, ext, fn = FORMATS[fmt]
    data = fn(job.result)
    name = re.sub(r'[^A-Za-z0-9]+', '-', job.result.bible.title or 'script').strip('-')[:60] or 'script'
    return Response(content=data if isinstance(data, bytes) else data.encode('utf-8'), media_type=ctype,
                    headers={'Content-Disposition': f'attachment; filename="{name}{ext}"'})


@app.exception_handler(IngestError)
def ingest_error(_req, exc: IngestError):
    return JSONResponse({'detail': str(exc)}, status_code=400)
