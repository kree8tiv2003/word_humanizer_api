"""Background jobs with progress, persisted to disk so finished scripts survive a restart."""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from pydantic import BaseModel

from .models import ScriptResult, Settings, SourceDoc

JOBS_DIR = os.getenv('SCRIPT_JOBS_DIR', os.path.join(os.path.dirname(__file__), '..', 'script_jobs'))
_pool = ThreadPoolExecutor(max_workers=int(os.getenv('SCRIPT_WORKERS', '2')))


class Job(BaseModel):
    id: str
    created: float
    status: str = 'queued'        # queued, running, done, error
    stage: str = 'Queued'
    progress: float = 0.0
    log: list[str] = []
    error: str = ''
    title: str = ''
    settings: Settings
    result: Optional[ScriptResult] = None
    doc: Optional[SourceDoc] = None


class JobStore:
    def __init__(self, root: str = JOBS_DIR):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.RLock()

    def _path(self, jid: str) -> str:
        return os.path.join(self.root, f'{jid}.json')

    def create(self, settings: Settings) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], created=time.time(), settings=settings)
        with self.lock:
            self.jobs[job.id] = job
        return job

    def get(self, jid: str) -> Optional[Job]:
        if not jid.isalnum():
            return None
        with self.lock:
            if jid in self.jobs:
                return self.jobs[jid]
            p = self._path(jid)
            if os.path.exists(p):
                with open(p, encoding='utf-8') as f:
                    job = Job.model_validate_json(f.read())
                if job.status in ('queued', 'running'):
                    job.status, job.error = 'error', 'The server restarted while this job was running. Use Resume.'
                self.jobs[jid] = job
                return job
        return None

    def save(self, job: Job) -> None:
        with self.lock:
            tmp = self._path(job.id) + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                f.write(job.model_dump_json())
            os.replace(tmp, self._path(job.id))

    def list(self, limit: int = 30) -> list[dict]:
        out = []
        for name in os.listdir(self.root):
            if name.endswith('.json'):
                try:
                    with open(os.path.join(self.root, name), encoding='utf-8') as f:
                        d = json.load(f)
                    out.append({'id': d['id'], 'title': d.get('title') or 'Untitled', 'status': d['status'],
                                'created': d['created'], 'mode': d['settings']['mode'], 'increment': d['settings']['increment']})
                except Exception:
                    continue
        with self.lock:
            for j in self.jobs.values():
                if not any(o['id'] == j.id for o in out):
                    out.append({'id': j.id, 'title': j.title or 'Untitled', 'status': j.status, 'created': j.created,
                                'mode': j.settings.mode, 'increment': j.settings.increment})
        return sorted(out, key=lambda d: -d['created'])[:limit]

    def submit(self, job: Job, fn) -> None:
        def wrapper():
            job.status = 'running'
            try:
                fn(job)
                job.status, job.stage, job.progress = 'done', 'Script complete', 1.0
            except Exception as e:  # report every failure on the job itself
                job.status, job.error = 'error', str(e) or e.__class__.__name__
                job.log.append('ERROR: ' + job.error)
                if os.getenv('SCRIPT_DEBUG'):
                    traceback.print_exc()
            finally:
                self.save(job)
        _pool.submit(wrapper)

    @staticmethod
    def progress(job: Job, msg: str, frac: Optional[float] = None) -> None:
        job.stage = msg
        if frac is not None:
            job.progress = max(job.progress, min(1.0, frac))
        if not job.log or job.log[-1] != msg:
            job.log.append(msg)
