"""HTTP routes behind the node's batch upload buttons (max 50 files per request)."""
import os
import shutil

from aiohttp import web

import folder_paths
from server import PromptServer

from . import mv_core as core

routes = PromptServer.instance.routes


def _status(project):
    input_root = folder_paths.get_input_directory()
    output_root = folder_paths.get_output_directory()
    seg_dir = os.path.join(core.project_output_dir(output_root, project), "segments")
    rendered = sorted(f for f in os.listdir(seg_dir) if f.endswith(".mp4")) if os.path.isdir(seg_dir) else []
    return {
        "project": project,
        "images": core.list_files(input_root, project, "images"),
        "audio": core.list_files(input_root, project, "audio"),
        "script": core.list_files(input_root, project, "script"),
        "rendered_segments": len(rendered),
    }


@routes.post("/musicvideo/upload")
async def mv_upload(request):
    reader = await request.multipart()
    project, kind, saved, rejected = None, None, [], []
    target = None
    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name == "project":
            project = core.safe_name(await part.text())
            continue
        if part.name == "kind":
            kind = (await part.text()).strip()
            if kind not in core.KIND_EXTS:
                return web.json_response({"error": f"unknown kind '{kind}'"}, status=400)
            continue
        if part.name != "files":
            continue
        if project is None or kind is None:
            return web.json_response({"error": "send 'project' and 'kind' fields before the files"}, status=400)
        if len(saved) >= core.MAX_FILES_PER_BATCH:
            for f in saved:  # all-or-nothing: undo this batch
                os.remove(os.path.join(target, f))
            return web.json_response({"error": f"max {core.MAX_FILES_PER_BATCH} files per batch"}, status=400)
        name = core.safe_filename(part.filename)
        if os.path.splitext(name)[1].lower() not in core.KIND_EXTS[kind]:
            rejected.append(part.filename)
            continue
        target = core.kind_dir(folder_paths.get_input_directory(), project, kind)
        os.makedirs(target, exist_ok=True)
        with open(os.path.join(target, name), "wb") as f:
            while True:
                chunk = await part.read_chunk(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        saved.append(name)
    if project is None:
        return web.json_response({"error": "missing project"}, status=400)
    return web.json_response({"saved": saved, "rejected": rejected, "status": _status(project)})


@routes.get("/musicvideo/status")
async def mv_status(request):
    project = core.safe_name(request.rel_url.query.get("project", ""))
    return web.json_response(_status(project))


@routes.post("/musicvideo/clear")
async def mv_clear(request):
    data = await request.json()
    project = core.safe_name(data.get("project", ""))
    kind = data.get("kind")
    if kind not in core.KIND_EXTS:
        return web.json_response({"error": "unknown kind"}, status=400)
    d = core.kind_dir(folder_paths.get_input_directory(), project, kind)
    if os.path.isdir(d):
        shutil.rmtree(d)
    return web.json_response({"status": _status(project)})
