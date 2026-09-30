"""Headless runner: batch-upload files to a running ComfyUI and queue every segment.

    python tools/queue_all.py --project my_video --audio song.mp3 \
        --images storyboard/*.png [--script script.txt] [--lyrics lyrics.txt] \
        [--server http://127.0.0.1:8188] [--max-quality]

Uploads go through the node pack's /musicvideo/upload route in batches of 50.
Segments are queued in order (0..N-1); ComfyUI renders them sequentially and
the save node writes output/music_video/<project>/<project>_final.mp4 after the last one.
"""
import argparse
import glob
import json
import os
import time
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
API_WORKFLOW = os.path.join(HERE, "..", "workflows", "music_video_lipsync_api.json")
BATCH = 50


def _multipart(fields, files):
    boundary = uuid.uuid4().hex
    body = bytearray()
    for k, v in fields.items():
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    for path in files:
        name = os.path.basename(path)
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"{name}\"\r\n"
                 "Content-Type: application/octet-stream\r\n\r\n").encode()
        with open(path, "rb") as f:
            body += f.read()
        body += b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _request(url, data=None, content_type="application/json"):
    req = urllib.request.Request(url, data=data, headers={"Content-Type": content_type} if data else {})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def upload(server, project, kind, paths):
    paths = sorted(paths)
    for i in range(0, len(paths), BATCH):
        chunk = paths[i:i + BATCH]
        body, ctype = _multipart({"project": project, "kind": kind}, chunk)
        res = _request(f"{server}/musicvideo/upload", body, ctype)
        print(f"  {kind}: batch {i // BATCH + 1} -> {len(res['saved'])} saved, {len(res['rejected'])} rejected")


def expand(patterns):
    out = []
    for p in patterns or []:
        out.extend(glob.glob(p) or [p])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--project", default="my_music_video")
    ap.add_argument("--audio", nargs="*", help="full song (recommended) or the 5 s clips")
    ap.add_argument("--images", nargs="*", help="storyboard images, named in story order")
    ap.add_argument("--script", help="optional .txt/.json/.csv script")
    ap.add_argument("--lyrics", help="optional lyrics .txt")
    ap.add_argument("--style", help="override the visual style text")
    ap.add_argument("--source", default="auto", choices=["auto", "uploaded_script", "claude_vision", "template"])
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--start", type=int, default=0, help="first segment to queue (resume)")
    ap.add_argument("--max-quality", action="store_true", help="skip the 4-step LoRA: 20 steps, cfg 6")
    args = ap.parse_args()
    server = args.server.rstrip("/")

    print("Uploading…")
    if args.audio:
        upload(server, args.project, "audio", expand(args.audio))
    if args.images:
        upload(server, args.project, "images", expand(args.images))
    if args.script:
        upload(server, args.project, "script", [args.script])
    status = _request(f"{server}/musicvideo/status?project={args.project}")
    print(f"Project: {len(status['images'])} images, {len(status['audio'])} audio, {len(status['script'])} script")

    with open(API_WORKFLOW, "r", encoding="utf-8") as f:
        wf = json.load(f)
    by_type = {n["class_type"]: k for k, n in wf.items()}
    sb = wf[by_type["MVScriptBuilder"]]["inputs"]
    sb["project"] = args.project
    sb["script_source"] = args.source
    if args.lyrics:
        with open(args.lyrics, "r", encoding="utf-8") as f:
            sb["lyrics"] = f.read()
    if args.style:
        sb["style"] = args.style
    ld = wf[by_type["MVSegmentLoader"]]["inputs"]
    ld["width"], ld["height"] = args.width, args.height
    if args.max_quality:
        lora_id = by_type["LoraLoaderModelOnly"]
        ms = wf[by_type["ModelSamplingSD3"]]["inputs"]
        ms["model"] = wf[lora_id]["inputs"]["model"]
        del wf[lora_id]
        ks = wf[by_type["KSampler"]]["inputs"]
        ks["steps"], ks["cfg"] = 20, 6.0

    # Queue segment 0 first; its script-builder log tells us how many segments exist.
    client_id = uuid.uuid4().hex
    total = None
    idx = args.start
    while total is None or idx < total:
        ld["segment_index"] = idx
        res = _request(f"{server}/prompt", json.dumps({"prompt": wf, "client_id": client_id}).encode())
        if res.get("node_errors"):
            raise SystemExit(f"ComfyUI rejected the workflow: {json.dumps(res['node_errors'], indent=2)}")
        print(f"queued segment {idx} (prompt {res['prompt_id']})")
        if total is None:
            total = wait_for_segment_count(server, res["prompt_id"])
            print(f"Song has {total} segments.")
        idx += 1
    print(f"All queued. Final video: ComfyUI/output/music_video/{args.project}/{args.project}_final.mp4")


def wait_for_segment_count(server, prompt_id):
    """Poll history for the first job, then read the plan the script builder saved."""
    while True:
        hist = _request(f"{server}/history/{prompt_id}")
        if prompt_id in hist:
            st = hist[prompt_id].get("status", {})
            if st.get("status_str") == "error":
                raise SystemExit(f"Segment 0 failed: {json.dumps(st.get('messages', [])[-1:], indent=2)}")
            for out in hist[prompt_id].get("outputs", {}).values():
                for img in out.get("images", []):
                    sub = img.get("subfolder", "")
                    # subfolder = music_video/<project>/segments -> script_used.json is one level up
                    view = f"{server}/view?filename=script_used.json&type=output&subfolder=" + \
                        urllib.request.quote(os.path.dirname(sub) if sub.endswith("segments") else sub)
                    with urllib.request.urlopen(view) as r:
                        return len(json.loads(r.read())["segments"])
        time.sleep(5)


if __name__ == "__main__":
    main()
