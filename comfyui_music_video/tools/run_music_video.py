#!/usr/bin/env python3
"""Render the whole "God Inside" music video on a running ComfyUI server (stdlib only).

  python3 tools/run_music_video.py --server http://127.0.0.1:8188            # stills -> clips -> final mp4
  python3 tools/run_music_video.py --stage stills --scenes 1,2,3             # only some scenes / one stage
  python3 tools/run_music_video.py --stage clips --scenes 9-12

Steps: upload the references and song, generate the 27 start frames (Qwen-Image-Edit-2509; Scene 01
first because it is the venue plate for the others), render each scene with LTX-2 audio-to-video,
download everything to output/, then call assemble_video.py to cut it to the song.
"""
import argparse
import json
import mimetypes
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")


class Comfy:
    def __init__(self, url):
        self.url = url.rstrip("/")
        self.client_id = str(uuid.uuid4())

    def _req(self, path, data=None, headers=None):
        req = urllib.request.Request(self.url + path, data=data, headers=headers or {})
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()

    def upload(self, path, name=None):
        name = name or os.path.basename(path)
        boundary = uuid.uuid4().hex
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
        parts = []
        for k, v in (("overwrite", "true"), ("type", "input")):
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
                     f"Content-Type: {mime}\r\n\r\n".encode() + open(path, "rb").read() + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        self._req("/upload/image", b"".join(parts), {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        print("  uploaded", name)

    def run(self, prompt, label):
        body = json.dumps({"prompt": prompt, "client_id": self.client_id}).encode()
        try:
            pid = json.loads(self._req("/prompt", body, {"Content-Type": "application/json"}))["prompt_id"]
        except urllib.error.HTTPError as e:
            raise SystemExit(f"{label}: ComfyUI rejected the prompt:\n{e.read().decode()}")
        t0 = time.time()
        while True:
            hist = json.loads(self._req(f"/history/{pid}"))
            if pid in hist:
                h = hist[pid]
                status = h.get("status", {})
                if status.get("status_str") == "error":
                    msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                    raise SystemExit(f"{label} failed: {json.dumps(msgs, indent=1)[:3000]}")
                if status.get("completed", True):
                    print(f"  {label} done in {time.time() - t0:.0f}s")
                    return h.get("outputs", {})
            time.sleep(3)

    def download(self, outputs, dest_dir, rename=None):
        files = [f for o in outputs.values() for v in o.values() if isinstance(v, list)
                 for f in v if isinstance(f, dict) and "filename" in f]
        if not files:
            raise SystemExit("no output file returned")
        f = files[-1]
        q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""),
                                    "type": f.get("type", "output")})
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, rename or f["filename"])
        with open(dest, "wb") as fh:
            fh.write(self._req(f"/view?{q}"))
        return dest


def parse_scenes(spec):
    if not spec:
        return list(range(1, 28))
    ids = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        ids += list(range(int(a), int(b or a) + 1))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--stage", choices=["all", "stills", "clips", "assemble"], default="all")
    ap.add_argument("--scenes", help="e.g. 1,2,9-12 (default: all 27)")
    ap.add_argument("--skip-existing", action="store_true", help="skip scenes already downloaded to output/")
    args = ap.parse_args()
    ids = parse_scenes(args.scenes)
    c = Comfy(args.server)

    if args.stage != "assemble":
        print("uploading inputs")
        for f in ("God_Inside.mp3", "god_inside_ref_closeup.png", "god_inside_ref_sheet.png"):
            c.upload(os.path.join(ROOT, "input", f))

    if args.stage in ("all", "stills"):
        print("generating start frames")
        order = sorted(ids, key=lambda i: i != 1)  # the venue plate first
        if 1 not in ids and not os.path.exists(os.path.join(OUT, "stills", "god_inside_scene_01.png")):
            raise SystemExit("scene 01 (the venue plate) must be generated before the other stills")
        for i in order:
            name = f"god_inside_scene_{i:02d}.png"
            local = os.path.join(OUT, "stills", name)
            if i == order[0] and i != 1:
                c.upload(os.path.join(OUT, "stills", "god_inside_scene_01.png"))
            if not (args.skip_existing and os.path.exists(local)):
                prompt = json.load(open(os.path.join(ROOT, "api", "stills", f"scene_{i:02d}.json")))
                c.download(c.run(prompt, f"still {i:02d}"), os.path.join(OUT, "stills"), name)
            c.upload(local, name)  # scene 01 is uploaded here before any later still needs it

    if args.stage in ("all", "clips"):
        print("rendering LTX-2 scene clips")
        for i in ids:
            name = f"god_inside_scene_{i:02d}.png"
            local = os.path.join(OUT, "stills", name)
            if not os.path.exists(local):
                raise SystemExit(f"missing start frame {local}: run --stage stills --scenes {i} first")
            c.upload(local, name)
            dest = os.path.join(OUT, "clips", f"clip_{i:02d}.mp4")
            if args.skip_existing and os.path.exists(dest):
                continue
            prompt = json.load(open(os.path.join(ROOT, "api", "clips", f"clip_{i:02d}.json")))
            c.download(c.run(prompt, f"clip {i:02d}"), os.path.join(OUT, "clips"), f"clip_{i:02d}.mp4")

    if args.stage in ("all", "assemble"):
        subprocess.run([sys.executable, os.path.join(ROOT, "tools", "assemble_video.py"),
                        "--clips", os.path.join(OUT, "clips"), "--stills", os.path.join(OUT, "stills")], check=True)


if __name__ == "__main__":
    main()
