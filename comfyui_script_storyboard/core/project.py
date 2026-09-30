"""On-disk storyboard project.

A project folder is the hand-off point between workflow stages, so a
1000-page script can be planned once and rendered over many queue runs (or
many days) with resume support. Progress is derived from which files exist,
so there is no shared state to corrupt if a run is interrupted.

    <output>/storyboards/<project>/
        project.json        settings, character bible, reference images
        screenplay.json     parsed script
        shots.json          ordered shot list (index == storyboard order)
        references/         up to 5 user reference images
        characters/         generated/assigned character reference sheets
        images/             shot_000000.png ...
        audio/              shot_000000.wav (dialogue)
        clips/              shot_000000.mp4 (lip-synced dialogue clips)
        exports/            final video, .srt, .csv edit list
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from typing import List, Optional

from .characters import CharacterBible
from .parser import Screenplay
from .shots import Shot

MAX_REFERENCES = 5
REF_USAGES = ("reference", "init_image", "use_as_frame")


def safe_name(name: str) -> str:
    n = re.sub(r"[^A-Za-z0-9_\-]+", "_", (name or "").strip()).strip("_")
    return n or "storyboard"


def _atomic_write_json(path: str, data):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


class Project:
    SUBDIRS = ("references", "characters", "images", "audio", "clips", "exports", "segments")

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.meta: dict = {}
        self._shots: Optional[List[Shot]] = None
        self._bible: Optional[CharacterBible] = None

    # ---------- creation / loading ----------
    @classmethod
    def create(cls, root: str, screenplay: Screenplay, bible: CharacterBible,
               shots: List[Shot], settings: dict, overwrite_shots: bool = True):
        p = cls(root)
        os.makedirs(p.root, exist_ok=True)
        for sd in cls.SUBDIRS:
            os.makedirs(os.path.join(p.root, sd), exist_ok=True)
        old = {}
        if os.path.exists(p.path("project.json")):
            with open(p.path("project.json"), encoding="utf-8") as f:
                old = json.load(f)
        # keep references + reference_image assignments across re-plans
        refs = old.get("references", [])
        old_chars = (old.get("bible") or {}).get("characters", {})
        for name, c in bible.characters.items():
            if not c.reference_image and old_chars.get(name, {}).get("reference_image"):
                c.reference_image = old_chars[name]["reference_image"]
        p.meta = {
            "version": 1,
            "title": screenplay.title,
            "created": old.get("created", time.time()),
            "updated": time.time(),
            "settings": {**old.get("settings", {}), **settings},
            "pages": screenplay.pages,
            "scene_count": len(screenplay.scenes),
            "shot_count": len(shots),
            "bible": bible.to_dict(),
            "references": refs,
        }
        _atomic_write_json(p.path("screenplay.json"), screenplay.to_dict())
        if overwrite_shots or not os.path.exists(p.path("shots.json")):
            _atomic_write_json(p.path("shots.json"), [s.to_dict() for s in shots])
        p._shots, p._bible = shots, bible
        p.save_meta()
        return p

    @classmethod
    def load(cls, root: str):
        p = cls(root)
        if not os.path.exists(p.path("project.json")):
            raise FileNotFoundError(
                f"No storyboard project at {p.root}. Run the planning workflow first.")
        with open(p.path("project.json"), encoding="utf-8") as f:
            p.meta = json.load(f)
        return p

    def save_meta(self):
        self.meta["updated"] = time.time()
        if self._bible is not None:
            self.meta["bible"] = self._bible.to_dict()
        _atomic_write_json(self.path("project.json"), self.meta)

    def save_shots(self, shots: List[Shot]):
        self._shots = shots
        self.meta["shot_count"] = len(shots)
        _atomic_write_json(self.path("shots.json"), [s.to_dict() for s in shots])
        self.save_meta()

    # ---------- accessors ----------
    def path(self, *parts) -> str:
        return os.path.join(self.root, *parts)

    @property
    def settings(self) -> dict:
        return self.meta.get("settings", {})

    @property
    def shots(self) -> List[Shot]:
        if self._shots is None:
            with open(self.path("shots.json"), encoding="utf-8") as f:
                self._shots = [Shot(**d) for d in json.load(f)]
        return self._shots

    @property
    def bible(self) -> CharacterBible:
        if self._bible is None:
            self._bible = CharacterBible.from_dict(self.meta["bible"])
        return self._bible

    def screenplay(self) -> Screenplay:
        with open(self.path("screenplay.json"), encoding="utf-8") as f:
            return Screenplay.from_dict(json.load(f))

    def image_path(self, idx: int) -> str:
        return self.path("images", f"shot_{idx:06d}.png")

    def audio_path(self, idx: int) -> str:
        return self.path("audio", f"shot_{idx:06d}.wav")

    def clip_path(self, idx: int) -> str:
        return self.path("clips", f"shot_{idx:06d}.mp4")

    def has_image(self, idx: int) -> bool:
        return os.path.exists(self.image_path(idx)) or self.frame_override(self.shots[idx]) is not None

    def next_missing(self, start: int = 0, predicate=None) -> Optional[int]:
        shots = self.shots
        for i in range(max(0, start), len(shots)):
            if predicate is None and not os.path.exists(self.image_path(i)) \
                    and self.frame_override(shots[i]) is None:
                return i
            if predicate is not None and predicate(i):
                return i
        return None

    def status(self) -> dict:
        shots = self.shots
        imgs = set(os.listdir(self.path("images"))) if os.path.isdir(self.path("images")) else set()
        auds = set(os.listdir(self.path("audio"))) if os.path.isdir(self.path("audio")) else set()
        clips = set(os.listdir(self.path("clips"))) if os.path.isdir(self.path("clips")) else set()
        dlg = [s for s in shots if s.dialogue]
        return {
            "shots": len(shots),
            "images": sum(1 for s in shots if f"shot_{s.index:06d}.png" in imgs),
            "frames_from_references": sum(1 for s in shots if self.frame_override(s)),
            "missing_frames": sum(1 for s in shots if f"shot_{s.index:06d}.png" not in imgs
                                  and not self.frame_override(s)),
            "dialogue_shots": len(dlg),
            "audio": sum(1 for s in dlg if f"shot_{s.index:06d}.wav" in auds),
            "lipsync_clips": sum(1 for s in dlg if f"shot_{s.index:06d}.mp4" in clips),
            "runtime_seconds": round(sum(s.duration for s in shots), 1),
        }

    # ---------- reference images (max 5) ----------
    @property
    def references(self) -> List[dict]:
        return self.meta.setdefault("references", [])

    def set_reference(self, slot: int, src_path: Optional[str], bind: str,
                      usage: str = "reference", weight: float = 1.0, denoise: float = 0.6):
        """slot 1..5. src_path=None clears the slot."""
        if not 1 <= slot <= MAX_REFERENCES:
            raise ValueError(f"Reference slot must be 1..{MAX_REFERENCES}")
        if usage not in REF_USAGES:
            raise ValueError(f"usage must be one of {REF_USAGES}")
        rid = f"ref{slot}"
        refs = [r for r in self.references if r["id"] != rid]
        if src_path:
            ext = os.path.splitext(src_path)[1].lower() or ".png"
            rel = os.path.join("references", rid + ext)
            for old in os.listdir(self.path("references")):
                if old.startswith(rid + "."):
                    os.remove(self.path("references", old))
            if os.path.abspath(src_path) != self.path(rel):
                shutil.copyfile(src_path, self.path(rel))
            refs.append({"id": rid, "file": rel, "bind": parse_bind(bind),
                         "bind_raw": bind, "usage": usage, "weight": float(weight),
                         "denoise": float(denoise)})
        refs.sort(key=lambda r: r["id"])
        self.meta["references"] = refs
        # Character-bound references become that character's canonical reference image.
        for r in refs:
            if r["bind"]["type"] == "character":
                name = self.bible.resolve(r["bind"]["value"])
                if name:
                    self.bible.characters[name].reference_image = r["file"]
        self.save_meta()

    def references_for_shot(self, shot: Shot) -> List[dict]:
        """Ordered references that apply to a shot: characters in the shot
        (primary first), then location, scene range, shot id, then global."""
        out, seen = [], set()

        def push(r):
            if r["id"] not in seen:
                seen.add(r["id"])
                out.append(r)

        refs = self.references
        bible = self.bible
        for name in shot.characters:
            matched = False
            for r in refs:
                b = r["bind"]
                if b["type"] == "character" and bible.resolve(b["value"]) == name:
                    push(r)
                    matched = True
            c = bible.characters.get(name)
            if not matched and c and c.reference_image and os.path.exists(self.path(c.reference_image)):
                push({"id": f"char:{name}", "file": c.reference_image, "usage": "reference",
                      "weight": 1.0, "denoise": 0.6, "bind": {"type": "character", "value": name}})
        for r in refs:
            b = r["bind"]
            if b["type"] == "location" and b["value"] and b["value"].upper() in (shot.location or shot.heading).upper():
                push(r)
            elif b["type"] == "scenes" and b["start"] <= shot.scene_index + 1 <= b["end"]:
                push(r)
            elif b["type"] == "shot" and b["value"].upper() in (shot.id, str(shot.index)):
                push(r)
        for r in refs:
            if r["bind"]["type"] in ("all", "style"):
                push(r)
        return out

    def frame_override(self, shot: Shot) -> Optional[str]:
        """A reference with usage=use_as_frame bound to this exact shot/scene
        replaces the generated image."""
        for r in self.references:
            if r.get("usage") != "use_as_frame":
                continue
            b = r["bind"]
            if (b["type"] == "shot" and b["value"].upper() in (shot.id, str(shot.index))) or \
               (b["type"] == "scenes" and b["start"] <= shot.scene_index + 1 <= b["end"]
                    and shot.shot_in_scene == 0):
                p = self.path(r["file"])
                if os.path.exists(p):
                    return p
        return None

    def frame_for_shot(self, shot: Shot) -> Optional[str]:
        p = self.frame_override(shot)
        if p:
            return p
        p = self.image_path(shot.index)
        return p if os.path.exists(p) else None


def parse_bind(bind: str) -> dict:
    """Binding syntax for a reference image:
        all | style | character:NAME | location:TEXT | scenes:3-10 | scene:4 | shot:SC0003_SH002 | shot:42
    A bare name (e.g. "JOHN") is treated as character:JOHN."""
    b = (bind or "all").strip()
    if not b or b.lower() in ("all", "global", "*"):
        return {"type": "all"}
    if b.lower() == "style":
        return {"type": "style"}
    if ":" in b:
        kind, val = b.split(":", 1)
        kind, val = kind.strip().lower(), val.strip()
        if kind in ("character", "char"):
            return {"type": "character", "value": val.upper()}
        if kind in ("location", "loc", "set"):
            return {"type": "location", "value": val.upper()}
        if kind in ("scene", "scenes"):
            m = re.match(r"^(\d+)\s*(?:-\s*(\d+))?$", val)
            if not m:
                raise ValueError(f"Bad scene range: {val!r}")
            a = int(m.group(1))
            return {"type": "scenes", "start": a, "end": int(m.group(2) or a)}
        if kind == "shot":
            return {"type": "shot", "value": val.upper()}
        raise ValueError(f"Unknown reference binding: {bind!r}")
    return {"type": "character", "value": b.upper()}
