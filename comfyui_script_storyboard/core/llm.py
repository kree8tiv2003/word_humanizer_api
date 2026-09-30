"""Optional LLM rewrite of shot prompts (one request per scene).

The deterministic planner already produces usable prompts; this pass makes
them read like a cinematographer wrote them. Character descriptions are
passed in and must be kept verbatim so consistency is preserved.

Backends:
  * anthropic - Claude via the official `anthropic` SDK (ANTHROPIC_API_KEY)
  * ollama    - a local Ollama server (native /api/generate endpoint)

Results are cached per scene (keyed by content hash) so a 1000-page script
can be enhanced across several runs without paying twice.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from itertools import groupby
from typing import Callable, List, Optional

from .characters import CharacterBible
from .shots import Shot

LLM_BACKENDS = ("none", "anthropic", "ollama")

SYSTEM = (
    "You are a storyboard artist and cinematographer turning screenplay scenes into text-to-image "
    "prompts. For each shot you receive, write one image prompt of at most 70 words describing a "
    "single frozen frame: shot size and angle, who is in frame and what they are doing, setting, "
    "lighting, mood and composition. Rules: copy every character description in the CHARACTERS "
    "block verbatim wherever that character appears (this keeps characters consistent across "
    "thousands of frames); never put character names, dialogue or any other readable text in the "
    "prompt; keep the given style words at the start; keep the shot size you were given; "
    "return every shot id exactly once.")

SCHEMA = {
    "type": "object",
    "properties": {
        "shots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "prompt": {"type": "string"}},
                "required": ["id", "prompt"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["shots"],
    "additionalProperties": False,
}


def _scene_payload(shots: List[Shot], bible: CharacterBible, style_hint: str) -> str:
    sc = shots[0]
    names = sorted({n for s in shots for n in s.characters})
    chars = "\n".join(f"- {n}: {bible.prompt_fragment(n, sc.scene_index)}" for n in names) or "- (none)"
    lines = []
    for s in shots:
        extra = ""
        if s.dialogue:
            extra = f' | speaker: {s.dialogue["character"]} | line (do not render as text): "{s.dialogue["text"][:200]}"'
        lines.append(f"[{s.id}] size: {s.shot_size} | in frame: {', '.join(s.characters) or 'nobody'}"
                     f" | beat: {s.source_text[:600]}{extra}\n    draft prompt: {s.prompt}")
    return (f"STYLE: {style_hint}\nSCENE HEADING: {sc.heading}\n\nCHARACTERS:\n{chars}\n\n"
            f"SHOTS:\n" + "\n".join(lines))


def _call_anthropic(payload: str, model: str, effort: str) -> Optional[dict]:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM,
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": payload}],
    )
    if resp.stop_reason in ("refusal", "max_tokens"):
        return None
    text = next((b.text for b in resp.content if b.type == "text"), "")
    return json.loads(text) if text else None


def _call_ollama(payload: str, model: str, host: str) -> Optional[dict]:
    body = json.dumps({"model": model, "system": SYSTEM, "prompt": payload,
                       "format": SCHEMA, "stream": False,
                       "options": {"temperature": 0.4}}).encode()
    req = urllib.request.Request(host.rstrip("/") + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    return json.loads(data.get("response") or "{}")


def enhance_prompts(shots: List[Shot], bible: CharacterBible, backend: str, model: str,
                    style_hint: str = "", effort: str = "low", ollama_host: str = "http://127.0.0.1:11434",
                    cache_path: Optional[str] = None, scene_start: int = 1, scene_end: int = -1,
                    progress: Optional[Callable[[int, int], None]] = None) -> dict:
    if backend == "none":
        return {"enhanced": 0, "failed": 0}
    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    groups = [list(g) for _, g in groupby(shots, key=lambda s: s.scene_index)]
    groups = [g for g in groups
              if g[0].scene_index + 1 >= scene_start and (scene_end < 1 or g[0].scene_index + 1 <= scene_end)]
    ok = failed = 0
    errors = []
    for k, group in enumerate(groups):
        payload = _scene_payload(group, bible, style_hint)
        key = hashlib.sha256(f"{backend}|{model}|{payload}".encode()).hexdigest()
        result = cache.get(key)
        if result is None:
            try:
                if backend == "anthropic":
                    result = _call_anthropic(payload, model, effort)
                elif backend == "ollama":
                    result = _call_ollama(payload, model, ollama_host)
                else:
                    raise ValueError(f"Unknown LLM backend {backend}")
            except Exception as e:
                errors.append(f"scene {group[0].scene_index + 1}: {e}")
                result = None
            if result:
                cache[key] = result
                if cache_path:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(cache, f)
        by_id = {r.get("id"): r.get("prompt", "") for r in (result or {}).get("shots", [])}
        for s in group:
            p = (by_id.get(s.id) or "").strip()
            if p:
                s.prompt, s.enhanced = p, True
                ok += 1
            else:
                failed += 1
        if progress:
            progress(k + 1, len(groups))
    return {"enhanced": ok, "kept_original": failed, "errors": errors[:20]}
