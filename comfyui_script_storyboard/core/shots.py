"""Scene -> shot list -> image prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from .characters import CharacterBible, _name_pattern, stable_seed
from .parser import Screenplay, Scene

STYLE_PRESETS = {
    "cinematic film still": ("cinematic film still, 35mm, anamorphic lens, shallow depth of field, film grain",
                             "color graded, highly detailed"),
    "storyboard sketch": ("storyboard frame, rough pencil sketch, black and white, loose linework, "
                          "clear staging", "grayscale, professional storyboard artist"),
    "comic / graphic novel": ("graphic novel panel, bold ink lines, flat colors", "dynamic composition"),
    "anime": ("anime key visual, cel shaded", "detailed background, studio quality"),
    "3d animated feature": ("3d animated feature film still, stylized characters, soft global illumination",
                            "pixar-like render quality"),
    "photoreal": ("photorealistic, natural lighting, 50mm lens", "8k, sharp focus"),
    "none": ("", ""),
}
DEFAULT_NEGATIVE = ("text, watermark, logo, signature, caption, subtitles, blurry, lowres, deformed, "
                    "extra limbs, extra fingers, bad anatomy, disfigured, duplicate person, split screen, collage")

TOD_LIGHTING = {
    "NIGHT": "night, low-key lighting, practical lights", "DAY": "daytime, natural light",
    "MORNING": "morning light, soft warm sun", "EVENING": "evening, warm fading light",
    "AFTERNOON": "afternoon light", "DAWN": "dawn, cool blue pre-sunrise light",
    "DUSK": "dusk, golden hour", "SUNSET": "sunset, golden hour backlight",
    "SUNRISE": "sunrise, soft golden light", "MAGIC HOUR": "golden hour",
    "MIDNIGHT": "midnight, deep shadows", "NOON": "harsh midday sun",
}
STAGE_DIRECTION_RE = re.compile(r"\((?:beat|pause|cont'd|continuing|then|re:[^)]*)\)", re.I)
WORDS_PER_SECOND = 2.6


@dataclass
class Shot:
    index: int                  # global, sequential - this IS the storyboard order
    id: str                     # e.g. SC0012_SH003
    scene_index: int
    shot_in_scene: int
    kind: str                   # establishing | action | dialogue
    shot_size: str
    heading: str
    location: str
    time_of_day: str
    page: int
    source_text: str
    characters: List[str] = field(default_factory=list)
    primary_character: Optional[str] = None
    prompt: str = ""
    negative: str = ""
    seed: int = 0
    duration: float = 3.0
    dialogue: Optional[dict] = None   # {character, text, parenthetical, extension, voice}
    references: List[str] = field(default_factory=list)  # reference image ids used for this shot
    enhanced: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class PlannerSettings:
    style: str = "cinematic film still"
    style_prefix: str = ""
    style_suffix: str = ""
    negative: str = DEFAULT_NEGATIVE
    establishing_shots: bool = True
    max_shots_per_scene: int = 12
    max_action_words: int = 45
    dialogue_shots: bool = True
    include_names: bool = False
    min_duration: float = 2.0
    max_duration: float = 8.0
    project_seed: int = 0


def _clean(text: str) -> str:
    text = STAGE_DIRECTION_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _truncate_words(text: str, n: int) -> str:
    w = text.split()
    if len(w) <= n:
        return text
    cut = " ".join(w[:n])
    # prefer ending on a sentence boundary
    p = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[:p + 1] if p > len(cut) * 0.5 else cut


def _split_action(text: str, max_words: int) -> List[str]:
    sents = re.split(r"(?<=[.!?])\s+", text)
    beats, cur = [], []
    for s in sents:
        if cur and len(" ".join(cur + [s]).split()) > max_words:
            beats.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        beats.append(" ".join(cur))
    return [b for b in beats if b.strip()]


def _merge_to_limit(beats: List[dict], limit: int) -> List[dict]:
    """Merge adjacent action beats until the shot count fits. Dialogue beats are
    kept (each is a lip-sync unit)."""
    if limit <= 0:
        return beats
    while len(beats) > limit:
        best = None
        for i in range(len(beats) - 1):
            a, b = beats[i], beats[i + 1]
            if a["kind"] == "action" and b["kind"] == "action":
                size = len(a["text"].split()) + len(b["text"].split())
                if best is None or size < best[0]:
                    best = (size, i)
        if best is None:
            break
        i = best[1]
        beats[i] = {"kind": "action", "text": beats[i]["text"] + " " + beats[i + 1]["text"],
                    "page": beats[i]["page"]}
        del beats[i + 1]
    return beats


class _CharMatcher:
    def __init__(self, bible: CharacterBible):
        self.bible = bible
        self.patterns = {n: _name_pattern(c) for n, c in bible.characters.items()}

    def find(self, text: str) -> List[str]:
        hits = []
        for n, p in self.patterns.items():
            m = p.search(text)
            if m:
                hits.append((m.start(), n))
        return [n for _, n in sorted(hits)]


def plan_scene(scene: Scene, bible: CharacterBible, s: PlannerSettings,
               matcher: _CharMatcher, start_index: int) -> List[Shot]:
    beats: List[dict] = []
    for e in scene.elements:
        if e.type == "action":
            for b in _split_action(_clean(e.text), s.max_action_words):
                beats.append({"kind": "action", "text": b, "page": e.page})
        elif e.type == "dialogue" and s.dialogue_shots:
            beats.append({"kind": "dialogue", "text": e.text, "page": e.page, "el": e})
        elif e.type == "shot" and beats and beats[-1]["kind"] == "action":
            beats[-1]["size_hint"] = e.text
        elif e.type == "shot":
            beats.append({"kind": "action", "text": "", "page": e.page, "size_hint": e.text})
    beats = [b for b in beats if b["text"] or b.get("size_hint")]
    budget = s.max_shots_per_scene - (1 if s.establishing_shots else 0)
    beats = _merge_to_limit(beats, max(1, budget))

    scene_chars: List[str] = []
    for b in beats:
        names = ([b["el"].character] if b["kind"] == "dialogue" else []) + matcher.find(b["text"])
        resolved = []
        for n in names:
            r = bible.resolve(n) if n else None
            if r and r not in resolved:
                resolved.append(r)
        b["chars"] = resolved
        for r in resolved:
            if r not in scene_chars:
                scene_chars.append(r)

    shots: List[Shot] = []

    def add(kind, size, text, chars, page, dialogue=None):
        k = len(shots)
        sid = f"SC{scene.index + 1:04d}_SH{k + 1:03d}"
        sh = Shot(index=start_index + k, id=sid, scene_index=scene.index, shot_in_scene=k,
                  kind=kind, shot_size=size, heading=scene.heading, location=scene.location,
                  time_of_day=scene.time_of_day, page=page, source_text=text,
                  characters=chars, primary_character=chars[0] if chars else None,
                  dialogue=dialogue,
                  seed=stable_seed(s.project_seed, "shot", scene.index, k))
        shots.append(sh)

    if s.establishing_shots:
        add("establishing", "extreme wide establishing shot", scene.heading, [], scene.page)

    n_speakers = len(scene.speakers)
    last_speaker = None
    for b in beats:
        if b["kind"] == "dialogue":
            e = b["el"]
            speaker = bible.resolve(e.character) or e.character
            others = [c for c in scene_chars if c != speaker]
            if e.extension and re.search(r"V\.?O|O\.?S|O\.?C|PHONE|FILTER|RADIO", e.extension, re.I):
                size = "medium shot" if others else "wide shot"
                chars = others[:2] if others else []
            elif n_speakers >= 2 and others and speaker == last_speaker:
                size, chars = "close-up", [speaker]
            elif n_speakers >= 2 and others:
                size, chars = "over-the-shoulder medium close-up", [speaker, others[0]]
            else:
                size, chars = "medium close-up", [speaker]
            last_speaker = speaker
            c = bible.characters.get(speaker)
            add("dialogue", size, e.text, chars, b["page"], dialogue={
                "character": speaker, "text": _clean(e.text), "parenthetical": e.parenthetical,
                "extension": e.extension, "voice": c.voice if c else "",
                "on_screen": speaker in chars})
        else:
            hint = (b.get("size_hint") or "").lower()
            if "close" in hint:
                size = "close-up"
            elif "insert" in hint:
                size = "insert shot, extreme close-up"
            elif "pov" in hint:
                size = "point-of-view shot"
            elif "aerial" in hint:
                size = "aerial shot"
            elif len(b["chars"]) >= 3:
                size = "wide shot"
            elif len(b["chars"]) == 2:
                size = "medium two-shot"
            elif len(b["chars"]) == 1:
                size = "medium shot"
            else:
                size = "wide shot"
            add("action", size, b["text"] or b.get("size_hint", ""), b["chars"], b["page"])
    return shots


def build_prompt(shot: Shot, bible: CharacterBible, s: PlannerSettings) -> str:
    pre, suf = STYLE_PRESETS.get(s.style, ("", ""))
    pre = ", ".join(x for x in (s.style_prefix, pre) if x)
    suf = ", ".join(x for x in (suf, s.style_suffix) if x)
    setting = ""
    if shot.location:
        ie = {"INT": "interior", "EXT": "exterior"}.get(shot.heading[:3], "")
        setting = f"{ie} {shot.location.lower()}".strip()
    light = ""
    for k, v in TOD_LIGHTING.items():
        if k in (shot.time_of_day or ""):
            light = v
            break

    subjects = []
    for name in shot.characters:
        frag = bible.prompt_fragment(name, shot.scene_index, include_name=s.include_names)
        if frag:
            subjects.append(f"({frag})")
    if len(shot.characters) > 1:
        subjects.insert(0, f"{len(shot.characters)} people")

    if shot.kind == "establishing":
        action = "no people in focus, sets the location"
    elif shot.kind == "dialogue":
        d = shot.dialogue or {}
        mood = d.get("parenthetical") or ""
        if d.get("on_screen", True):
            action = "speaking, mouth slightly open, facing camera three-quarter view"
            if mood:
                action += f", {mood}"
        else:
            action = "listening, reacting"
    else:
        action = _truncate_words(re.sub(r"\s*\([^)]*\)", "", shot.source_text), s.max_action_words)
        for name in shot.characters:
            c = bible.characters.get(name)
            if not c:
                continue
            if c.description and c.description in action:  # already in the subject block
                action = action.replace(c.description, "")
            noun = {"male": "the man", "female": "the woman"}.get(c.gender, "the person")
            if c.variant_modifier in ("child", "baby"):
                noun = {"male": "the boy", "female": "the girl"}.get(c.gender, "the child")
            for tok in sorted([name] + list(c.aliases), key=len, reverse=True):
                action = re.sub(r"(?<![A-Za-z])" + re.escape(tok) + r"(?![A-Za-z])",
                                tok.title() if s.include_names else noun, action, flags=re.I)
        action = re.sub(r"\s*,\s*(,\s*)+", ", ", action)
        action = re.sub(r"\b(the (?:man|woman|person|boy|girl|child))\s*,\s*\.", r"\1.", action).strip(" ,")
        action = re.sub(r"\b(the (?:man|woman|person|boy|girl|child)),\s+(?=[a-z])", r"\1 ", action)
        action = re.sub(r"(^|[.!?]\s+)(the )", lambda m: m.group(1) + "The ", action)

    parts = [pre, shot.shot_size, ", ".join(subjects), action, setting, light, suf]
    return ", ".join(p.strip(", ") for p in parts if p and p.strip(", "))


def estimate_duration(shot: Shot, s: PlannerSettings) -> float:
    if shot.kind == "dialogue":
        words = len((shot.dialogue or {}).get("text", "").split())
        d = words / WORDS_PER_SECOND + 0.6
        return round(max(1.5, d), 2)  # dialogue shots are never clipped
    if shot.kind == "establishing":
        return max(s.min_duration, min(3.0, s.max_duration))
    words = len(shot.source_text.split())
    return round(max(s.min_duration, min(s.max_duration, words / 4.0 + 1.5)), 2)


def plan_shots(sp: Screenplay, bible: CharacterBible, s: PlannerSettings,
               scene_start: int = 0, scene_end: int = -1) -> List[Shot]:
    matcher = _CharMatcher(bible)
    scenes = sp.scenes[scene_start: None if scene_end < 0 else scene_end + 1]
    shots: List[Shot] = []
    for sc in scenes:
        for sh in plan_scene(sc, bible, s, matcher, len(shots)):
            sh.prompt = build_prompt(sh, bible, s)
            sh.negative = s.negative
            sh.duration = estimate_duration(sh, s)
            shots.append(sh)
    return shots


def shots_preview(shots: List[Shot], limit: int = 60) -> str:
    out = []
    for sh in shots[:limit]:
        dl = ""
        if sh.dialogue:
            dl = f'\n      "{sh.dialogue["character"]}: {sh.dialogue["text"][:80]}"'
        out.append(f"#{sh.index:05d} {sh.id} [{sh.kind}/{sh.shot_size}] {sh.duration:.1f}s\n      {sh.prompt}{dl}")
    if len(shots) > limit:
        out.append(f"... {len(shots) - limit} more shots")
    return "\n".join(out)
