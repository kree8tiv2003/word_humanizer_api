"""Screenplay parser (Fountain-compatible, tolerant of PDF/plain-text scripts).

Produces a list of scenes; each scene holds ordered elements:
action, dialogue, transition, note, shot.

Scripts with no scene headings (treatments, prose, AV scripts) fall back to
chunking paragraphs into pseudo-scenes so the rest of the pipeline still works.
Parsing is a single linear pass, so 1000-page scripts parse in well under a
second.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional

SCENE_RE = re.compile(
    r"^(?:\d+[A-Z]?\s+)?(?P<prefix>INT\.?/EXT|EXT\.?/INT|INT|EXT|EST|I/E|I\.E)[.\s]\s*(?P<rest>.*?)(?:\s+#?\d+[A-Z]?#?)?$",
    re.I)
FORCED_SCENE_RE = re.compile(r"^\.(?!\.)(?P<rest>\S.*)$")
SCENE_NUMBER_RE = re.compile(r"\s*#([\w.\-]+)#\s*$")
TRANSITION_RE = re.compile(
    r"^(?:[A-Z .'\-]+ TO:|FADE (?:IN|OUT)[.:]?|FADE TO BLACK\.?|CUT TO BLACK\.?|SMASH CUT\.?|THE END\.?)$")
CUE_RE = re.compile(
    r"^(?P<name>(?=.*[A-Z])[A-Z0-9][A-Z0-9 .'\-#&]*?)\s*(?P<ext>(?:\([^)]*\)\s*)*)\^?$")
SHOT_RE = re.compile(
    r"^(?:ANGLE ON|CLOSE ON|CLOSE UP|CLOSE-UP|WIDE ON|WIDE SHOT|POV|INSERT|BACK TO SCENE|"
    r"INTERCUT|MONTAGE|SERIES OF SHOTS|ESTABLISHING|AERIAL|EXTREME CLOSE)\b.*$")
TIME_WORDS = ("DAY", "NIGHT", "MORNING", "EVENING", "AFTERNOON", "DAWN", "DUSK",
              "SUNSET", "SUNRISE", "LATER", "CONTINUOUS", "MOMENTS LATER",
              "SAME TIME", "NOON", "MIDNIGHT", "MAGIC HOUR")
NOT_CUES = {"THE END", "CONTINUED", "MORE", "FADE IN", "FADE OUT", "END CREDITS",
            "TITLE", "SUPER", "BLACK", "LATER", "MONTAGE", "END MONTAGE",
            "INTERCUT", "BACK TO SCENE", "FLASHBACK", "END FLASHBACK", "OMITTED"}


@dataclass
class Element:
    type: str                     # action | dialogue | transition | note | shot
    text: str
    character: Optional[str] = None
    parenthetical: Optional[str] = None
    extension: Optional[str] = None   # V.O., O.S., CONT'D ...
    page: int = 1
    dual: bool = False


@dataclass
class Scene:
    index: int
    heading: str
    int_ext: str = ""
    location: str = ""
    time_of_day: str = ""
    number: Optional[str] = None
    page: int = 1
    elements: List[Element] = field(default_factory=list)

    @property
    def speakers(self):
        seen = []
        for e in self.elements:
            if e.type == "dialogue" and e.character not in seen:
                seen.append(e.character)
        return seen


@dataclass
class Screenplay:
    title: str
    scenes: List[Scene]
    pages: int
    title_page: dict = field(default_factory=dict)
    fallback_chunked: bool = False

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        scenes = [Scene(**{**s, "elements": [Element(**e) for e in s["elements"]]})
                  for s in d["scenes"]]
        return Screenplay(title=d["title"], scenes=scenes, pages=d["pages"],
                          title_page=d.get("title_page", {}),
                          fallback_chunked=d.get("fallback_chunked", False))


def normalize_character(name: str) -> str:
    name = re.sub(r"\([^)]*\)", "", name)
    name = name.replace("^", "").replace("@", "")
    name = re.sub(r"\s+", " ", name).strip(" .:-")
    return name.upper()


def parse_heading(heading: str):
    """INT. DINER - NIGHT -> ("INT", "DINER", "NIGHT")."""
    h = SCENE_NUMBER_RE.sub("", heading).strip()
    m = SCENE_RE.match(h)
    if m:
        ie = m.group("prefix").upper().replace(".", "")
        rest = m.group("rest")
    else:
        ie, rest = "", h.lstrip(".")
    parts = [p.strip() for p in re.split(r"\s+[-–—]+\s+|\s*--\s*", rest) if p.strip()]
    tod = ""
    if len(parts) > 1 and any(w in parts[-1].upper() for w in TIME_WORDS):
        tod = parts.pop().upper()
    location = " - ".join(parts).upper()
    return ie, location, tod


def _strip_boneyard(text: str) -> str:
    return re.sub(r"/\*.*?\*/", "", text, flags=re.S)


def _parse_title_page(lines):
    """Fountain title page: `Key: value` block at the very top."""
    tp, i = {}, 0
    if not lines or not re.match(r"^[A-Za-z ]+:", lines[0]):
        return tp, 0
    key = None
    while i < len(lines) and lines[i].strip():
        ln = lines[i]
        m = re.match(r"^([A-Za-z ]+):\s*(.*)$", ln)
        if m:
            key = m.group(1).strip().lower()
            tp[key] = m.group(2).strip()
        elif key:
            tp[key] = (tp[key] + " " + ln.strip()).strip()
        i += 1
    return tp, i


def parse_screenplay(text: str, title: str = "Untitled",
                     fallback_paragraphs_per_scene: int = 8) -> Screenplay:
    text = _strip_boneyard(text)
    raw_lines = text.split("\n")
    title_page, start = _parse_title_page(raw_lines)
    if title_page.get("title"):
        title = title_page["title"]

    scenes: List[Scene] = []
    current: Optional[Scene] = None
    page = 1
    lines = raw_lines[start:]
    n = len(lines)

    def ensure_scene():
        nonlocal current
        if current is None:
            current = Scene(index=len(scenes), heading="OPENING", page=page)
            scenes.append(current)
        return current

    action_buf: List[str] = []

    def flush_action():
        if action_buf:
            txt = " ".join(s.strip() for s in action_buf).strip()
            if txt:
                ensure_scene().elements.append(Element("action", txt, page=page))
            action_buf.clear()

    i = 0
    while i < n:
        line = lines[i]
        if "\f" in line:
            page += line.count("\f")
            line = line.replace("\f", "")
        s = line.strip()
        prev_blank = i == 0 or not lines[i - 1].replace("\f", "").strip()
        next_line = lines[i + 1].replace("\f", "").strip() if i + 1 < n else ""

        if not s:
            flush_action()
            i += 1
            continue

        # Notes [[...]] (may span lines)
        if s.startswith("[[") :
            note = s
            while "]]" not in note and i + 1 < n:
                i += 1
                note += " " + lines[i].strip()
            flush_action()
            ensure_scene().elements.append(
                Element("note", note.strip()[2:].split("]]")[0].strip(), page=page))
            i += 1
            continue

        # Scene heading
        forced = FORCED_SCENE_RE.match(s)
        if (SCENE_RE.match(s) and prev_blank) or forced:
            flush_action()
            heading = forced.group("rest") if forced else s
            num = SCENE_NUMBER_RE.search(heading)
            ie, loc, tod = parse_heading(heading)
            current = Scene(index=len(scenes), heading=SCENE_NUMBER_RE.sub("", heading).strip().upper(),
                            int_ext=ie, location=loc, time_of_day=tod,
                            number=num.group(1) if num else None, page=page)
            scenes.append(current)
            i += 1
            continue

        # Transition
        if s.startswith(">") and not s.endswith("<"):
            flush_action()
            ensure_scene().elements.append(Element("transition", s[1:].strip(), page=page))
            i += 1
            continue
        if TRANSITION_RE.match(s) and prev_blank:
            flush_action()
            ensure_scene().elements.append(Element("transition", s, page=page))
            i += 1
            continue

        # Character cue + dialogue block
        forced_cue = s.startswith("@")
        cue_text = s[1:] if forced_cue else s
        m = CUE_RE.match(cue_text)
        is_cue = (
            (forced_cue or (m and cue_text == cue_text.upper()))
            and prev_blank and next_line
            and not SCENE_RE.match(cue_text)
            and not SHOT_RE.match(cue_text)
            and not TRANSITION_RE.match(cue_text)
            and normalize_character(cue_text) not in NOT_CUES
            and len(cue_text) <= 50
            and len(normalize_character(cue_text)) >= 1
        )
        if is_cue:
            flush_action()
            ext = None
            ext_m = re.findall(r"\(([^)]*)\)", cue_text)
            if ext_m:
                ext = ", ".join(x.strip() for x in ext_m)
            name = normalize_character(cue_text)
            dual = cue_text.rstrip().endswith("^")
            i += 1
            paren, dlg = [], []
            while i < n:
                dl = lines[i]
                if "\f" in dl:
                    page += dl.count("\f")
                    dl = dl.replace("\f", "")
                ds = dl.strip()
                if not ds:
                    break
                if ds.startswith("(") and ds.endswith(")") and not dlg:
                    paren.append(ds[1:-1])
                elif ds.startswith("(") and ds.endswith(")"):
                    # mid-speech parenthetical: keep as a stage beat
                    dlg.append(f"({ds[1:-1]})")
                else:
                    dlg.append(ds)
                i += 1
            speech = " ".join(dlg).strip()
            if speech:
                ensure_scene().elements.append(Element(
                    "dialogue", speech, character=name,
                    parenthetical="; ".join(paren) or None,
                    extension=ext, page=page, dual=dual))
            continue

        if SHOT_RE.match(s) and s == s.upper() and prev_blank:
            flush_action()
            ensure_scene().elements.append(Element("shot", s, page=page))
            i += 1
            continue

        # Centered text >THE END< and everything else is action
        if s.startswith(">") and s.endswith("<"):
            s = s[1:-1].strip()
        if s.startswith("!"):
            s = s[1:]
        action_buf.append(s)
        i += 1
    flush_action()

    # Drop empty "OPENING" pseudo-scene (e.g. just FADE IN:)
    scenes = [sc for sc in scenes
              if not (sc.heading == "OPENING"
                      and all(e.type in ("transition", "note") for e in sc.elements))]
    real_headings = sum(1 for sc in scenes if sc.heading != "OPENING")
    fallback = False
    if real_headings == 0:
        scenes = _chunk_fallback(scenes, fallback_paragraphs_per_scene)
        fallback = True
    last_tod = ""
    for idx, sc in enumerate(scenes):
        sc.index = idx
        if sc.time_of_day and any(w in sc.time_of_day for w in ("LATER", "CONTINUOUS", "SAME")):
            sc.time_of_day = f"{last_tod} ({sc.time_of_day})" if last_tod else sc.time_of_day
        base = sc.time_of_day.split(" (")[0]
        if base and not any(w in base for w in ("LATER", "CONTINUOUS", "SAME")):
            last_tod = base
    total_pages = max(page, 1)
    if "\f" not in text:
        total_pages = max(1, -(-text.count("\n") // 55))
    return Screenplay(title=title, scenes=scenes, pages=total_pages,
                      title_page=title_page, fallback_chunked=fallback)


def _chunk_fallback(scenes, per_scene):
    elements = [e for sc in scenes for e in sc.elements]
    out, chunk = [], []
    for e in elements:
        chunk.append(e)
        if len(chunk) >= per_scene:
            out.append(chunk)
            chunk = []
    if chunk:
        out.append(chunk)
    result = []
    for k, ch in enumerate(out):
        result.append(Scene(index=k, heading=f"SEQUENCE {k + 1}", location="",
                            page=ch[0].page, elements=ch))
    return result


def screenplay_stats(sp: Screenplay) -> str:
    n_dlg = sum(1 for sc in sp.scenes for e in sc.elements if e.type == "dialogue")
    n_act = sum(1 for sc in sp.scenes for e in sc.elements if e.type == "action")
    speakers = sorted({e.character for sc in sp.scenes for e in sc.elements if e.character})
    lines = [f"Title: {sp.title}", f"Pages: ~{sp.pages}", f"Scenes: {len(sp.scenes)}",
             f"Action blocks: {n_act}", f"Dialogue blocks: {n_dlg}",
             f"Speaking characters: {len(speakers)}"]
    if sp.fallback_chunked:
        lines.append("NOTE: no scene headings found - text was chunked into sequences.")
    return "\n".join(lines)
