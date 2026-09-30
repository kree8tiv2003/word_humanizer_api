"""Character bible + continuity tracking.

Consistency model ("consistent unless otherwise stated"):

* Every character gets ONE canonical description, a fixed seed, an optional
  voice and optional reference image. These are injected into every shot the
  character appears in.
* The script can change a character's look from a point onward:
    - natural language in action lines:  "John, now wearing a tuxedo, ..."
      / "Mara changes into scrubs." / "JOHN, now 70, ..."
    - explicit Fountain notes (most reliable):
        [[LOOK JOHN: torn tuxedo, bloodied face]]      persists until changed
        [[SCENE LOOK JOHN: soaking wet]]              this scene only
        [[LOOK JOHN: reset]]                          back to canonical look
        [[AGE JOHN: 70]]                              persists
        [[DESC JOHN: full replacement description]]   persists
* Age variants in cues (YOUNG JOHN, OLDER MARA) are linked to the base
  character but rendered with an age modifier.
* User overrides (JSON) always win over anything extracted from the script.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from .parser import Screenplay, normalize_character

VARIANT_PREFIXES = {
    "YOUNG": "younger", "YOUNGER": "younger", "LITTLE": "child", "TEEN": "teenage",
    "TEENAGE": "teenage", "KID": "child", "CHILD": "child", "OLD": "elderly",
    "OLDER": "older", "ELDERLY": "elderly", "ADULT": "adult", "BABY": "baby",
}
MALE_WORDS = r"\b(he|him|his|himself|man|boy|guy|father|dad|husband|brother|son|mr|sir|king|gentleman|beard|bearded|mustache|moustache|stubble)\b"
FEMALE_WORDS = r"\b(she|her|hers|herself|woman|girl|lady|mother|mom|wife|sister|daughter|mrs|ms|miss|queen)\b"
WARDROBE_RE = (r"(?:now wearing|now dressed in|is now in|changes into|changed into|"
               r"has changed into|slips into|is wearing|now in a|dressed in|wearing)\s+([^.,;!?]+)")
AGE_RE = re.compile(r"^\(?\s*((?:early |mid |late |mid-|early-|late-)?\d{1,2}s?)\s*\)?$", re.I)
DEFAULT_VOICES = {
    "male": ["en-US-GuyNeural", "en-US-DavisNeural", "en-US-TonyNeural", "en-GB-RyanNeural",
             "en-US-JasonNeural", "en-AU-WilliamNeural"],
    "female": ["en-US-JennyNeural", "en-US-AriaNeural", "en-US-SaraNeural", "en-GB-SoniaNeural",
               "en-US-NancyNeural", "en-AU-NatashaNeural"],
    "unknown": ["en-US-ChristopherNeural", "en-US-MichelleNeural"],
}


def stable_seed(*parts, bits: int = 32) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:16], 16) % (2 ** bits)


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "character"


@dataclass
class Character:
    name: str
    id: str
    description: str = ""
    age: str = ""
    gender: str = "unknown"
    wardrobe: str = ""
    seed: int = 0
    voice: str = ""
    reference_image: str = ""
    aliases: List[str] = field(default_factory=list)
    variant_of: Optional[str] = None
    variant_modifier: str = ""
    dialogue_lines: int = 0
    scenes: List[int] = field(default_factory=list)
    consistent: bool = True
    user_locked: List[str] = field(default_factory=list)  # fields set by user overrides


@dataclass
class CharacterBible:
    characters: Dict[str, Character]
    project_seed: int = 0
    consistent: bool = True
    # continuity[scene_index][char_name] = {"description", "wardrobe", "age"}
    continuity: List[Dict[str, dict]] = field(default_factory=list)

    def to_dict(self):
        return {"project_seed": self.project_seed, "consistent": self.consistent,
                "characters": {k: asdict(v) for k, v in self.characters.items()},
                "continuity": self.continuity}

    @staticmethod
    def from_dict(d):
        return CharacterBible(
            characters={k: Character(**v) for k, v in d["characters"].items()},
            project_seed=d.get("project_seed", 0), consistent=d.get("consistent", True),
            continuity=d.get("continuity", []))

    def resolve(self, token: str) -> Optional[str]:
        t = normalize_character(token)
        if t in self.characters:
            return t
        for name, c in self.characters.items():
            if t in c.aliases:
                return name
        return None

    def look(self, name: str, scene_index: int) -> dict:
        """Effective look of a character in a given scene."""
        c = self.characters[name]
        base = {"description": c.description, "wardrobe": c.wardrobe, "age": c.age, "look": ""}
        if c.variant_of and c.variant_of in self.characters and not c.description:
            parent = self.characters[c.variant_of]
            base["description"] = parent.description
            base["age"] = c.age or c.variant_modifier
            if not c.gender or c.gender == "unknown":
                c.gender = parent.gender
        if 0 <= scene_index < len(self.continuity):
            st = self.continuity[scene_index].get(name)
            if st:
                base.update({k: v for k, v in st.items() if v is not None})
        return base

    def prompt_fragment(self, name: str, scene_index: int, include_name: bool = False) -> str:
        c = self.characters.get(name)
        if c is None:
            return ""
        lk = self.look(name, scene_index)
        parts = []
        if include_name:
            parts.append(c.name.title())
        if lk.get("age"):
            parts.append(f"{lk['age']} years old" if lk["age"][0].isdigit() and not lk["age"].endswith("s")
                         else (f"in their {lk['age']}" if lk["age"][0].isdigit() else lk["age"]))
        if c.gender in ("male", "female") and not re.search(
                r"\b(man|woman|boy|girl|male|female|lady|guy)\b", lk.get("description", ""), re.I):
            parts.append(c.gender)
        if self.consistent and c.consistent and lk.get("description"):
            parts.append(lk["description"])
        if lk.get("wardrobe") and lk["wardrobe"].lower() not in lk.get("description", "").lower():
            parts.append(f"wearing {lk['wardrobe']}")
        if lk.get("look"):
            parts.append(lk["look"])
        return ", ".join(p for p in parts if p)

    def summary(self) -> str:
        rows = []
        for c in sorted(self.characters.values(), key=lambda c: -c.dialogue_lines):
            ref = " [ref image]" if c.reference_image else ""
            var = f" (variant of {c.variant_of})" if c.variant_of else ""
            rows.append(f"{c.name}{var}: {c.dialogue_lines} lines, {len(c.scenes)} scenes, "
                        f"seed {c.seed}, voice {c.voice or '-'}{ref}\n    {c.description or '(no description found - add one in overrides)'}")
        return "\n".join(rows)


def _trim_description(desc: str) -> str:
    """Drop trailing action clauses: 'a heavyset cook, wipes a mug' -> 'a heavyset cook'."""
    parts = [p.strip() for p in desc.split(",")]
    keep = [parts[0]] if parts else []
    for p in parts[1:]:
        first = p.split(" ", 1)[0].lower()
        if re.fullmatch(r"[a-z]+(s|es)", first) and first not in _NOT_VERBS:
            break
        keep.append(p)
    out = ", ".join(x for x in keep if x)
    return re.split(r"\s+(?:who|as|while|and then)\s+[a-z]+s\b", out)[0].strip(" ,")


_NOT_VERBS = {"glasses", "boots", "jeans", "pants", "overalls", "clothes", "dress", "less", "dreadlocks",
              "braids", "tattoos", "freckles", "eyes", "curls", "whiskers", "sneakers", "heels",
              "scrubs", "fatigues", "sweats", "shorts", "gloves", "earrings", "looks", "has"}


def _find_intro(name: str, texts: List[str]):
    """Find a screenplay-style introduction: 'JOHN SMITH (40s), a weathered cop.'
    A short cue name (MARA) also matches a longer caps intro (MARA OKAFOR)."""
    pat = re.compile(r"(?<![A-Za-z])" + re.escape(name) + r"(?: [A-Z][A-Z'\-]+){0,2}(?![A-Za-z])"
                     r"\s*(\([^)]{1,30}\))?\s*,?\s*([^.!?]{3,220})")
    for t in texts:
        for m in pat.finditer(t):
            # intro convention: name written in CAPS in action text
            if t[m.start():m.start() + len(name)] != name:
                continue
            paren, desc = m.group(1), m.group(2).strip()
            age = ""
            if paren and AGE_RE.match(paren):
                age = AGE_RE.match(paren).group(1)
            desc_parts = [p.strip() for p in desc.split(",")]
            if not age and desc_parts and AGE_RE.match(desc_parts[0]):
                age = AGE_RE.match(desc_parts[0]).group(1)
                desc_parts = desc_parts[1:]
            desc = ", ".join(p for p in desc_parts if p)
            # Only accept if it reads like a description, not an action verb.
            looks_descriptive = bool(age) or bool(paren) or re.match(
                r"^(a|an|the|his|her|their|late|early|mid|\d|tall|short|young|old|elderly|"
                r"thin|heavy|slim|stocky|beautiful|handsome|scruffy|wiry|lanky|petite|burly|"
                r"weathered|grizzled|bald|blonde|red-haired|dark-haired|gray-haired|grey-haired)\b",
                desc, re.I) or (m.group(0).find(",") != -1 and m.group(0).find(",") <= len(name) + (len(paren) if paren else 0) + 2)
            if looks_descriptive:
                words = _trim_description(desc).split()
                return age, " ".join(words[:40]).rstrip(",; "), t[m.end():m.end() + 400]
            return age, "", t[m.end():m.end() + 400]
    return "", "", ""


def _mention_context(c: "Character", texts: List[str], limit: int = 40) -> str:
    pat = _name_pattern(c)
    ctx = []
    for t in texts:
        for m in pat.finditer(t):
            rest = t[m.end():m.end() + 160]
            ctx.append(re.split(r"[.!?]", rest, maxsplit=1)[0])  # same sentence only
            if len(ctx) >= limit:
                return " ".join(ctx)
    return " ".join(ctx)


def _guess_gender(desc: str, context: str) -> str:
    text = f"{desc} {context}".lower()
    m = len(re.findall(MALE_WORDS, text))
    f = len(re.findall(FEMALE_WORDS, text))
    if m > f:
        return "male"
    if f > m:
        return "female"
    return "unknown"


def build_bible(sp: Screenplay, project_seed: int = 0, consistent: bool = True,
                overrides: Optional[dict] = None, min_lines: int = 1,
                include_nonspeaking: bool = True) -> CharacterBible:
    chars: Dict[str, Character] = {}
    action_texts = [e.text for sc in sp.scenes for e in sc.elements if e.type == "action"]

    for sc in sp.scenes:
        for e in sc.elements:
            if e.type != "dialogue" or not e.character:
                continue
            c = chars.get(e.character)
            if c is None:
                c = chars[e.character] = Character(name=e.character, id=slug(e.character))
            c.dialogue_lines += 1
            if sc.index not in c.scenes:
                c.scenes.append(sc.index)

    chars = {k: v for k, v in chars.items() if v.dialogue_lines >= min_lines}

    if include_nonspeaking:
        # Non-speaking characters introduced with an age: "a GUARD (50s)" / "MRS. OKAFOR, 60,"
        intro_re = re.compile(r"(?<![A-Za-z])([A-Z][A-Z.'\-]{1,}(?: [A-Z][A-Z.'\-]+){0,3})\s*(?:\(|,\s*)"
                              r"((?:early |mid |late |mid-)?\d{1,2}s?)\b")
        for t in action_texts:
            for m in intro_re.finditer(t):
                nm = normalize_character(m.group(1))
                if nm and nm not in chars and len(nm) > 2 and nm not in ("INT", "EXT"):
                    chars[nm] = Character(name=nm, id=slug(nm))

    # Full-name intro of a speaking character (MARA OKAFOR introduces cue MARA)
    for nm in list(chars):
        c = chars[nm]
        if c.dialogue_lines:
            continue
        owner = [sp_ for sp_ in chars if sp_ != nm and chars[sp_].dialogue_lines
                 and (nm.startswith(sp_ + " ") or nm.endswith(" " + sp_))]
        if len(owner) == 1:
            chars[owner[0]].aliases.append(nm)
            del chars[nm]

    # Variants (YOUNG JOHN -> JOHN)
    for name, c in chars.items():
        parts = name.split(" ", 1)
        if len(parts) == 2 and parts[0] in VARIANT_PREFIXES and parts[1] in chars:
            c.variant_of = parts[1]
            c.variant_modifier = VARIANT_PREFIXES[parts[0]]

    # Aliases: first / last name when unambiguous
    token_owner: Dict[str, List[str]] = {}
    for name in chars:
        toks = [t for t in re.split(r"[ .]+", name) if len(t) > 2 and t not in VARIANT_PREFIXES
                and t not in ("MR", "MRS", "MS", "DR", "THE", "OFFICER", "DETECTIVE", "AGENT")]
        if len(toks) > 1:
            for t in toks:
                token_owner.setdefault(t, []).append(name)
    for tok, owners in token_owner.items():
        if len(owners) == 1 and tok not in chars:
            chars[owners[0]].aliases.append(tok)

    # Descriptions, age, gender, seed, scenes for non-speakers
    for name, c in chars.items():
        age, desc, ctx = _find_intro(name, action_texts)
        c.age = c.age or age
        c.description = c.description or desc
        ctx = re.split(r"[.!?]", ctx, maxsplit=1)[0]
        c.gender = _guess_gender(desc, ctx + " " + _mention_context(c, action_texts))
        c.seed = stable_seed(project_seed, name)
        if not c.scenes:
            pat = _name_pattern(c)
            c.scenes = [sc.index for sc in sp.scenes
                        if any(e.type == "action" and pat.search(e.text) for e in sc.elements)]

    # Voices: deterministic assignment per gender
    counters = {"male": 0, "female": 0, "unknown": 0}
    for name in sorted(chars, key=lambda n: -chars[n].dialogue_lines):
        c = chars[name]
        pool = DEFAULT_VOICES[c.gender]
        c.voice = pool[counters[c.gender] % len(pool)]
        counters[c.gender] += 1
    for c in chars.values():  # variants share their parent's voice by default
        if c.variant_of:
            c.voice = chars[c.variant_of].voice

    bible = CharacterBible(characters=chars, project_seed=project_seed, consistent=consistent)
    if overrides:
        apply_overrides(bible, overrides)
    bible.continuity = track_continuity(sp, bible)
    return bible


def parse_overrides(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Allow a simple "NAME: description" per line format too.
        data = {}
        for ln in text.splitlines():
            if ":" in ln:
                k, v = ln.split(":", 1)
                data[k.strip()] = {"description": v.strip()}
    if not isinstance(data, dict):
        raise ValueError("Character overrides must be a JSON object keyed by character name")
    return data


def apply_overrides(bible: CharacterBible, overrides: dict):
    allowed = {"description", "age", "gender", "wardrobe", "seed", "voice",
               "reference_image", "aliases", "consistent"}
    for raw_name, vals in overrides.items():
        if isinstance(vals, str):
            vals = {"description": vals}
        name = bible.resolve(raw_name) or normalize_character(raw_name)
        c = bible.characters.get(name)
        if c is None:
            c = bible.characters[name] = Character(name=name, id=slug(name),
                                                   seed=stable_seed(bible.project_seed, name))
        for k, v in vals.items():
            if k in allowed:
                if k == "aliases":
                    v = [normalize_character(a) for a in v]
                setattr(c, k, v)
                if k not in c.user_locked:
                    c.user_locked.append(k)


def _name_pattern(c: Character):
    names = [c.name] + list(c.aliases)
    alts = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    return re.compile(r"(?<![A-Za-z])(?:" + alts + r")(?![A-Za-z])", re.I)


NOTE_RE = re.compile(r"^(SCENE\s+LOOK|LOOK|AGE|DESC|WARDROBE)\s+(.+?)\s*:\s*(.*)$", re.I)


def track_continuity(sp: Screenplay, bible: CharacterBible) -> List[Dict[str, dict]]:
    persistent: Dict[str, dict] = {}
    out: List[Dict[str, dict]] = []
    patterns = {n: _name_pattern(c) for n, c in bible.characters.items()}
    for sc in sp.scenes:
        scene_only: Dict[str, dict] = {}
        for e in sc.elements:
            if e.type == "note":
                m = NOTE_RE.match(e.text)
                if not m:
                    continue
                kind, who, val = m.group(1).upper(), m.group(2), m.group(3).strip()
                name = bible.resolve(who)
                if not name:
                    continue
                c = bible.characters[name]
                if kind.startswith("SCENE"):
                    scene_only.setdefault(name, {})["look"] = val
                elif kind in ("LOOK", "WARDROBE"):
                    if val.lower() in ("reset", "default", "normal"):
                        persistent.pop(name, None)
                    elif kind == "WARDROBE":
                        persistent.setdefault(name, {})["wardrobe"] = val
                    else:
                        persistent.setdefault(name, {})["look"] = val
                elif kind == "AGE":
                    persistent.setdefault(name, {})["age"] = val
                elif kind == "DESC":
                    persistent.setdefault(name, {})["description"] = val
            elif e.type == "action":
                for name, pat in patterns.items():
                    c = bible.characters[name]
                    if "wardrobe" in c.user_locked and not c.consistent:
                        continue
                    for sent in re.split(r"(?<=[.!?])\s+", e.text):
                        if not pat.search(sent):
                            continue
                        # Only attribute if the name is the closest subject before the verb
                        wm = re.search(pat.pattern + r"[^.;!?]{0,40}?" + WARDROBE_RE, sent, re.I)
                        if wm:
                            val = wm.group(1).strip().rstrip(",")
                            if val and val.lower() not in (c.description or "").lower():
                                persistent.setdefault(name, {})["wardrobe"] = val
                        am = re.search(pat.pattern + r",?\s+now\s+(\d{1,3})\b", sent, re.I)
                        if am:
                            persistent.setdefault(name, {})["age"] = am.group(1)
        merged = {k: dict(v) for k, v in persistent.items()}
        for k, v in scene_only.items():
            merged.setdefault(k, {}).update(v)
        out.append(merged)
    return out


def characters_in_text(bible: CharacterBible, text: str) -> List[str]:
    found = []
    for name, c in bible.characters.items():
        if _name_pattern(c).search(text):
            found.append(name)
    return found
