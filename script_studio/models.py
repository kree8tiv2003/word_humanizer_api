"""Data models shared by ingestion, planning, writing and export.

Classes named *Out are the structured-output schemas the language model fills in.
Keep them to plain types (str, int, float, bool, lists of models) so they convert
cleanly to JSON schema.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- source material
class Unit(BaseModel):
    """One addressable piece of the source: a paragraph, a lyric line or a transcript segment."""
    idx: int
    text: str
    start: Optional[float] = None  # seconds, only for timed (audio) sources
    end: Optional[float] = None

    @property
    def words(self) -> int:
        return max(1, len(self.text.split()))


class Section(BaseModel):
    start: float
    end: float
    label: str            # "A", "B", "C"... sections with the same letter sound alike
    role: str             # intro / verse / chorus / bridge / outro (best guess)
    energy: float         # 0..1
    energy_level: Literal['low', 'medium', 'high']


class AudioAnalysis(BaseModel):
    duration: float
    tempo: float
    beats: list[float] = []
    downbeats: list[float] = []
    sections: list[Section] = []
    energy_curve: list[float] = []  # one value per second, 0..1


class SourceDoc(BaseModel):
    kind: str                          # text, pdf, docx, html, url, audio, lyrics
    title: str = ''
    units: list[Unit] = []
    reference: str = ''                # supporting text (e.g. lyrics sheet next to a transcript)
    audio: Optional[AudioAnalysis] = None
    notes: list[str] = []              # warnings shown to the user

    @property
    def text(self) -> str:
        return '\n\n'.join(u.text for u in self.units)

    @property
    def timed(self) -> bool:
        return bool(self.units) and all(u.start is not None for u in self.units)

    @property
    def word_count(self) -> int:
        return sum(u.words for u in self.units)


# --------------------------------------------------------------------------- user settings
Increment = Literal['5', '10', '15', '30', '60', 'full']


class Settings(BaseModel):
    mode: Literal['story', 'music'] = 'story'
    increment: Increment = '10'
    target_seconds: Optional[float] = None   # story mode: desired running time (text sources)
    visual_style: str = 'cinematic photoreal'
    tone: str = ''                           # optional override, e.g. "hopeful", "noir"
    aspect_ratio: str = '16:9'
    target_generator: str = 'any'            # veo, sora, kling, runway, luma, any
    include_dialogue: bool = True
    direction: str = ''                      # free-text notes from the user
    model: str = ''                          # overrides the default Claude model


# --------------------------------------------------------------------------- model outputs
class CharacterOut(BaseModel):
    name: str
    role: str
    visual_signature: str = Field(description='Exact visual description repeated verbatim in every prompt the character appears in: age, build, face, hair, wardrobe.')
    voice: str = Field(description='How the character speaks or sings.')


class LocationOut(BaseModel):
    name: str
    visual_signature: str = Field(description='Exact visual description of the place, light and time of day, reused in prompts.')


class BeatOut(BaseModel):
    first_unit: int = Field(description='Index of the first source unit this beat covers.')
    last_unit: int = Field(description='Index of the last source unit this beat covers.')
    summary: str
    emotion: str
    intensity: int = Field(description='1 = still and quiet, 5 = peak action or emotion.')
    weight: float = Field(description='Relative screen time this beat deserves; 1.0 is average.')
    characters: list[str]
    location: str


class StoryBibleOut(BaseModel):
    title: str
    logline: str
    genre: str
    tone: str
    visual_style: str = Field(description='Look of the whole piece: palette, lighting, lens character, texture, realism level.')
    pacing_notes: str = Field(description='How the source moves: where it is fast, slow, builds and releases.')
    world_rules: str = Field(description='Recurring motifs, powers, VFX language, props and anything that must stay consistent.')
    characters: list[CharacterOut]
    locations: list[LocationOut]
    beats: list[BeatOut]


class BeatsOut(BaseModel):
    beats: list[BeatOut]
    new_characters: list[CharacterOut]
    new_locations: list[LocationOut]


class DialogueLine(BaseModel):
    character: str
    line: str
    delivery: str = Field(description='Parenthetical: tone, volume or action while speaking. Empty if none.')


class ShotOut(BaseModel):
    start: float = Field(description='Absolute start time in seconds.')
    end: float = Field(description='Absolute end time in seconds.')
    camera: str = Field(description='Camera technique name, from the camera library.')
    prompt: str = Field(description='Complete, ready-to-paste AI video prompt for this shot.')
    action: str = Field(description='Screenplay action line describing what happens.')
    dialogue: list[DialogueLine]
    voiceover: str
    lyrics: str
    sound: str = Field(description='Sound effects, ambience and music cue.')
    vfx: str = Field(description='Visual effects in this shot, or empty.')
    negative: str = Field(description='Negative prompt for this shot.')


class SegmentOut(BaseModel):
    index: int
    scene_heading: str = Field(description='INT./EXT. LOCATION - TIME')
    summary: str
    shots: list[ShotOut]
    transition_out: str = Field(description='How this segment flows into the next one.')
    continuity: str = Field(description='End state carried into the next segment: positions, wardrobe, light, props, motion direction.')


class SegmentsOut(BaseModel):
    segments: list[SegmentOut]


# --------------------------------------------------------------------------- plan & result
class PlannedSegment(BaseModel):
    index: int
    start: float
    end: float
    beat_ids: list[int] = []
    source: str = ''               # the source text this segment must cover
    intensity: float = 3.0
    section: str = ''              # music: section label/role
    energy_level: str = ''
    beat_times: list[float] = []   # music: beat positions inside the segment (absolute)
    downbeat_times: list[float] = []
    shot_hint: str = ''            # e.g. "2-3 shots"

    @property
    def duration(self) -> float:
        return self.end - self.start


class Plan(BaseModel):
    increment: Increment
    total_seconds: float
    segments: list[PlannedSegment]
    beat_times: list[tuple[float, float]] = []  # (start, end) per story beat


class ScriptResult(BaseModel):
    settings: Settings
    source_kind: str
    bible: StoryBibleOut
    plan: Plan
    segments: list[SegmentOut] = []
    notes: list[str] = []
