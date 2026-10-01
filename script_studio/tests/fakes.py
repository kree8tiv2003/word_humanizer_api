"""A stand-in for the Anthropic client that returns valid structured output derived from the prompt."""
from __future__ import annotations

import re
from types import SimpleNamespace

from script_studio.models import (BeatOut, BeatsOut, CharacterOut, DialogueLine, LocationOut, SegmentOut, SegmentsOut,
                                  ShotOut, StoryBibleOut)


class FakeMessages:
    def __init__(self, owner):
        self.owner = owner

    def parse(self, model, max_tokens, system, messages, output_format):
        prompt = messages[0]['content']
        self.owner.calls.append({'model': model, 'system': system, 'prompt': prompt, 'schema': output_format.__name__})
        if self.owner.truncate_once and output_format is SegmentsOut and len(re.findall(r'SEGMENT index=', prompt)) > 1:
            self.owner.truncate_once = False
            return SimpleNamespace(stop_reason='max_tokens', parsed_output=None)
        out = getattr(self, '_' + output_format.__name__)(prompt)
        return SimpleNamespace(stop_reason='end_turn', parsed_output=out)

    @staticmethod
    def _units(prompt):
        return [int(x) for x in re.findall(r'^\[(\d+)[\] ]', prompt, re.M)]

    def _beats(self, prompt):
        ids = self._units(prompt)
        lo, hi = min(ids), max(ids)
        out = []
        for a in range(lo, hi + 1, 2):
            out.append(BeatOut(first_unit=a, last_unit=min(a + 1, hi), summary=f'Beat covering units {a}-{min(a + 1, hi)}',
                               emotion='wonder', intensity=1 + (a % 5), weight=1.0 + (a % 3) * 0.5,
                               characters=['Mara'], location='Lighthouse' if a < 4 else 'Harbor'))
        return out

    def _StoryBibleOut(self, prompt):
        return StoryBibleOut(
            title='The Keeper', logline='A lighthouse keeper guides a lost ship home.', genre='drama', tone='hopeful',
            visual_style='cinematic photoreal, teal and amber', pacing_notes='slow build, fast storm, quiet end',
            world_rules='the lamp is a motif',
            characters=[CharacterOut(name='Mara', role='keeper', visual_signature='a woman in her sixties, grey braid, yellow oilskin coat', voice='low, calm')],
            locations=[LocationOut(name='Lighthouse', visual_signature='a white stone lighthouse on black rocks at night')],
            beats=self._beats(prompt))

    def _BeatsOut(self, prompt):
        return BeatsOut(beats=self._beats(prompt), new_characters=[], new_locations=[])

    def _SegmentsOut(self, prompt):
        segs = []
        for idx, a, b in re.findall(r'SEGMENT index=(\d+) .*?start=([\d.]+)s, end=([\d.]+)s', prompt):
            a, b = float(a), float(b)
            mid = (a + b) / 2
            shots = [ShotOut(start=mid, end=b + 3, camera='Close-up', prompt='Close-up: a woman in her sixties, grey braid, yellow oilskin coat, lighting the lamp.',
                             action='Mara strikes a match.', dialogue=[DialogueLine(character='Mara', line='Not tonight.', delivery='quietly')],
                             voiceover='', lyrics='', sound='wind', vfx='', negative='flicker'),
                     ShotOut(start=a, end=mid, camera='Wide shot', prompt='Wide shot: a white stone lighthouse on black rocks at night, storm rolling in.',
                             action='Storm clouds roll in.', dialogue=[], voiceover='', lyrics='', sound='thunder', vfx='lightning', negative='')]
            segs.append(SegmentOut(index=int(idx), scene_heading='EXT. LIGHTHOUSE - NIGHT', summary=f'Segment {idx}', shots=shots,
                                   transition_out='Match cut on the lamp', continuity='Mara at the lamp, facing the sea'))
        return SegmentsOut(segments=segs)


class FakeClient:
    def __init__(self, truncate_once=False):
        self.calls = []
        self.truncate_once = truncate_once
        self.messages = FakeMessages(self)
