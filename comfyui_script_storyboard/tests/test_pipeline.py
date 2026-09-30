import os
import random
import sys
import time
import zipfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from core.script_io import load_script, estimate_pages, check_length, ScriptTooLongError  # noqa: E402
from core.parser import parse_screenplay, parse_heading  # noqa: E402
from core.characters import build_bible, parse_overrides  # noqa: E402
from core.shots import plan_shots, PlannerSettings  # noqa: E402
from core.project import Project, parse_bind  # noqa: E402

SAMPLE = os.path.join(os.path.dirname(HERE), "examples", "sample_script.fountain")


@pytest.fixture(scope="module")
def sp():
    return parse_screenplay(load_script(SAMPLE))


@pytest.fixture(scope="module")
def bible(sp):
    return build_bible(sp, project_seed=7)


# ------------------------------------------------------------------ parsing
def test_scenes_and_headings(sp):
    assert sp.title == "The Last Diner"
    assert [s.location for s in sp.scenes] == [
        "DESERT HIGHWAY", "LAST CHANCE DINER", "DINER PARKING LOT", "LAST CHANCE DINER"]
    assert sp.scenes[1].int_ext == "INT"
    # LATER inherits the lighting of the previous scene
    assert sp.scenes[3].time_of_day.startswith("NIGHT")


def test_dialogue_parsing(sp):
    d = [e for e in sp.scenes[1].elements if e.type == "dialogue"]
    assert d[0].character == "MARA" and d[0].parenthetical == "tired"
    assert d[0].text.startswith("Coffee.")
    assert any(e.extension == "O.S." for e in d)


def test_heading_variants():
    assert parse_heading("INT./EXT. CAR - MOVING - DAY") == ("INT/EXT", "CAR - MOVING", "DAY")
    assert parse_heading("EXT. BEACH -- DAWN") == ("EXT", "BEACH", "DAWN")
    assert parse_heading("12 INT. OFFICE - NIGHT 12")[1] == "OFFICE"


def test_plain_pdf_style_text_without_blank_lines():
    text = "INT. OFFICE - DAY\n\nBOB sits.\n\nBOB\nHello there.\n(beat)\nGoodbye.\n\nALICE\nHi."
    sp = parse_screenplay(text)
    d = [e for e in sp.scenes[0].elements if e.type == "dialogue"]
    assert [e.character for e in d] == ["BOB", "ALICE"]
    assert "(beat)" in d[0].text


def test_fallback_chunking_for_prose():
    text = "\n\n".join(f"Paragraph {i} about the hero walking." for i in range(20))
    sp = parse_screenplay(text, fallback_paragraphs_per_scene=5)
    assert sp.fallback_chunked and len(sp.scenes) == 4


def test_fdx_and_docx_loaders(tmp_path):
    fdx = tmp_path / "s.fdx"
    fdx.write_text("""<?xml version="1.0"?><FinalDraft><Content>
<Paragraph Type="Scene Heading"><Text>INT. LAB - NIGHT</Text></Paragraph>
<Paragraph Type="Action"><Text>DR. VOSS (50s), wiry, peers into a microscope.</Text></Paragraph>
<Paragraph Type="Character"><Text>DR. VOSS</Text></Paragraph>
<Paragraph Type="Dialogue"><Text>It's alive.</Text></Paragraph>
</Content></FinalDraft>""")
    sp = parse_screenplay(load_script(str(fdx)))
    assert sp.scenes[0].location == "LAB"
    assert sp.scenes[0].elements[-1].character == "DR. VOSS"

    docx = tmp_path / "s.docx"
    ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    paras = "".join(f"<w:p><w:r><w:t>{t}</w:t></w:r></w:p>" for t in
                    ["INT. LAB - NIGHT", "", "Rain.", "", "VOSS", "Hello."])
    with zipfile.ZipFile(docx, "w") as z:
        z.writestr("word/document.xml", f"<w:document {ns}><w:body>{paras}</w:body></w:document>")
    sp = parse_screenplay(load_script(str(docx)))
    assert sp.scenes[0].elements[-1].text == "Hello."


def test_page_limit():
    text = "x\n" * (55 * 1001)
    assert estimate_pages(text) == 1001
    with pytest.raises(ScriptTooLongError):
        check_length(text, 1000)


# ------------------------------------------------------------------ characters
def test_short_cue_linked_to_full_name_intro(bible):
    mara = bible.characters["MARA"]
    assert "MARA OKAFOR" not in bible.characters
    assert "shaved head" in mara.description
    assert "climbs out" not in mara.description
    assert mara.age == "30s" and mara.gender == "female"
    assert bible.characters["EDDIE"].gender == "male"


def test_variant_inherits_parent(bible):
    y = bible.characters["YOUNG MARA"]
    assert y.variant_of == "MARA"
    frag = bible.prompt_fragment("YOUNG MARA", 2)
    assert "younger" in frag and "shaved head" in frag


def test_continuity_unless_otherwise_stated(bible):
    # scene 3 (index 2): natural-language wardrobe change + scene-only look note
    assert "yellow raincoat" in bible.prompt_fragment("MARA", 2)
    assert "soaked" in bible.prompt_fragment("MARA", 2)
    # the wardrobe change persists, the scene-only look doesn't
    assert "yellow raincoat" in bible.prompt_fragment("MARA", 3)
    assert "soaked" not in bible.prompt_fragment("MARA", 3)
    # before the change she's in her canonical look
    assert "raincoat" not in bible.prompt_fragment("MARA", 1)


def test_overrides_win(sp):
    b = build_bible(sp, overrides=parse_overrides(
        '{"eddie": {"description": "a tall thin cook with a grey ponytail", "voice": "en-GB-RyanNeural"}}'))
    assert b.characters["EDDIE"].description.startswith("a tall thin")
    assert b.characters["EDDIE"].voice == "en-GB-RyanNeural"
    b2 = build_bible(sp, overrides=parse_overrides("MARA: a short woman with red braids"))
    assert "red braids" in b2.characters["MARA"].description


def test_consistency_toggle(sp):
    b = build_bible(sp, consistent=False)
    assert "shaved head" not in b.prompt_fragment("MARA", 1)


def test_seeds_are_stable(sp):
    assert build_bible(sp, 7).characters["MARA"].seed == build_bible(sp, 7).characters["MARA"].seed
    assert build_bible(sp, 7).characters["MARA"].seed != build_bible(sp, 8).characters["MARA"].seed


# ------------------------------------------------------------------ shots
def test_shot_order_and_content(sp, bible):
    shots = plan_shots(sp, bible, PlannerSettings())
    assert [s.index for s in shots] == list(range(len(shots)))
    assert [s.scene_index for s in shots] == sorted(s.scene_index for s in shots)
    dl = [s for s in shots if s.kind == "dialogue"]
    assert len(dl) == 7 and all(s.dialogue["text"] for s in dl)
    for s in shots:
        assert "Mara" not in s.prompt and "MARA" not in s.prompt  # names never reach the image model
    # every shot featuring Mara carries her fixed description
    assert all("shaved head" in s.prompt for s in shots if "MARA" in s.characters)


def test_max_shots_per_scene_keeps_dialogue(sp, bible):
    shots = plan_shots(sp, bible, PlannerSettings(max_shots_per_scene=3))
    sc1 = [s for s in shots if s.scene_index == 1]
    assert sum(1 for s in sc1 if s.kind == "dialogue") == 4  # dialogue is never merged away


# ------------------------------------------------------------------ project + references
def test_project_roundtrip_and_references(tmp_path, sp, bible):
    from PIL import Image
    shots = plan_shots(sp, bible, PlannerSettings())
    p = Project.create(str(tmp_path / "proj"), sp, bible, shots, {"width": 64, "height": 36})
    p = Project.load(p.root)
    assert len(p.shots) == len(shots)
    assert p.next_missing() == 0

    img = tmp_path / "r.png"
    Image.new("RGB", (32, 32), "red").save(img)
    p.set_reference(1, str(img), "character:mara")
    p.set_reference(2, str(img), "location:diner")
    p.set_reference(3, str(img), "style")
    p.set_reference(4, str(img), "shot:SC0001_SH001", usage="use_as_frame")
    p.set_reference(5, str(img), "scenes:4", usage="init_image")
    with pytest.raises(ValueError):
        p.set_reference(6, str(img), "all")

    p = Project.load(p.root)
    assert len(p.references) == 5
    assert p.bible.characters["MARA"].reference_image == "references/ref1.png"
    mara_shot = next(s for s in p.shots if s.characters and s.characters[0] == "MARA" and s.scene_index == 1)
    ids = [r["id"] for r in p.references_for_shot(mara_shot)]
    assert ids[0] == "ref1" and "ref2" in ids and ids[-1] == "ref3"
    highway = p.shots[1]
    assert [r["id"] for r in p.references_for_shot(highway)] == ["ref3"]
    # use_as_frame replaces generation for that shot, so next_missing skips it
    assert p.frame_override(p.shots[0]) is not None
    assert p.next_missing() == 1
    last_scene = [s for s in p.shots if s.scene_index == 3][0]
    assert any(r["usage"] == "init_image" for r in p.references_for_shot(last_scene))
    # clearing a slot
    p.set_reference(5, None, "")
    assert len(Project.load(p.root).references) == 4


def test_parse_bind():
    assert parse_bind("JOHN") == {"type": "character", "value": "JOHN"}
    assert parse_bind("scenes:3-10") == {"type": "scenes", "start": 3, "end": 10}
    assert parse_bind("") == {"type": "all"}
    with pytest.raises(ValueError):
        parse_bind("bogus:1")


# ------------------------------------------------------------------ scale
def _synthetic_script(pages: int) -> str:
    rnd = random.Random(0)
    names = ["ALICE", "BOB", "CARLOS", "DANA", "EVE", "FRANK"]
    out = ["Title: Epic", ""]
    for n in names:
        out += [f"EXT. FIELD - DAY", "", f"{n} ({rnd.randint(20, 60)}), a person with a {n.lower()} hat.", ""]
    lines = len(out)
    k = 0
    while lines < pages * 55:
        k += 1
        out += [f"INT. ROOM {k} - {'DAY' if k % 2 else 'NIGHT'}", "",
                "Wind rattles the window. " * rnd.randint(1, 4), ""]
        for _ in range(rnd.randint(3, 8)):
            n = rnd.choice(names)
            out += [n, "Some line of dialogue that goes on for a bit.", ""]
            if rnd.random() < 0.4:
                out += [f"{n.title()} crosses to the door.", ""]
        lines = len(out)
    return "\n".join(out)


def test_thousand_page_script_is_fast():
    text = _synthetic_script(1000)
    assert 990 <= estimate_pages(text) <= 1010
    t0 = time.time()
    sp = parse_screenplay(text)
    b = build_bible(sp)
    shots = plan_shots(sp, b, PlannerSettings())
    elapsed = time.time() - t0
    assert len(sp.scenes) > 1000 and len(shots) > 8000
    assert elapsed < 60, elapsed
