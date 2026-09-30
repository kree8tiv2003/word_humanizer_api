"""End-to-end run of every node with a fake ComfyUI environment.
Needs torch + numpy + Pillow + ffmpeg (system or imageio-ffmpeg)."""

import json
import os
import shutil
import sys
import types

import pytest

torch = pytest.importorskip("torch")
HERE = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PKG_DIR))


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("comfy")
    (root / "input" / "scripts").mkdir(parents=True)
    (root / "output").mkdir()
    shutil.copy(os.path.join(PKG_DIR, "examples", "sample_script.fountain"), root / "input" / "scripts")
    fp = types.ModuleType("folder_paths")
    fp.get_input_directory = lambda: str(root / "input")
    fp.get_output_directory = lambda: str(root / "output")
    sys.modules["folder_paths"] = fp
    pkg = os.path.basename(PKG_DIR)
    for m in list(sys.modules):
        if m.startswith(pkg):
            del sys.modules[m]
    mod = __import__(pkg + ".nodes", fromlist=["x"])
    try:
        mod.media.ffmpeg_exe()
    except RuntimeError:
        pytest.skip("ffmpeg unavailable")
    return mod, root


def _res(r):
    return r["result"] if isinstance(r, dict) else r


def test_full_pipeline(env):
    n, root = env
    M = n.NODE_CLASS_MAPPINGS
    assert "scripts/sample_script.fountain" in M["S2S_LoadScript"].INPUT_TYPES()["required"]["script_file"][0]

    script, info = _res(M["S2S_LoadScript"]().load("scripts/sample_script.fountain", 1000))
    sp, stats = _res(M["S2S_ParseScript"]().parse(script, 8))
    bible, summary = _res(M["S2S_CharacterBible"]().build(sp, True, 99, 1, True, ""))
    shots, preview = _res(M["S2S_PlanShots"]().plan(sp, bible, "storyboard sketch", "", "", "blurry",
                                                    True, True, 12, 45, False, 1, -1))
    shots2, report = _res(M["S2S_EnhancePrompts"]().enhance(shots, bible, "none", "", "low", "", "demo", 1, -1))
    name, txt = _res(M["S2S_SaveProject"]().save(sp, bible, shots2, "demo", "16:9 SD1.5 (768x432)", 24))
    total = len(shots["shots"])

    # ---- reference images: 2 of 5 slots used
    red = torch.zeros((1, 64, 64, 3)); red[..., 0] = 1
    blue = torch.zeros((1, 80, 40, 3)); blue[..., 2] = 1
    out = M["S2S_SetReferenceImages"]().set_refs("demo", 0.6, False,
                                                  image_1=red, bind_1="character:MARA", usage_1="reference",
                                                  weight_1=1.0, image_2=blue, bind_2="shot:SC0001_SH001",
                                                  usage_2="use_as_frame", weight_2=1.0)
    assert "ref1" in out["ui"]["text"][0] and len(out["ui"]["images"]) == 2

    # ---- character sheet iterator
    it = M["S2S_CharacterIterator"]()
    pos, neg, seed, cname, *_ = _res(it.next("demo", "next_missing", 0, "{description}, {style}"))
    assert cname in ("EDDIE",)  # MARA already has an uploaded reference
    M["S2S_SaveCharacterReference"]().save(torch.rand((1, 64, 64, 3)), "demo", cname)
    with pytest.raises(RuntimeError):
        it.next("demo", "next_missing", 0, "{description}")

    # ---- render loop exactly like queued runs in next_missing mode
    si, save = M["S2S_ShotIterator"](), M["S2S_SaveShotImage"]()
    seen = []
    for _ in range(total):
        try:
            r = _res(si.next("demo", "next_missing", 0, 256, 0))
        except RuntimeError:
            break
        pos, neg, seed, w, h, idx, refs, nref, has_ref, info, _ = r
        assert (w, h) == (768, 432) and refs.shape[1:] == (256, 256, 3)
        seen.append(idx)
        # reference-conditioning + latent nodes with stand-in model objects
        class CV:
            def encode_image(self, img, crop=True):
                return img
        class SM:
            def get_cond(self, cv):
                return torch.ones((1, 4, 8))
        cond = [[torch.zeros((1, 10, 8)), {}]]
        c2, applied = M["S2S_ReferenceConditioning"]().apply(cond, SM(), CV(), "demo", idx, 0.5, 3, "center")
        assert c2[0][0].shape[1] == 10 + 4 * applied
        class VAE:
            def encode(self, px):
                return torch.zeros((1, 4, px.shape[1] // 8, px.shape[2] // 8))
        lat, denoise, used = M["S2S_ShotLatent"]().make(VAE(), "demo", idx, w, h, "16")
        assert lat["samples"].shape[1] in (4, 16) and denoise == 1.0
        img = torch.rand((1, h, w, 3))
        save.save(img, "demo", idx)
    assert 0 not in seen  # shot 0 uses the uploaded frame
    assert seen == sorted(seen) and len(seen) == total - 1
    shots_with_mara = [s for s in n.Project.load(n.project_root("demo")).shots if "MARA" in s.characters]
    assert shots_with_mara

    seq, cnt, timeline = M["S2S_LoadStoryboardSequence"]().load("demo", 0, 100, 128, 72, "placeholder")
    assert cnt == total and seq.shape == (total, 72, 128, 3)

    # ---- TTS via command backend (python writes a sine wave)
    tts_cmd = (f"{sys.executable} -c \"import sys,math,wave,struct;"
               "t=open(sys.argv[1]).read();n=int(8000*(0.3+0.05*len(t.split())));"
               "w=wave.open(sys.argv[2],'wb');w.setnchannels(1);w.setsampwidth(2);w.setframerate(8000);"
               "w.writeframes(b''.join(struct.pack('<h',int(8000*math.sin(i/9))) for i in range(n)));w.close()\" "
               "{text_file} {output}")
    rep = json.loads(_res(M["S2S_DialogueAudio"]().run("demo", "command", 0, -1, False, "", "", tts_cmd))[1])
    assert rep["generated"] == 7 and rep["failed"] == 0, rep

    # ---- lip-sync: batch backend (ffmpeg stands in for Wav2Lip) ...
    ff = n.media.ffmpeg_exe()
    ls_cmd = (f"{ff} -y -loglevel error -loop 1 -i {{image}} -i {{audio}} -shortest "
              "-vf scale=320:180 -c:v libx264 -pix_fmt yuv420p -c:a aac {output}")
    rep = json.loads(_res(M["S2S_LipSyncBatch"]().run("demo", "custom-command", "", "", "", ls_cmd, "",
                                                      0, 2, False, True))[1])
    assert rep["lipsynced"] == 2, rep
    # ... and the node route (frames + AUDIO -> any lip-sync node -> Save Lip-Sync Clip)
    frames, audio, idx, nframes, fps, info, _ = _res(M["S2S_LipSyncShotInputs"]().get("demo", "next_missing", 0, 12, True, True))
    assert frames.shape[0] == nframes and audio["waveform"].dim() == 3
    M["S2S_SaveLipSyncClip"]().save(frames, "demo", idx, fps, audio)
    assert os.path.exists(n.Project.load(n.project_root("demo")).clip_path(idx))

    # ---- final assembly
    out = M["S2S_AssembleVideo"]().run("demo", "final", 320, 180, 12, 1, -1, True, True, True, True, "placeholder")
    video, report = out["result"]
    rep = json.loads(report)
    assert os.path.getsize(video) > 1000 and rep["shots"] == total
    dur = n.media.media_duration(video)
    assert abs(dur - rep["duration_s"]) < 1.5, (dur, rep["duration_s"])
    srt = open(rep["srt"]).read()
    assert "MARA: Coffee. Black." in srt
    # re-assembly reuses cached segments
    out2 = M["S2S_AssembleVideo"]().run("demo", "final", 320, 180, 12, 1, -1, True, True, True, False, "placeholder")
    assert os.path.exists(out2["result"][0])

    status, remaining = _res(M["S2S_ProjectStatus"]().run("demo"))
    assert remaining == 0
