"""smoke: the checks of each file type, the files a run must have written,
and the `smoke` object a run publishes."""
import hashlib
import io
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import output_kinds
import run_regression
import smoke

FIX = Path(__file__).resolve().parent / "fixtures" / "smoke"
MIB = 1024 * 1024
TEXT = ("A man in a red hat stands on a street at dusk. Warm light, shallow depth of field, "
        "35 mm photograph, detailed fabric texture, soft bokeh in the background.")


def _ids(problems):
    return [p["check"] for p in problems]


def _run(tmp_path, files, unsent=()):
    """An outputs folder and the run record that lists it, as the runner
    leaves them: {name: bytes or a file to copy}. A name in `unsent` is
    listed and not written, like a file over the worker's size limit."""
    out = tmp_path / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, data in sorted(files.items()):
        data = data.read_bytes() if isinstance(data, Path) else data
        rec = {"filename": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        if name in unsent:
            rec["truncated"] = True
        else:
            (out / name).parent.mkdir(parents=True, exist_ok=True)
            (out / name).write_bytes(data)
        outputs.append(rec)
    return out, {"status": "ok", "outputs": outputs}


def _checked(tmp_path, files, workflow=None, expect=None, unsent=()):
    out, rec = _run(tmp_path, files, unsent)
    return smoke.check_outputs(out, rec, workflow or {}, expect, "ok")


def _one(tmp_path, name, data, expect=None):
    """(the file entry, its problems) of one output file, through check_outputs."""
    res = _checked(tmp_path, {name: data}, expect=expect)
    assert all(p["file"] == name for p in res["problems"])
    return res["files"][0], res["problems"]


# -- images --------------------------------------------------------------------------
def _png(arr) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def _picture(h=352, w=640):
    """Smooth colour bands with a little texture: a stand-in for a real output."""
    yy, xx = np.mgrid[0:h, 0:w]
    bands = np.stack([120 + 80 * np.sin(xx / 40), 110 + 70 * np.cos(yy / 30),
                      100 + 60 * np.sin((xx + yy) / 50)], axis=-1)
    return np.clip(bands + 6 * np.sin(xx * 1.7 + yy * 2.3)[..., None], 0, 255).astype(np.uint8)


def _flat(side, value):
    return np.full((side, side, 3), value, np.uint8)


def test_a_real_picture_passes(tmp_path):
    entry, problems = _one(tmp_path, "wf_00001_.png", _png(_picture()))
    assert problems == []
    assert list(entry) == ["name", "type", "bytes", "sha256", "width", "height", "std",
                           "flat_share"]
    assert (entry["type"], entry["width"], entry["height"]) == ("image", 640, 352)
    assert 20 < entry["std"] < 60 and entry["flat_share"] < 0.05


@pytest.mark.parametrize("arr", [_flat(1024, 0), _flat(1024, 127), _flat(1024, 255),
                                 _flat(64, 0)], ids=["black", "grey", "white", "black-64px"])
def test_a_blank_image_fails(tmp_path, arr):
    """The last one is what `manager_import_smoke` writes on purpose: a
    workflow whose output is blank cannot be smoke level."""
    entry, problems = _one(tmp_path, "wf_00001_.png", _png(arr))
    assert (entry["std"], entry["flat_share"]) == (0.0, 1.0)
    assert problems == [{"file": "wf_00001_.png", "check": "blank",
                         "text": "blank: brightness spread 0.00, under 1.0"}]


def test_a_speck_on_a_flat_colour_fails(tmp_path):
    arr = _flat(1024, 0)
    arr[500:520, 500:520] = 255   # 0.04 % of the pixels
    entry, problems = _one(tmp_path, "wf_00001_.png", _png(arr))
    assert (entry["std"], entry["flat_share"]) == (4.98, 0.9996)
    assert _ids(problems) == ["flat"]
    assert problems[0]["text"] == ("one flat colour with a speck: 99.96 % of the pixels have "
                                   "one brightness, over 99.5 %")


def test_a_very_dark_picture_passes(tmp_path):
    """The reason the limit is 1.0 and not 2.0."""
    dark = (_picture() * 0.06).astype(np.uint8)
    entry, problems = _one(tmp_path, "wf_00001_.png", _png(dark))
    assert problems == [] and 1.0 < entry["std"] < 3.0


def test_a_render_on_a_black_background_passes(tmp_path):
    arr = _flat(1024, 0)
    arr[200:824, 200:824] = _picture(624, 624)   # the object covers 37 % of the frame
    entry, problems = _one(tmp_path, "wf_00001_.png", _png(arr))
    assert problems == [] and 0.6 < entry["flat_share"] < 0.65


def test_half_a_picture_is_not_caught(tmp_path):
    """Smoke level does not judge what a picture shows."""
    arr = _flat(1024, 0)
    arr[:512] = _picture(512, 1024)
    entry, problems = _one(tmp_path, "wf_00001_.png", _png(arr))
    assert problems == [] and 0.5 <= entry["flat_share"] < 0.52


def test_the_type_of_a_file_is_its_extension_in_any_case(tmp_path):
    res = _checked(tmp_path, {"WF_00001_.PNG": _png(_flat(64, 0)), "a.Txt": b"", "b.webp": b"",
                              "c": b"", "d.png.bak": b"x"})
    assert [(f["name"], f["type"]) for f in res["files"]] == [
        ("WF_00001_.PNG", "image"), ("a.Txt", "text"), ("b.webp", None), ("c", None),
        ("d.png.bak", None)]
    assert [(p["file"], p["check"]) for p in res["problems"]] == [("WF_00001_.PNG", "blank"),
                                                                  ("a.Txt", "empty")]


def test_a_tiny_image_fails(tmp_path):
    entry, problems = _one(tmp_path, "wf_00001_.png", _png(_picture(32, 32)))
    assert problems == [{"file": "wf_00001_.png", "check": "size",
                         "text": "32x32 px: each side must have at least 64 px"}]
    assert _ids(_one(tmp_path, "wf_00001_.png", _png(_picture(64, 64)))[1]) == []


def test_an_image_is_held_to_the_expected_size(tmp_path):
    png = _png(_picture())
    assert _one(tmp_path, "a.png", png, {"width": 640, "height": 352})[1] == []
    _, problems = _one(tmp_path, "a.png", png, {"width": 1024, "height": 1024})
    assert [(p["check"], p["text"]) for p in problems] == [
        ("expect", "width 640 px, 1024 px expected"),
        ("expect", "height 352 px, 1024 px expected")]
    # Only what the entry sets, and only a real number, is held against the file.
    assert _one(tmp_path, "a.png", png, {"height": 352, "frames": 9, "seconds": 3})[1] == []
    for odd in ({"width": None}, {"width": True}, {"width": "640"}, {"width": 0}, [], "x"):
        assert _one(tmp_path, "a.png", png, odd)[1] == []


def test_a_16_bit_grey_image_is_scaled_not_clipped(tmp_path):
    yy, xx = np.mgrid[0:128, 0:128]
    ramp = (xx * 500 + yy * 10).astype(np.uint16)   # every value above 255 from the second column
    entry, problems = _one(tmp_path, "depth.png", _png(ramp))
    assert problems == [] and entry["std"] > 50


def test_an_image_in_another_format_is_still_decoded(tmp_path):
    """Pillow decides what the bytes are: the check asks for a picture."""
    buf = io.BytesIO()
    Image.fromarray(_picture()).save(buf, "JPEG")
    assert _one(tmp_path, "a.png", buf.getvalue())[1] == []


# -- text ----------------------------------------------------------------------------
def test_a_sentence_passes(tmp_path):
    entry, problems = _one(tmp_path, "wf_00001.txt", TEXT.encode())
    assert problems == []
    assert entry == {"name": "wf_00001.txt", "type": "text", "bytes": len(TEXT),
                     "sha256": hashlib.sha256(TEXT.encode()).hexdigest(),
                     "chars": len(TEXT.replace(" ", "")), "words": len(TEXT.split())}
    # A byte order mark is no character, and other line ends change nothing.
    marked = b"\xef\xbb\xbf" + TEXT.replace(". ", ".\r\n").encode()
    assert _one(tmp_path, "wf_00001.txt", marked)[0]["chars"] == entry["chars"]


@pytest.mark.parametrize("text, checks, said", [
    ("", ["empty"], "empty"),
    ("  \n\n \t ", ["empty"], "empty"),
    ("Yes.", ["empty"], "4 non-blank characters, at least 8 expected"),
    ("!" * 300, ["repeat"], "only 1 different character"),
    (" ".join(["the"] * 200), ["repeat", "repeat"], "only 3 different characters"),
    (" ".join(["Stop", "stop", "STOP"] * 7 + ["now", "please"]), ["repeat"],
     "one word is 21 of the 23 words"),
], ids=["empty", "blanks", "short", "one-character", "one-word", "one-word-any-case"])
def test_a_text_that_says_nothing_fails(tmp_path, text, checks, said):
    _, problems = _one(tmp_path, "wf_00001.txt", text.encode())
    assert _ids(problems) == checks and problems[0]["text"] == said


def test_the_repeat_rule_needs_twenty_words_and_more_than_half(tmp_path):
    few = " ".join(["no"] * 15 + ["one", "two", "three", "four"])           # 19 words
    half = " ".join(["no"] * 10 + [f"word{i}" for i in range(10)])           # 10 of 20
    over = " ".join(["no"] * 11 + [f"word{i}" for i in range(9)])            # 11 of 20
    assert _one(tmp_path, "a.txt", few.encode())[1] == []
    assert _one(tmp_path, "a.txt", half.encode())[1] == []
    assert _ids(_one(tmp_path, "a.txt", over.encode())[1]) == ["repeat"]


def test_the_bars_of_a_table_are_no_words(tmp_path):
    """A model that is asked for a table answers with one. Its bars and its
    rule line outnumber what narrow cells say: 52 of these 92 tokens are "|"."""
    table = "| Year | Wins | Losses |\n|---|---|---|\n" + "".join(
        f"| {2000 + i} | {i} | {12 - i} |\n" for i in range(12))
    assert table.split().count("|") == 52 and len(table.split()) == 92
    entry, problems = _one(tmp_path, "a.txt", table.encode())
    assert problems == [] and entry["words"] == 92
    # What stands between the bars is still held to the rule.
    stuck = "| Answer | Again | More |\n" + "| none | none | none |\n" * 12
    _, problems = _one(tmp_path, "a.txt", stuck.encode())
    assert problems == [{"file": "a.txt", "check": "repeat",
                         "text": "one word is 36 of the 39 words"}]
    # A word with a letter or a digit counts whatever is attached to it.
    marked = " ".join(["**no**"] * 11 + [f"`word{i}`" for i in range(9)])
    assert _ids(_one(tmp_path, "a.txt", marked.encode())[1]) == ["repeat"]


def test_a_text_is_held_to_the_expected_length(tmp_path):
    assert _one(tmp_path, "a.txt", TEXT.encode(), {"min_chars": 40})[1] == []
    _, problems = _one(tmp_path, "a.txt", b"A short answer.", {"min_chars": 40})
    assert problems == [{"file": "a.txt", "check": "expect",
                         "text": "13 non-blank characters, at least 40 expected"}]
    # The entry's number replaces the 8, also downwards.
    assert _one(tmp_path, "a.txt", b"Yes.", {"min_chars": 4})[1] == []
    assert _ids(_one(tmp_path, "a.txt", b"", {"min_chars": 4})[1]) == ["empty"]


def test_a_text_over_the_size_limit_is_not_read(tmp_path):
    big = (TEXT + "\n").encode() * (MIB // len(TEXT) + 1)
    assert len(big) > MIB
    entry, problems = _one(tmp_path, "a.txt", big)
    assert problems == [{"file": "a.txt", "check": "size",
                         "text": f"{len(big)} bytes: a text output may have at most {MIB}"}]
    assert "chars" not in entry
    assert _one(tmp_path, "a.txt", big[:MIB])[1] == []


def test_a_text_that_is_not_utf8_cannot_be_read(tmp_path):
    cut = "Der Hut ist rot, die Stra\xdfe".encode()[:-2]   # ends inside a character
    for data in (b"\xff\xfe\x00A long enough line of text", cut):
        entry, problems = _one(tmp_path, "a.txt", data)
        assert _ids(problems) == ["decode"] and "chars" not in entry
        assert problems[0]["text"].startswith("could not be read: UnicodeDecodeError: ")


# -- audio ---------------------------------------------------------------------------
def test_sound_passes_and_says_what_it_is(tmp_path):
    entry, problems = _one(tmp_path, "wf_00001.flac", FIX / "audio_1s.flac")
    assert problems == []
    assert entry == {"name": "wf_00001.flac", "type": "audio", "bytes": 21900,
                     "sha256": hashlib.sha256((FIX / "audio_1s.flac").read_bytes()).hexdigest(),
                     "sample_rate": 8000, "channels": 1, "bits": 16, "seconds": 1.0,
                     "bytes_per_second": 13614, "pcm_md5": "abfaa56541974a64950333b3fa71fd6e"}


def test_quiet_sound_is_not_silence(tmp_path):
    """The same sound at 1 % volume."""
    entry, problems = _one(tmp_path, "wf_00001.flac", FIX / "audio_quiet_1s.flac")
    assert problems == [] and entry["bytes_per_second"] == 7108


def test_silence_fails(tmp_path):
    """An all-NaN signal and a constant offset are written the same way: a
    block of equal samples takes a few bytes."""
    entry, problems = _one(tmp_path, "wf_00001.flac", FIX / "audio_silent_1s.flac")
    assert (entry["sample_rate"], entry["channels"], entry["seconds"]) == (44100, 2, 1.0)
    assert problems == [{"file": "wf_00001.flac", "check": "silent",
                         "text": "silent or constant: 142 bytes of audio a second, under 4000"}]


def test_a_clip_under_a_second_fails(tmp_path):
    entry, problems = _one(tmp_path, "wf_00001.flac", FIX / "audio_short.flac")
    assert entry["seconds"] == 0.3
    assert problems == [{"file": "wf_00001.flac", "check": "duration",
                         "text": "0.30 s long, at least 1.0 s expected"}]


def test_sound_is_held_to_the_expected_length(tmp_path):
    """Within 5 %, and never tighter than half a second."""
    flac = FIX / "audio_1s.flac"
    for seconds in (1, 1.0, 1.5, 0.6):
        assert _one(tmp_path, "a.flac", flac, {"seconds": seconds})[1] == []
    for seconds, said in ((1.6, "1.6"), (30, "30")):
        _, problems = _one(tmp_path, "a.flac", flac, {"seconds": seconds})
        assert problems == [{"file": "a.flac", "check": "expect",
                             "text": f"1.0 s long, {said} s expected"}]
    # A long sound gets the 5 % of what is expected: 20.0 s passes for 19.1 to 21.
    long = _flac([4096] * 39 + [256], constant=True)
    for seconds, passes in ((19.1, True), (20, True), (21, True), (19, False), (21.1, False)):
        entry, problems = _one(tmp_path, "a.flac", long, {"seconds": seconds})
        assert entry["seconds"] == 20.0 and ("expect" not in _ids(problems)) is passes


def test_metadata_is_no_part_of_the_bytes_a_second(tmp_path):
    """ComfyUI stores the prompt in the file: the same sound with a longer
    prompt must not measure as more sound."""
    data = (FIX / "audio_silent_1s.flac").read_bytes()
    assert data[4] & 0x7F == 0 and not data[4] & 0x80   # STREAMINFO first, more blocks follow
    padded = data[:42] + bytes([1]) + (50000).to_bytes(3, "big") + bytes(50000) + data[42:]
    plain, _ = _one(tmp_path, "a.flac", data)
    entry, problems = _one(tmp_path, "a.flac", padded)
    assert _ids(problems) == ["silent"]
    assert (entry["bytes_per_second"], entry["pcm_md5"]) == (142, plain["pcm_md5"])
    assert entry["bytes"] == plain["bytes"] + 50004


def test_flac_streaminfo():
    info = smoke.flac_streaminfo((FIX / "audio_silent_1s.flac").read_bytes())
    assert info == {"sample_rate": 44100, "channels": 2, "bits": 16, "samples": 44100,
                    "pcm_md5": "d2b120199019b639d5a7e2b3463e9c97"}
    # The MD5 of the decoded samples: 44,100 stereo frames of 16-bit zeros.
    assert info["pcm_md5"] == hashlib.md5(bytes(44100 * 2 * 2)).hexdigest()
    assert smoke.flac_streaminfo(_flac([64]))["pcm_md5"] is None   # the writer left it out
    for data in (b"", b"fLaC", b"RIFF" + bytes(60), (FIX / "mesh_12.glb").read_bytes(),
                 (FIX / "audio_1s.flac").read_bytes()[:20]):
        assert smoke.flac_streaminfo(data) is None


def _crc(data: bytes, poly: int, bits: int) -> int:
    crc = 0
    for byte in data:
        crc ^= byte << (bits - 8)
        for _ in range(8):
            crc = ((crc << 1) ^ poly if crc & (1 << (bits - 1)) else crc << 1) & ((1 << bits) - 1)
    return crc


def _coded(n: int) -> bytes:
    """A frame or sample number the way a FLAC frame header holds it."""
    if n < 0x80:
        return bytes([n])
    size = next(s for s in range(2, 8) if n < 1 << (5 * s + 1))
    first = ((0xFF << (8 - size)) & 0xFF) | (n >> (6 * (size - 1)))
    return bytes([first] + [0x80 | (n >> (6 * k)) & 0x3F for k in range(size - 2, -1, -1)])


def _flac(blocks, rate=8000, variable=False, total=None, rate_code=0, constant=False,
          fill=None) -> bytes:
    """A FLAC file of 16-bit mono frames that store their samples as they
    are (`constant`: one sample for the whole frame; `fill`: these bytes,
    repeated), one frame for each entry of `blocks` (its samples). Every way
    a frame header can state its block size is used where the size allows it."""
    block_size = max(blocks)
    total = sum(blocks) if total is None else total
    info = (struct.pack(">HH", block_size, block_size) + bytes(6)
            + ((rate << 44) | (15 << 36) | total).to_bytes(8, "big") + bytes(16))
    out = b"fLaC" + bytes([0x80]) + len(info).to_bytes(3, "big") + info
    start = 0
    for i, n in enumerate(blocks):
        tail = b""
        if n == 192:
            code = 1
        elif n in (576, 1152, 2304, 4608):
            code = 2 + (576, 1152, 2304, 4608).index(n)
        elif n in (256, 512, 1024, 2048, 4096):
            code = 8 + (256, 512, 1024, 2048, 4096).index(n)
        elif n <= 256:
            code, tail = 6, bytes([n - 1])
        else:
            code, tail = 7, (n - 1).to_bytes(2, "big")
        tail += {12: bytes([rate // 1000]), 13: rate.to_bytes(2, "big"),
                 14: (rate // 10).to_bytes(2, "big")}.get(rate_code, b"")
        head = (bytes([0xFF, 0xF9 if variable else 0xF8, (code << 4) | rate_code, 0x08])
                + _coded(start if variable else i) + tail)
        if constant:
            body = b"\x00" + (i % 65536).to_bytes(2, "big")
        elif fill:
            body = b"\x02" + (fill * (2 * n // len(fill) + 1))[:2 * n]
        else:
            body = b"\x02" + b"".join(((s * 37 + i) % 65536).to_bytes(2, "big")
                                      for s in range(n))
        frame = head + bytes([_crc(head, 0x07, 8)]) + body
        out += frame + _crc(frame, 0x8005, 16).to_bytes(2, "big")
        start += n
    return out


@pytest.mark.parametrize("blocks, more", [
    ([4096, 4096, 1000], {}),
    ([192] * 3 + [17], {}),
    ([576, 1152, 2304, 4608, 256, 512, 1024, 2048, 4096, 100, 300], {"variable": True}),
    ([64] * 200 + [5], {}),                              # frame numbers of two bytes
    ([4096] * 40 + [4000], {"variable": True}),          # sample numbers of three bytes
    ([4096] * 2100 + [77], {"constant": True}),          # frame numbers of three bytes
    ([65535] * 40 + [60000], {"variable": True, "constant": True}),   # of four and five bytes
    ([1024, 1024], {"rate": 44100, "rate_code": 12}),
    ([1024, 1024], {"rate": 44100, "rate_code": 13}),
    ([1024, 1024], {"rate": 44100, "rate_code": 14}),
])
def test_a_flac_file_must_end_with_its_last_frame(tmp_path, blocks, more):
    """A cut FLAC file keeps its header, so only its last frame tells: that
    frame must be whole and end at the sample the header names."""
    data = _flac(blocks, **more)
    rate = more.get("rate", 8000)
    entry, problems = _one(tmp_path, "a.flac", data)
    assert entry["seconds"] == round(sum(blocks) / rate, 3)
    assert "decode" not in _ids(problems)
    # Without its last frame the file ends at a frame, and too early.
    whole = _flac(blocks[:-1], total=sum(blocks), **more)
    _, problems = _one(tmp_path, "a.flac", whole)
    assert problems == [{"file": "a.flac", "check": "decode", "text": (
        f"could not be read: the audio ends at sample {sum(blocks[:-1])}, the header says "
        f"{sum(blocks)}")}]
    # Cut anywhere else, or with one changed byte in the last frame, it ends in no frame.
    broken = bytearray(data)
    broken[-5] ^= 0x40
    for bad in (data[:-1], data[:-7], data[:len(data) // 2 + 3], bytes(broken)):
        _, problems = _one(tmp_path, "a.flac", bad)
        assert _ids(problems) == ["decode"], len(bad)


@pytest.mark.parametrize("last", [192, 576, 1152, 2304, 4608, 256, 512, 1024, 2048, 4096, 1,
                                  100, 257, 300, 4607])
def test_the_last_frame_states_its_length_in_every_form(tmp_path, last):
    for variable in (False, True):
        data = _flac([4608, 4608, last], variable=variable, constant=True)
        entry, problems = _one(tmp_path, "a.flac", data)
        assert entry["seconds"] == round((9216 + last) / 8000, 3)
        assert "decode" not in _ids(problems)


def test_sound_that_looks_like_frame_headers_is_still_sound(tmp_path):
    """The last frame is looked for from the end of the file. Bytes of the
    sound that read like the start of a frame are told apart by the checksum
    of a frame header, however many there are."""
    lure = bytes.fromhex("fff819080000")
    assert _crc(lure[:5], 0x07, 8) != lure[5]   # a header in every way but its checksum
    data = _flac([192, 192, 192], fill=lure)
    assert data.count(lure[:4]) > 50
    entry, problems = _one(tmp_path, "a.flac", data)
    assert entry["seconds"] == 0.072 and _ids(problems) == ["duration"]


def test_a_flac_file_that_is_no_usable_audio(tmp_path):
    _, problems = _one(tmp_path, "a.flac", _flac([192] * 3, rate=4000))
    assert problems == [{"file": "a.flac", "check": "decode", "text": (
        "not usable audio: 4000 Hz and 1 channel, 8000 to 192000 Hz expected")}]
    entry, problems = _one(tmp_path, "a.flac", _flac([192] * 3, total=0))
    assert problems == [{"file": "a.flac", "check": "duration",
                         "text": "no samples: the header gives the length 0"}]
    assert (entry["seconds"], entry["bytes_per_second"]) == (0.0, 0)
    for data, said in ((b"RIFF" + bytes(100), "not a FLAC file with a STREAMINFO block"),
                       (_flac([192])[:30], "not a FLAC file with a STREAMINFO block"),
                       (_flac([192])[:42], "the audio data does not end with a whole frame")):
        _, problems = _one(tmp_path, "a.flac", data)
        assert problems == [{"file": "a.flac", "check": "decode",
                             "text": f"could not be read: {said}"}]


# -- meshes --------------------------------------------------------------------------
def _glb_positions(data: bytes):
    """(where the positions of a fixture GLB start, how many vertices)."""
    json_len = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20:20 + json_len])
    accessor = gltf["accessors"][gltf["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    view = gltf["bufferViews"][accessor["bufferView"]]
    return 20 + json_len + 8 + view.get("byteOffset", 0), accessor["count"]


def _moved(change) -> bytes:
    """mesh_300.glb with its positions changed by `change(xyz)`."""
    data = bytearray((FIX / "mesh_300.glb").read_bytes())
    at, count = _glb_positions(data)
    xyz = np.frombuffer(bytes(data[at:at + count * 12]), "<f4").reshape(-1, 3).copy()
    change(xyz)
    data[at:at + count * 12] = xyz.astype("<f4").tobytes()
    return bytes(data)


def _glb(gltf: dict, blob: bytes) -> bytes:
    doc = json.dumps(gltf).encode()
    doc += b" " * (-len(doc) % 4)
    return (struct.pack("<4sII", b"glTF", 2, 28 + len(doc) + len(blob))
            + struct.pack("<II", len(doc), 0x4E4F534A) + doc
            + struct.pack("<II", len(blob), 0x004E4942) + blob)


def test_a_mesh_from_comfyui_passes(tmp_path):
    entry, problems = _one(tmp_path, "wf_mesh_00001_.glb", FIX / "mesh_300.glb")
    assert problems == []
    assert entry == {"name": "wf_mesh_00001_.glb", "type": "mesh", "bytes": 11164,
                     "sha256": hashlib.sha256((FIX / "mesh_300.glb").read_bytes()).hexdigest(),
                     "vertices": 300, "faces": 560, "extent": [2.0, 2.0, 2.0]}


def test_a_mesh_of_a_few_points_fails(tmp_path):
    entry, problems = _one(tmp_path, "wf_mesh_00001_.glb", FIX / "mesh_12.glb")
    assert (entry["vertices"], entry["faces"]) == (12, 20)
    assert problems == [{"file": "wf_mesh_00001_.glb", "check": "size", "text": (
        "12 vertices and 20 faces, at least 100 and 100 expected")}]


def test_a_mesh_with_a_position_that_is_not_a_number_fails(tmp_path):
    def spoil(xyz):
        xyz[10, 1] = np.nan
        xyz[20] = np.inf
    entry, problems = _one(tmp_path, "a.glb", _moved(spoil))
    assert problems == [{"file": "a.glb", "check": "nonfinite",
                         "text": "positions that are NaN or infinite: 2 of 300"}]
    assert entry["extent"] is None
    json.dumps(entry, allow_nan=False)   # nothing that is not JSON reaches the record


@pytest.mark.parametrize("change, box", [
    (lambda xyz: xyz.fill(0.25), "0 x 0 x 0"),
    (lambda xyz: xyz[:, 1:].fill(0.0), "2 x 0 x 0"),
], ids=["one-point", "a-line"])
def test_a_mesh_without_real_geometry_fails(tmp_path, change, box):
    _, problems = _one(tmp_path, "a.glb", _moved(change))
    assert problems == [{"file": "a.glb", "check": "geometry", "text": (
        f"no real geometry: the bounding box is {box}, a point or a line")}]


def test_a_flat_sheet_is_geometry(tmp_path):
    def flatten(xyz):
        xyz[:, 2] = 0.0
    entry, problems = _one(tmp_path, "a.glb", _moved(flatten))
    assert problems == [] and entry["extent"] == [2.0, 2.0, 0.0]

    def stretch(xyz):
        xyz[0], xyz[1] = 3e38, -3e38
    # Large numbers stay numbers: the box is worked out in 64 bits.
    entry, problems = _one(tmp_path, "a.glb", _moved(stretch))
    assert problems == [] and entry["extent"] == pytest.approx([6e38] * 3, rel=1e-6)
    json.dumps(entry, allow_nan=False)


def test_meshes_as_other_writers_lay_them_out(tmp_path):
    xyz = np.array([[x, y, (x * y) % 3] for x in range(12) for y in range(10)], "<f4")
    faces = np.arange(360, dtype="<u2")
    packed = xyz.tobytes() + faces.tobytes()
    head = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": len(packed)}]}
    plain = {**head, "bufferViews": [{"buffer": 0, "byteLength": 1440},
                                     {"buffer": 0, "byteOffset": 1440, "byteLength": 720}],
             "accessors": [{"bufferView": 0, "componentType": 5126, "count": 120,
                            "type": "VEC3"},
                           {"bufferView": 1, "componentType": 5123, "count": 360,
                            "type": "SCALAR"}],
             "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}]}
    entry, problems = _one(tmp_path, "a.glb", _glb(plain, packed))
    assert problems == [] and (entry["vertices"], entry["faces"]) == (120, 120)
    assert entry["extent"] == [11.0, 9.0, 2.0]
    # Positions woven with normals: every vertex is 24 bytes apart.
    woven = np.zeros((120, 6), "<f4")
    woven[:, :3], woven[:, 3:] = xyz, 9e9
    stride = {**plain, "bufferViews": [{"buffer": 0, "byteLength": 2880, "byteStride": 24},
                                       {"buffer": 0, "byteOffset": 2880, "byteLength": 720}]}
    entry, problems = _one(tmp_path, "a.glb", _glb(stride, woven.tobytes() + faces.tobytes()))
    assert problems == [] and entry["extent"] == [11.0, 9.0, 2.0]
    # No index list: three positions make a face. A point cloud has none.
    bare = {**plain, "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}]}
    assert _one(tmp_path, "a.glb", _glb(bare, packed))[0]["faces"] == 40
    cloud = {**plain, "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "mode": 0}]}]}
    entry, problems = _one(tmp_path, "a.glb", _glb(cloud, packed))
    assert entry["faces"] == 0 and _ids(problems) == ["size"]
    strip = {**plain, "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "mode": 5}]}]}
    assert _one(tmp_path, "a.glb", _glb(strip, packed))[0]["faces"] == 118


@pytest.mark.parametrize("gltf, blob, said", [
    ({}, bytes(16), "no mesh with positions in the binary chunk"),
    ({"meshes": [{"primitives": [{"attributes": {}}]}]}, bytes(16),
     "no mesh with positions in the binary chunk"),
    ({"meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
      "accessors": [{"componentType": 5126, "count": 100, "type": "VEC3"}]}, bytes(16),
     "no mesh with positions in the binary chunk"),            # compressed: no buffer view
    ({"meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}], "bufferViews": [{}],
      "accessors": [{"bufferView": 0, "componentType": 5123, "count": 4, "type": "VEC3"}]},
     bytes(64), "the positions are not triples of 32-bit floats"),
    ({"meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}], "bufferViews": [{}],
      "accessors": [{"bufferView": 0, "componentType": 5126, "count": 10 ** 12,
                     "type": "VEC3"}]}, bytes(64), "the positions end past the binary chunk"),
    ({"meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
      "bufferViews": [{"byteOffset": 60}],
      "accessors": [{"bufferView": 0, "componentType": 5126, "count": 1, "type": "VEC3"}]},
     bytes(64), "the positions end past the binary chunk"),
])
def test_a_glb_file_without_a_readable_mesh(tmp_path, gltf, blob, said):
    _, problems = _one(tmp_path, "a.glb", _glb(gltf, blob))
    assert problems == [{"file": "a.glb", "check": "decode",
                         "text": f"could not be read: {said}"}]


def test_a_file_that_is_no_glb(tmp_path):
    whole = (FIX / "mesh_12.glb").read_bytes()
    json_len = struct.unpack_from("<I", whole, 12)[0]
    no_bin = bytearray(whole[:20 + json_len])
    no_bin[8:12] = struct.pack("<I", len(no_bin))
    for data, said in ((b"PK\x03\x04" + bytes(60), "not a GLB file"),
                       (b"glTF" + bytes(64), "no JSON chunk"),
                       (bytes(no_bin), "no binary chunk"),
                       (whole[:-1], "the file has 1291 bytes, its header says 1292"),
                       (_glb({}, b"")[:20] + b"[1]" + _glb({}, b"")[23:],
                        "the JSON chunk is not a glTF document")):
        _, problems = _one(tmp_path, "a.glb", data)
        assert problems == [{"file": "a.glb", "check": "decode",
                             "text": f"could not be read: {said}"}], said


# -- video ---------------------------------------------------------------------------
def _box(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(body), kind) + body


def _with_sound_track(data: bytes, size: int) -> bytes:
    """An MP4 file with a second track ahead of its video track: sound, of
    50 frames of `size` bytes each."""
    stbl = _box(b"stbl", _box(b"stsz", struct.pack(">III", 0, size, 50)))
    mdia = _box(b"mdia", _box(b"mdhd", bytes(20)) + _box(b"hdlr", bytes(8) + b"soun" + bytes(12))
                + _box(b"minf", stbl))
    trak = _box(b"trak", _box(b"tkhd", bytes(84)) + mdia)
    at = data.index(b"moov") - 4
    grown = struct.pack(">I", struct.unpack_from(">I", data, at)[0] + len(trak))
    return data[:at] + grown + data[at + 4:at + 8] + trak + data[at + 8:]


def test_a_moving_clip_passes(tmp_path):
    entry, problems = _one(tmp_path, "wf_video_00001_.mp4", FIX / "video_5f.mp4")
    assert problems == []
    assert entry == {"name": "wf_video_00001_.mp4", "type": "video", "bytes": 9079,
                     "sha256": hashlib.sha256((FIX / "video_5f.mp4").read_bytes()).hexdigest(),
                     "width": 640, "height": 352, "frames": 5, "seconds": 0.208,
                     "bytes_per_frame_mp": 5034}


def test_a_black_clip_fails(tmp_path):
    entry, problems = _one(tmp_path, "wf_video_00001_.mp4", FIX / "video_black_5f.mp4")
    assert (entry["frames"], entry["bytes_per_frame_mp"]) == (5, 352)
    assert problems == [{"file": "wf_video_00001_.mp4", "check": "blank", "text": (
        "blank or still: 79 bytes a frame after the first, and a 640x352 clip needs 213 "
        "(100, plus 500 for each megapixel)")}]


def _frame_sizes(data: bytes, sizes=None, fixed: int = 0) -> bytes:
    """An MP4 file whose video track lists other frame sizes: one for each
    frame, or `fixed`, the one size of them all."""
    at = data.index(b"stsz")
    count = struct.unpack_from(">II", data, at + 8)[1]
    table = struct.pack(f">I I {count}I", fixed, count, *(sizes or [0] * count))
    return data[:at + 8] + table + data[at + 8 + len(table):]


def _picture_size(data: bytes, width: int, height: int) -> bytes:
    """An MP4 file whose video track says another picture size."""
    at = data.index(b"tkhd")
    end = at - 4 + struct.unpack_from(">I", data, at - 4)[0]
    return data[:end - 8] + struct.pack(">II", width << 16, height << 16) + data[end:]


def test_the_first_frame_does_not_carry_a_blank_or_a_still_clip(tmp_path):
    """The first frame is stored whole and has the encoder's headers. At the
    small sizes a run is cut down to, its bytes alone lifted a black clip over
    the limit, and a clip of one picture repeated passed too. The frames after
    the first say whether anything is there and moves."""
    black, clip = ((FIX / name).read_bytes() for name in ("video_black_5f.mp4", "video_5f.mp4"))
    assert struct.unpack_from(">5I", black, black.index(b"stsz") + 16) == (1015, 90, 77, 75, 75)
    # The black clip at 256x256: 4,065 bytes a frame for each megapixel with
    # its first frame, which is over the limit, and 1,209 without it.
    assert round(1332 / 5 / (256 * 256 / 1e6)) == 4065
    entry, problems = _one(tmp_path, "a.mp4", _picture_size(black, 256, 256))
    assert (entry["width"], entry["height"], entry["bytes_per_frame_mp"]) == (256, 256, 1209)
    assert _ids(problems) == ["blank"]
    # One picture repeated: a first frame as large as the moving clip's, then nothing.
    still = _frame_sizes(clip, (3572, 60, 60, 60, 60))
    entry, problems = _one(tmp_path, "a.mp4", still)
    assert entry["bytes_per_frame_mp"] == 266 and _ids(problems) == ["blank"]
    # Two frames: the second decides.
    two = _frame_sizes(clip, (3572, 1711, 0, 0, 0))
    two = two[:two.index(b"stsz") + 12] + struct.pack(">I", 2) + two[two.index(b"stsz") + 16:]
    entry, problems = _one(tmp_path, "a.mp4", two)
    assert (entry["frames"], entry["bytes_per_frame_mp"]) == (2, 7595) and problems == []
    # Frames of one size, which the table then says once.
    for size, checks in ((500, []), (100, ["blank"])):
        entry, problems = _one(tmp_path, "a.mp4", _frame_sizes(clip, fixed=size))
        assert entry["bytes_per_frame_mp"] == round(size / 0.22528) and _ids(problems) == checks


# What the blank limit of a video rests on: `bytes_per_frame_mp` of 267 made-up
# clips, written as ComfyUI writes a clip (PyAV 14.2.0, h264, crf 18, yuv420p,
# MP4). (what the clip shows, whether it is blank or still, its frame counts,
# {(width, height): the number at each frame count}). "moving pattern" and
# "stripes, no noise" are one pattern, measured twice at three sizes.
MEASURED_CLIPS = (
    ("black", True, (5, 9, 17, 25, 49, 81), {
        (256, 256): (931, 988, 1017, 1026, 1038, 1042),
        (384, 256): (621, 659, 678, 684, 692, 695),
        (512, 288): (414, 441, 456, 461, 468, 471),
        (512, 512): (488, 515, 528, 533, 541, 545),
        (640, 352): (356, 377, 390, 394, 398, 399),
        (832, 480): (295, 312, 320, 323, 327, 328),
        (1280, 720): (209, 219, 225, 226, 229, 230)}),
    ("one grey", True, (5, 9, 17, 25, 49, 81), {
        (256, 256): (931, 988, 1017, 1026, 1038, 1042),
        (384, 256): (621, 659, 678, 684, 692, 695),
        (512, 288): (414, 441, 456, 461, 468, 471),
        (512, 512): (488, 515, 528, 533, 541, 545),
        (640, 352): (356, 377, 390, 394, 398, 399),
        (832, 480): (295, 312, 320, 323, 327, 328),
        (1280, 720): (209, 219, 225, 226, 229, 230)}),
    ("one frame of noise repeated", True, (5, 9, 17, 25, 49, 81), {
        (256, 256): (839, 900, 923, 931, 941, 945),
        (384, 256): (559, 601, 619, 623, 633, 638),
        (512, 288): (373, 403, 416, 420, 427, 429),
        (512, 512): (439, 469, 484, 490, 498, 501),
        (640, 352): (323, 344, 356, 360, 364, 366),
        (832, 480): (271, 289, 296, 298, 301, 302),
        (1280, 720): (190, 203, 207, 209, 212, 213)}),
    ("moving pattern", False, (5, 9, 17, 25, 49, 81), {
        (256, 256): (5520, 6077, 6213, 6320, 6543, 6747),
        (384, 256): (4204, 4754, 4961, 5176, 5373, 5492),
        (512, 288): (3454, 4090, 4314, 4481, 4485, 4489),
        (512, 512): (3129, 3516, 3593, 3640, 3876, 3950),
        (640, 352): (2507, 3023, 3028, 3253, 3366, 3666),
        (832, 480): (1934, 2065, 2471, 2503, 2381, 2538),
        (1280, 720): (1409, 1613, 1774, 1932, 1852, 1956)}),
    ("stripes, no noise", False, (5, 17, 81), {
        (512, 288): (3454, 4314, 4489), (832, 480): (1934, 2471, 2538),
        (1280, 720): (1409, 1774, 1956)}),
    ("stripes, noise sigma 1", False, (5, 17, 81), {
        (512, 288): (3650, 3888, 3732), (832, 480): (2570, 2659, 2733),
        (1280, 720): (1926, 2230, 2344)}),
    ("stripes, noise sigma 2", False, (5, 17, 81), {
        (512, 288): (5291, 4998, 4843), (832, 480): (4131, 4075, 3967),
        (1280, 720): (3290, 3641, 3609)}),
    ("stripes, noise sigma 4", False, (5, 17, 81), {
        (512, 288): (25360, 28411, 34062), (832, 480): (24773, 27053, 32917),
        (1280, 720): (24176, 26792, 32912)}),
    ("soft scene, pan 1 px a frame", False, (5, 17, 81), {
        (512, 288): (2667, 1577, 1618), (832, 480): (1591, 1374, 1269),
        (1280, 720): (1962, 1396, 1162)}),
    ("soft scene, pan 4 px a frame", False, (5, 17, 81), {
        (512, 288): (1851, 1989, 1360), (832, 480): (1747, 1534, 1046),
        (1280, 720): (1493, 1359, 1054)}),
    ("soft scene, pan 4 px, noise sigma 1", False, (5, 17, 81), {
        (512, 288): (3045, 3117, 2819), (832, 480): (3437, 3777, 3518),
        (1280, 720): (3985, 4143, 3627)}),
    ("soft scene, pan 4 px, noise sigma 2", False, (5, 17, 81), {
        (512, 288): (5790, 6736, 6273), (832, 480): (6329, 6484, 7235),
        (1280, 720): (6609, 8031, 9466)}),
    ("soft scene, still, noise sigma 2", False, (5, 17, 81), {
        (512, 288): (3374, 4685, 5565), (832, 480): (4444, 5888, 6620),
        (1280, 720): (5687, 8593, 9756)}),
    ("soft scene, still, noise sigma 4", False, (5, 17, 81), {
        (512, 288): (85517, 81490, 78723), (832, 480): (91590, 87519, 95643),
        (1280, 720): (105642, 112365, 106066)}),
    ("soft scene that gets brighter", False, (5, 17, 81), {
        (512, 288): (4483, 9588, 8389), (832, 480): (4572, 8703, 8509),
        (1280, 720): (9271, 9053, 7409)}),
)


def test_the_blank_limit_of_a_video_has_a_fixed_part_for_each_frame(tmp_path):
    """A frame that holds nothing still costs some bytes, whatever its size:
    a black clip at 256x256 measures up to 1,042 bytes a frame for each
    megapixel, and a soft picture that pans at 832x480 measures 1,046. The
    limit was 1,500 for each megapixel. No such limit tells the two apart, and
    that one failed 11 of the 141 moving clips of MEASURED_CLIPS as "blank or
    still". Held against 100 bytes plus 500 for each megapixel, every blank or
    still clip has at most 0.62 of what a frame needs and every moving one at
    least 1.15.

    The clips are made up: flat colours, one noise picture repeated, moving
    stripes, a blurred noise picture that pans or gets brighter, with and
    without grain. No clip of a model is among them. The limit is to be
    checked against real ones before the first video workflow gets the smoke
    level (the README says so too)."""
    lim = smoke.LIMITS["video"]
    clip = (FIX / "video_5f.mp4").read_bytes()
    shares, per_megapixel = {True: [], False: []}, {True: [], False: []}
    for kind, still, counts, sizes in MEASURED_CLIPS:
        for (width, height), numbers in sizes.items():
            megapixels = width * height / 1e6
            need = lim["min_frame_bytes"] + lim["min_frame_bytes_per_mp"] * megapixels
            assert len(numbers) == len(counts), kind
            for number in numbers:
                shares[still].append(number * megapixels / need)
                per_megapixel[still].append(number)
                # The check itself, on a clip of this size whose frames after
                # the first have these bytes.
                data = _frame_sizes(_picture_size(clip, width, height),
                                    fixed=round(number * megapixels))
                entry, problems = _one(tmp_path, "a.mp4", data)
                assert abs(entry["bytes_per_frame_mp"] - number) <= 8, (kind, width, number)
                assert _ids(problems) == (["blank"] if still else []), (kind, width, number)
    assert (len(shares[True]), len(shares[False])) == (126, 141)
    assert 0.61 < max(shares[True]) < 0.62 and 1.15 < min(shares[False]) < 1.16
    # What a limit for each megapixel alone has to tell apart, and what 1,500 did.
    assert (max(per_megapixel[True]), min(per_megapixel[False])) == (1042, 1046)
    assert sum(number < 1500 for number in per_megapixel[False]) == 11


def test_a_sound_track_cannot_hide_a_blank_picture(tmp_path):
    """Only the bytes of the video track count."""
    black = (FIX / "video_black_5f.mp4").read_bytes()
    loud = _with_sound_track(black, 100000)
    entry, problems = _one(tmp_path, "a.mp4", loud)
    assert _ids(problems) == ["blank"]
    assert (entry["frames"], entry["width"], entry["bytes_per_frame_mp"]) == (5, 640, 352)
    # A file with sound alone has no picture at all.
    at = loud.index(b"vide")
    _, problems = _one(tmp_path, "a.mp4", loud[:at] + b"soun" + loud[at + 4:])
    assert problems == [{"file": "a.mp4", "check": "decode",
                         "text": "could not be read: no video track"}]


def test_a_clip_is_held_to_the_expected_frames(tmp_path):
    clip = FIX / "video_5f.mp4"
    assert _one(tmp_path, "a.mp4", clip, {"frames": 5})[1] == []
    # The expected size is for PNG files: a clip is not held to it.
    assert _one(tmp_path, "a.mp4", clip, {"frames": 5, "width": 1024, "height": 1024})[1] == []
    _, problems = _one(tmp_path, "a.mp4", clip, {"frames": 25})
    assert problems == [{"file": "a.mp4", "check": "expect", "text": "5 frames, 25 expected"}]


def test_a_clip_of_one_frame_fails(tmp_path):
    data = bytearray((FIX / "video_5f.mp4").read_bytes())
    at = data.index(b"stsz")
    assert struct.unpack_from(">II", data, at + 8) == (0, 5)   # five sizes follow
    data[at + 12:at + 16] = struct.pack(">I", 1)
    entry, problems = _one(tmp_path, "a.mp4", bytes(data))
    assert entry["frames"] == 1
    assert problems == [{"file": "a.mp4", "check": "frames",
                         "text": "1 frame, at least 2 expected"}]


def test_a_clip_with_a_tiny_frame_fails(tmp_path):
    data = bytearray((FIX / "video_5f.mp4").read_bytes())
    at = data.index(b"tkhd")
    end = at - 4 + struct.unpack_from(">I", data, at - 4)[0]
    assert struct.unpack_from(">II", data, end - 8) == (640 << 16, 352 << 16)
    data[end - 8:end] = struct.pack(">II", 48 << 16, 32 << 16)
    entry, problems = _one(tmp_path, "a.mp4", bytes(data))
    assert (entry["width"], entry["height"]) == (48, 32)
    assert problems == [{"file": "a.mp4", "check": "size",
                         "text": "48x32 px: each side must have at least 64 px"}]


def test_a_file_that_is_no_mp4(tmp_path):
    clip = (FIX / "video_5f.mp4").read_bytes()
    moov = clip.index(b"moov")
    stsz = clip.index(b"stsz")
    huge = clip[:stsz + 12] + struct.pack(">I", 2 ** 32 - 1) + clip[stsz + 16:]
    for data, said in ((bytes(64), "no ftyp box"),
                       (b"RIFF" + bytes(60), "a box is cut short"),
                       (_box(b"free", bytes(8)), "no ftyp box"),
                       (clip[:moov - 4], "no moov box"),
                       (clip[:moov] + b"mooo" + clip[moov + 4:], "no moov box"),
                       (clip[:-1], "the mdat box is cut short"),
                       (huge, "a box is too short for its fields")):
        _, problems = _one(tmp_path, "a.mp4", data)
        assert problems == [{"file": "a.mp4", "check": "decode",
                             "text": f"could not be read: {said}"}], said


# -- a file that cannot be read fails, and never raises ------------------------------
CUT = {"png": _png(_picture()), **{p.suffix[1:] + ":" + p.stem: p.read_bytes()
                                   for p in sorted(FIX.iterdir()) if p.suffix != ".md"}}


@pytest.mark.parametrize("key", sorted(CUT))
def test_the_first_half_of_a_file_is_a_decode_problem(tmp_path, key):
    data = CUT[key]
    name = "wf_00001_." + key.split(":")[0]
    res = _checked(tmp_path, {name: data[:len(data) // 2]})
    assert _ids(res["problems"]) == ["decode"] and res["problems"][0]["file"] == name
    assert res["problems"][0]["text"].startswith("could not be read: ")
    # Nothing was measured, and the entry is still there with what the worker said.
    assert list(res["files"][0]) == ["name", "type", "bytes", "sha256"]
    assert res["files"][0]["bytes"] == len(data) // 2


@pytest.mark.parametrize("key", sorted(k for k in CUT if k != "png"))
@pytest.mark.parametrize("keep", [0.1, 0.9, -1], ids=["a-tenth", "nine-tenths", "all-but-a-byte"])
def test_a_cut_file_never_passes(tmp_path, key, keep):
    data = CUT[key]
    cut = data[:keep] if keep < 0 else data[:int(len(data) * keep)]
    res = _checked(tmp_path, {"wf_00001_." + key.split(":")[0]: cut})
    assert _ids(res["problems"]) == ["decode"]


@pytest.mark.parametrize("ext", ["png", "flac", "glb", "mp4"])
def test_bytes_that_are_no_file_of_the_type_fail_without_an_exception(tmp_path, ext):
    noise = bytes((i * 131 + 7) % 256 for i in range(5000))
    for data in (noise, b"\x00", b"\xff" * 3000, TEXT.encode()):
        res = _checked(tmp_path, {f"a.{ext}": data})
        assert _ids(res["problems"]) == ["decode"], data[:8]
        assert res["problems"][0]["text"].startswith("could not be read: ")
        json.dumps(res, allow_nan=False)


def test_a_problem_never_names_a_folder_of_the_runner(tmp_path):
    out, rec = _run(tmp_path, {"wf_00001_.png": _png(_picture()), "wf_00001.flac": b"x",
                               "wf_00002_.png": b"not a picture"})
    (out / "wf_00001_.png").unlink()   # listed as sent, and not there
    res = smoke.check_outputs(out, rec, {}, None, "ok")
    assert res["problems"] == [
        {"file": "wf_00001.flac", "check": "decode",
         "text": "could not be read: not a FLAC file with a STREAMINFO block"},
        {"file": "wf_00001_.png", "check": "decode",
         "text": "could not be read: FileNotFoundError: No such file or directory"},
        {"file": "wf_00002_.png", "check": "decode",
         "text": "could not be read: not an image file"}]
    # The same texts on every runner: no path, no address of an object.
    assert tmp_path.name not in json.dumps(res) and "0x" not in json.dumps(res)
    # Without an outputs folder (a run record that never got one) nothing can be opened.
    res = smoke.check_outputs(None, rec, {}, None, "ok")
    assert {p["text"] for p in res["problems"]} == {
        "could not be read: the run has no outputs folder"}


def test_a_name_that_leaves_the_outputs_folder_is_not_opened(tmp_path):
    out, rec = _run(tmp_path, {"wf_00001_.png": _png(_picture())})
    (tmp_path / "secret.png").write_bytes(_png(_picture()))
    (tmp_path / "secret.txt").write_bytes(TEXT.encode())
    for name in ("../secret.png", "../secret.txt", "sub/../../secret.png",
                 str(tmp_path / "secret.png")):
        rec["outputs"] = [{"filename": name, "bytes": 5000, "sha256": "ab" * 32}]
        res = smoke.check_outputs(out, rec, {}, None, "ok")
        assert res["problems"] == [{"file": name, "check": "decode", "text": (
            "could not be read: its name leaves the outputs folder")}]
        assert list(res["files"][0]) == ["name", "type", "bytes", "sha256"]


def test_a_name_that_leaves_the_outputs_folder_never_passes_on_its_size(tmp_path):
    """The runner writes such a file nowhere and lists it as not sent
    (submit_and_poll._write_outputs). A file that was not sent passes on its
    size, so the run passed, with a note that blamed the worker's size limit.
    Only a worker that was tampered with names such a file."""
    out, rec = _run(tmp_path, {"wf_00001_.png": _png(_picture())})
    good = rec["outputs"][0]
    said = {"check": "decode", "text": "could not be read: its name leaves the outputs folder"}
    for name, kind in (("../escape.png", "image"), ("../../escape2.png", "image"),
                       (str(tmp_path / "abs_escape.png"), "image"), ("sub/../../x.flac", "audio"),
                       ("../run.json", None), ("..", None), (".", None)):
        rec["outputs"] = [good, {"filename": name, "bytes": 5000, "sha256": "ab" * 32,
                                 "truncated": True}]
        res = smoke.check_outputs(out, rec, {}, None, "ok")
        assert res["problems"] == [{"file": name, **said}], name
        assert res["notes"] == []
        # Listed with what the worker said of it, and as a file that is not there.
        assert {"name": name, "type": kind, "bytes": 5000, "sha256": "ab" * 32,
                "sent": False} in res["files"]
        assert run_regression.smoke_verdict("ok", res["problems"]) == "fail"
    # A name that stays inside is what it was: checked by its size when it was not sent.
    for name in ("sub/wf_00002_.png", "sub/../wf_00002_.png", "./wf_00002_.png",
                 str(out / "wf_00002_.png")):
        rec["outputs"] = [good, {"filename": name, "bytes": 5000, "sha256": "ab" * 32,
                                 "truncated": True}]
        res = smoke.check_outputs(out, rec, {}, None, "ok")
        assert res["problems"] == [] and len(res["notes"]) == 1, name
    # Without an outputs folder nothing can be judged: the size rule stays.
    assert smoke.check_outputs(None, {"outputs": [{"filename": "../x.png", "bytes": 5000,
                                                   "truncated": True}]}, {}, None, "ok")[
        "problems"] == []


# -- which files a run must have written ---------------------------------------------
def _node(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs}


MESH_WORKFLOW = {
    "1": _node("EmptyImage", width=64),
    "render": _node("RenderMesh", mesh=["hy", 0]),
    "9": _node("SaveImage", filename_prefix="wf", images=["render", 0]),
    "10": _node("SaveGLB", filename_prefix="wf_mesh", mesh=["hy", 0]),
    "11": _node("PreviewImage", images=["1", 0]),
}


def test_expected_outputs_come_from_the_save_nodes():
    assert smoke.expected_outputs(MESH_WORKFLOW) == [
        {"node": "9", "class": "SaveImage", "prefix": "wf", "extensions": [".png"]},
        {"node": "10", "class": "SaveGLB", "prefix": "wf_mesh",
         "extensions": [".glb", ".gltf", ".obj", ".fbx", ".stl", ".usdz", ".ply", ".splat",
                        ".spz", ".ksplat"]}]
    kinds = {"1": _node("SaveText", filename_prefix="t", format="txt"),
             "2": _node("SaveAudioAdvanced", filename_prefix="a", format="flac"),
             "3": _node("SaveAudio", filename_prefix="b"),
             "4": _node("SaveVideo", filename_prefix="v"),
             "5": _node("VHS_VideoCombine", filename_prefix="vhs")}   # nothing is expected
    assert [(e["class"], e["prefix"], e["extensions"]) for e in smoke.expected_outputs(kinds)] \
        == [("SaveText", "t", [".txt"]), ("SaveAudioAdvanced", "a", [".flac"]),
            ("SaveAudio", "b", [".flac"]), ("SaveVideo", "v", [".mp4"])]
    for odd in (None, [], "x", {"1": None, "2": {"inputs": {}},
                                "3": {"class_type": ["SaveImage"]}}):
        assert smoke.expected_outputs(odd) == []


def test_the_prefix_is_what_comfyui_would_write():
    def prefix(value):
        (found,) = smoke.expected_outputs({"1": _node("SaveImage", filename_prefix=value)})
        return found["prefix"]
    assert prefix("wf") == "wf" and prefix("audio/wf") == "audio/wf"
    assert prefix("audio\\wf") == "audio/wf" and prefix("./wf") == "wf"
    # A name only ComfyUI can fill in, or none at all: any file name will do.
    for unknown in ("%date:yyyy%/wf", "wf_%width%", "%year%/wf", ["4", 0], None, "", 7, "../wf",
                    "/abs/wf", "C:/x/wf"):
        assert prefix(unknown) is None, unknown
    assert smoke.expected_outputs({"1": {"class_type": "SaveImage"}})[0]["prefix"] is None


def test_a_run_with_every_expected_file_passes(tmp_path):
    png, glb = _png(_picture()), (FIX / "mesh_300.glb").read_bytes()
    res = _checked(tmp_path, {"wf_00001_.png": png, "wf_mesh_00001_.glb": glb}, MESH_WORKFLOW)
    assert list(res) == ["files", "problems", "notes"]
    assert res == {
        "files": [
            {"name": "wf_00001_.png", "type": "image", "bytes": len(png),
             "sha256": hashlib.sha256(png).hexdigest(), "width": 640, "height": 352,
             "std": res["files"][0]["std"], "flat_share": res["files"][0]["flat_share"]},
            {"name": "wf_mesh_00001_.glb", "type": "mesh", "bytes": 11164,
             "sha256": hashlib.sha256(glb).hexdigest(), "vertices": 300, "faces": 560,
             "extent": [2.0, 2.0, 2.0]}],
        "problems": [], "notes": []}
    assert json.loads(json.dumps(res, allow_nan=False)) == res


def test_a_missing_file_is_a_problem(tmp_path):
    res = _checked(tmp_path, {"wf_00001_.png": _png(_picture())}, MESH_WORKFLOW)
    assert res["problems"] == [{"file": None, "check": "missing",
                                "text": "SaveGLB wrote no 3D file with the prefix wf_mesh"}]
    assert list(res["problems"][0]) == ["file", "check", "text"]
    assert [f["name"] for f in res["files"]] == ["wf_00001_.png"]
    res = _checked(tmp_path / "b", {"wf_mesh_00001_.glb": FIX / "mesh_300.glb"}, MESH_WORKFLOW)
    assert res["problems"] == [{"file": None, "check": "missing",
                                "text": "SaveImage wrote no .png file with the prefix wf"}]
    # A file under another name is not the node's file: the prefix ends at an underscore.
    for other in ("other_00001_.glb", "wf_meshes_00001_.glb", "wf_mesh.glb", "wf_00001_.glb"):
        res = _checked(tmp_path / other, {"wf_00001_.png": _png(_picture()),
                                          other: FIX / "mesh_300.glb"}, MESH_WORKFLOW)
        assert _ids(res["problems"]) == ["missing"], other


def test_a_file_of_another_node_is_not_the_save_nodes_file(tmp_path):
    """A second sink is prefixed `<workflow id>_...`, so its files start like
    those of the main one. VHS_VideoCombine writes its first frame as a PNG
    next to its clip: that picture is not the file SaveImage owes."""
    roundtrip = {"1": _node("SaveImage", filename_prefix="wf", images=["0", 0]),
                 "2": _node("VHS_VideoCombine", filename_prefix="wf_gif", images=["0", 0])}
    png = _png(_picture())
    combined = {"wf_gif_00001.gif": b"GIF89a", "wf_gif_00001.png": png}
    assert _checked(tmp_path / "a", {**combined, "wf_00001_.png": png},
                    roundtrip)["problems"] == []
    res = _checked(tmp_path / "b", combined, roundtrip)
    assert res["problems"] == [{"file": None, "check": "missing",
                                "text": "SaveImage wrote no .png file with the prefix wf"}]
    assert [f["name"] for f in res["files"]] == ["wf_gif_00001.gif", "wf_gif_00001.png"]
    # Between the prefix and the extension stands the counter ComfyUI writes: five
    # digits or more, with or without an underscore after them.
    one = {"1": _node("SaveImage", filename_prefix="wf")}
    for name, counts in (("wf_00001_.png", True), ("wf_00001.png", True), ("wf_123456_.png", True),
                         ("wf_00001_.PNG", True), ("wf_0001_.png", False),
                         ("wf_00001_x.png", False), ("wf_00001__.png", False),
                         ("wf_00001_.tar.png", False), ("wf_a_00001_.png", False),
                         ("wf_00001_/x.png", False), ("wf__00001_.png", False)):
        res = _checked(tmp_path / name.replace("/", "-"), {name: png}, one)
        assert (res["problems"] == []) is counts, name
    # The same in a folder, and for the nodes that leave the underscore out.
    assert smoke._fits("3d/wf_mesh_00001_.glb", "3d/wf_mesh", (".glb",))
    assert smoke._fits("wf_00012.flac", "wf", (".flac",))
    assert smoke._fits("x.flac", None, (".flac",))
    assert not smoke._fits("3d/wf_mesh_00001_.glb", "wf_mesh", (".glb",))


def test_a_3d_file_in_another_format_counts_and_is_not_opened(tmp_path):
    """SaveGLB fed by a 3D file keeps that file's own extension."""
    for ext in ("obj", "OBJ", "gltf", "ply", "usdz"):
        res = _checked(tmp_path / ext, {"wf_00001_.png": _png(_picture()),
                                        f"wf_mesh_00001_.{ext}": b"v 0 0 0\n"}, MESH_WORKFLOW)
        assert res["problems"] == [] and res["notes"] == []
        assert res["files"][1] == {"name": f"wf_mesh_00001_.{ext}", "type": None, "bytes": 8,
                                   "sha256": hashlib.sha256(b"v 0 0 0\n").hexdigest()}
    # Not a 3D file: the node's file is still missing, and the file is listed.
    res = _checked(tmp_path / "gif", {"wf_00001_.png": _png(_picture()),
                                      "wf_mesh_00001_.gif": b"GIF89a"}, MESH_WORKFLOW)
    assert _ids(res["problems"]) == ["missing"] and res["files"][1]["type"] is None


def test_a_file_that_was_not_sent_passes_on_its_size_with_a_note(tmp_path):
    files = {"wf_00001_.png": _png(_picture()), "wf_mesh_00001_.glb": bytes(9 * MIB)}
    res = _checked(tmp_path, files, MESH_WORKFLOW, unsent=("wf_mesh_00001_.glb",))
    assert res["problems"] == []
    assert res["notes"] == ["wf_mesh_00001_.glb was not sent (over the worker's size limit): "
                            "checked by size only"]
    assert res["files"][1] == {"name": "wf_mesh_00001_.glb", "type": "mesh", "bytes": 9 * MIB,
                               "sha256": hashlib.sha256(bytes(9 * MIB)).hexdigest(),
                               "sent": False}
    assert "sent" not in res["files"][0]
    # From 1,024 bytes on. A smaller one cannot pass unseen: an empty file is listed like that.
    for size, passes in ((1024, True), (1023, False), (0, False)):
        res = _checked(tmp_path / str(size), {**files, "wf_mesh_00001_.glb": bytes(size)},
                       MESH_WORKFLOW, unsent=("wf_mesh_00001_.glb",))
        assert (res["problems"] == []) is passes
        if not passes:
            assert res["notes"] == [] and res["problems"] == [{
                "file": "wf_mesh_00001_.glb", "check": "size", "text": (
                    "not sent, and too small to pass by its size: "
                    f"{size} bytes, under 1024")}]
    # A file of a type that is not checked is listed, with no note.
    res = _checked(tmp_path / "gif", {**files, "wf_mesh_00001_.glb": FIX / "mesh_300.glb",
                                      "extra.gif": bytes(5000)}, MESH_WORKFLOW,
                   unsent=("extra.gif",))
    assert res["problems"] == [] and res["notes"] == []
    assert res["files"][0] == {"name": "extra.gif", "type": None, "bytes": 5000,
                               "sha256": hashlib.sha256(bytes(5000)).hexdigest(), "sent": False}
    # A text is held to its size limit unseen: a model that ran away writes megabytes,
    # the worker does not send them, and the size alone must not pass them.
    texts = {"1": _node("SaveText", filename_prefix="wf", format="txt")}
    for size, passes in ((MIB, True), (MIB + 1, False), (20 * MIB, False)):
        out, rec = _run(tmp_path / f"text{size}", {"wf_00001.txt": b""}, ("wf_00001.txt",))
        rec["outputs"][0]["bytes"] = size
        res = smoke.check_outputs(out, rec, texts, None, "ok")
        assert res["files"][0]["sent"] is False and "chars" not in res["files"][0]
        if passes:
            assert res["problems"] == [] and len(res["notes"]) == 1
        else:
            assert res["notes"] == [] and res["problems"] == [{
                "file": "wf_00001.txt", "check": "size",
                "text": f"{size} bytes: a text output may have at most {MIB}"}]


def test_save_nodes_that_share_a_prefix_must_write_a_file_each(tmp_path):
    """ComfyUI runs the valid branches of a prompt: one of two pictures can be missing."""
    two = {"1": _node("SaveImage", filename_prefix="wf"),
           "2": _node("SaveImage", filename_prefix="wf")}
    png = _png(_picture())
    assert _checked(tmp_path / "a", {"wf_00001_.png": png, "wf_00002_.png": png},
                    two)["problems"] == []
    res = _checked(tmp_path / "b", {"wf_00001_.png": png}, two)
    assert res["problems"] == [{"file": None, "check": "missing", "text": (
        "2 SaveImage nodes wrote 1 .png file with the prefix wf, 2 expected")}]
    res = _checked(tmp_path / "c", {"note.gif": b"x"}, two)
    assert res["problems"][0]["text"] == ("2 SaveImage nodes wrote 0 .png files with the "
                                          "prefix wf, 2 expected")
    both = {"1": _node("SaveAudio", filename_prefix="wf"),
            "2": _node("SaveAudioAdvanced", filename_prefix="wf")}
    res = _checked(tmp_path / "d", {"wf_00001.flac": FIX / "audio_1s.flac"}, both)
    assert res["problems"][0]["text"] == ("2 SaveAudio and SaveAudioAdvanced nodes wrote 1 "
                                          ".flac file with the prefix wf, 2 expected")


def test_a_file_counts_for_the_longest_prefix_it_starts_with(tmp_path):
    texts = {"1": _node("SaveText", filename_prefix="wf", format="txt"),
             "2": _node("SaveText", filename_prefix="wf_thinking", format="txt")}
    both = {"wf_00001.txt": TEXT.encode(), "wf_thinking_00001.txt": TEXT.encode()}
    assert _checked(tmp_path / "a", both, texts)["problems"] == []
    res = _checked(tmp_path / "b", {"wf_thinking_00001.txt": TEXT.encode()}, texts)
    assert res["problems"] == [{"file": None, "check": "missing",
                                "text": "SaveText wrote no .txt file with the prefix wf"}]
    res = _checked(tmp_path / "c", {"wf_00001.txt": TEXT.encode()}, texts)
    assert res["problems"][0]["text"] == "SaveText wrote no .txt file with the prefix wf_thinking"


def test_files_in_a_folder_and_names_that_cannot_be_known(tmp_path):
    flac = FIX / "audio_1s.flac"
    nested = {"1": _node("SaveAudioAdvanced", filename_prefix="audio/wf", format="flac")}
    assert _checked(tmp_path / "a", {"audio/wf_00001.flac": flac}, nested)["problems"] == []
    res = _checked(tmp_path / "b", {"wf_00001.flac": flac}, nested)
    assert res["problems"][0]["text"] == ("SaveAudioAdvanced wrote no .flac file with the "
                                          "prefix audio/wf")
    dated = {"1": _node("SaveAudioAdvanced", filename_prefix="%date:yyyy%/wf")}
    assert _checked(tmp_path / "c", {"2026/wf_00001.flac": flac}, dated)["problems"] == []
    res = _checked(tmp_path / "d", {"wf_00001_.png": _png(_picture())}, dated)
    assert res["problems"] == [{"file": None, "check": "missing",
                                "text": "SaveAudioAdvanced wrote no .flac file"}]


# -- the smoke object of a run -------------------------------------------------------
def test_a_failing_run_differs_only_in_its_problems(tmp_path):
    workflow = {"106": _node("SaveAudioAdvanced", filename_prefix="ace_step_15_t2a",
                             format="flac")}
    res = _checked(tmp_path, {"ace_step_15_t2a_00001.flac": FIX / "audio_silent_1s.flac"},
                   workflow, {"seconds": 1})
    assert res == {
        "files": [{"name": "ace_step_15_t2a_00001.flac", "type": "audio", "bytes": 8428,
                   "sha256": "34e1d4bad9c7c9734a1b6d6e7749be8c870de5b78c5bf6cda23fcdcfc81a2313",
                   "sample_rate": 44100, "channels": 2, "bits": 16, "seconds": 1.0,
                   "bytes_per_second": 142, "pcm_md5": "d2b120199019b639d5a7e2b3463e9c97"}],
        "problems": [{"file": "ace_step_15_t2a_00001.flac", "check": "silent",
                      "text": "silent or constant: 142 bytes of audio a second, under 4000"}],
        "notes": []}


def test_only_a_run_that_finished_is_checked(tmp_path):
    out, rec = _run(tmp_path, {"wf_00001_.png": _png(_flat(64, 0))})
    empty = {"files": [], "problems": [], "notes": []}
    # Finished and wrote nothing: the first thing a smoke check exists to catch.
    assert smoke.check_outputs(out, {"status": "no_outputs", "outputs": []}, MESH_WORKFLOW, None,
                               "no_outputs") == {**empty, "problems": [{
                                   "file": None, "check": "missing",
                                   "text": "the workflow finished and wrote no file"}]}
    # Did not finish: the worker status is the result, and nothing is opened.
    for status in ("execution_error", "timeout", "infra_error", "missing_model", None, ""):
        assert smoke.check_outputs(out, rec, MESH_WORKFLOW, None, status) == empty
    assert smoke.check_outputs(None, None, None, None, "infra_error") == empty
    # Finished, with a record that lists nothing: every save node's file is missing.
    for record in ({"outputs": []}, {"outputs": None}, {}, None, {"outputs": [None, {}, 7]}):
        res = smoke.check_outputs(out, record, MESH_WORKFLOW, None, "ok")
        assert _ids(res["problems"]) == ["missing", "missing"] and res["files"] == []


def test_files_are_listed_by_name_and_the_lists_are_bounded(tmp_path):
    black = _png(_flat(64, 0))
    files = {f"wf_{i:05}_.png": black for i in range(60, 0, -1)}
    out, rec = _run(tmp_path, files)
    rec["outputs"].reverse()   # whatever order the record has
    res = smoke.check_outputs(out, rec, {"1": _node("SaveImage", filename_prefix="wf")}, None,
                              "ok")
    assert [f["name"] for f in res["files"]] == [f"wf_{i:05}_.png" for i in range(1, 51)]
    assert [p["file"] for p in res["problems"]] == [f"wf_{i:05}_.png" for i in range(1, 21)]
    assert len(res["problems"]) == 20 and set(_ids(res["problems"])) == {"blank"}
    assert res["notes"] == ["60 output files: only the first 50 are listed",
                            "60 problems: only the first 20 are listed"]
    # A missing file is named first, so the bound never drops it.
    res = _checked(tmp_path, files, MESH_WORKFLOW)
    assert _ids(res["problems"])[:2] == ["missing", "blank"] and len(res["problems"]) == 20


def test_names_from_the_worker_are_data(tmp_path):
    """They are kept as they came: whoever shows them escapes them."""
    res = _checked(tmp_path, {"sub dir/x y.txt": b"", "wf_(b)`!_00001_.png": _png(_flat(64, 0))})
    assert [f["name"] for f in res["files"]] == ["sub dir/x y.txt", "wf_(b)`!_00001_.png"]
    assert [p["file"] for p in res["problems"]] == ["sub dir/x y.txt", "wf_(b)`!_00001_.png"]
    # A name no file system would take arrives the same way, in a record alone.
    name = "wf_<b>`|_00001_.png"
    out, rec = _run(tmp_path / "b", {})
    rec["outputs"] = [{"filename": name, "bytes": 5000, "sha256": "<img src=x>",
                       "truncated": True}]
    res = smoke.check_outputs(out, rec, {}, None, "ok")
    assert res["files"] == [{"name": name, "type": "image", "bytes": 5000,
                             "sha256": "<img src=x>", "sent": False}]
    assert res["notes"] == [f"{name} was not sent (over the worker's size limit): checked by "
                            "size only"]


# -- the sha256 of a run, and what it publishes --------------------------------------
def _rec(*files):
    return {"outputs": [{"filename": name, "sha256": sha} for name, sha in files]}


def test_main_sha_of_an_image_run_is_the_reference_level_rule():
    for rec in (_rec(("wf_00002_.png", "b" * 64), ("wf_00001_.png", "a" * 64),
                     ("wf_mesh_00001_.glb", "c" * 64)),
                _rec(("b/x.png", "b" * 64), ("a.PNG", "a" * 64), ("z.txt", "c" * 64)),
                _rec(("wf_00001.flac", "a" * 64)), {"outputs": []}, {"outputs": None}, {}):
        assert smoke.main_sha(rec, "wf", "image") == run_regression.first_output_sha(rec)
    assert smoke.main_sha(_rec(("wf_00002_.png", "b" * 64), ("wf_00001_.png", "a" * 64)), "wf",
                          "image") == "a" * 64
    assert smoke.main_sha(None, "wf", "image") is None


def test_main_sha_of_a_text_run_is_the_text_rule():
    for rec in (_rec(("wf_00001.txt", "a" * 64)),
                _rec(("wf_00001.txt", "a" * 64), ("wf_thinking_00001.txt", "b" * 64)),
                _rec(("wf_00001_.png", "a" * 64)), {}, None):
        assert smoke.main_sha(rec, "wf", "text") == output_kinds.text_sha(rec)
    assert smoke.main_sha(_rec(("wf_00001.txt", "a" * 64)), "wf", "text") == "a" * 64


def test_main_sha_of_an_audio_run_is_its_first_flac_named_after_the_workflow():
    rec = _rec(("other_00001.flac", "0" * 64), ("wf_00002.flac", "b" * 64),
               ("wf_00001.FLAC", "a" * 64), ("wf_00001_.png", "c" * 64))
    assert smoke.main_sha(rec, "wf", "audio") == "a" * 64
    assert smoke.main_sha(rec, "other", "audio") == "0" * 64
    assert smoke.main_sha(rec, "none", "audio") is None
    assert smoke.main_sha(_rec(("wf_00001.flac", None)), "wf", "audio") is None
    # A kind without a rule has no main output.
    assert smoke.main_sha(rec, "wf", "mesh") is None


def _file(name, size, sha):
    return {"name": name, "type": None, "bytes": size, "sha256": sha}


def test_a_large_output_is_published_only_when_it_repeats():
    """git stores equal files once: an output that never repeats is never stored."""
    first = [_file("wf_00001_.png", 2 * MIB, "a" * 64), _file("wf_00002_.png", MIB // 2, "b" * 64),
             _file("wf_mesh_00001_.glb", MIB + 1, "c" * 64), _file("wf_00003_.png", MIB, "d" * 64)]
    # First run: nothing to compare with, the two files over 1 MiB are held back.
    for nothing in (None, [], "x", [None, 7]):
        assert smoke.unpublished(first, nothing) == {"wf_00001_.png", "wf_mesh_00001_.glb"}
    # Second run with the same bytes: everything is published.
    assert smoke.unpublished(first, first) == set()
    # A file that changed is held back again, the one that repeats is not.
    second = [dict(first[0], sha256="e" * 64), *first[1:]]
    assert smoke.unpublished(second, first) == {"wf_00001_.png"}
    # The name counts: the same bytes under another name did not repeat.
    renamed = [dict(first[0], name="wf_00009_.png")]
    assert smoke.unpublished(renamed, first) == {"wf_00009_.png"}
    # A file without a sha256 never repeats, and one without a size is taken as large.
    assert smoke.unpublished([_file("a.glb", 2 * MIB, None)], [_file("a.glb", 2 * MIB, None)]) \
        == {"a.glb"}
    assert smoke.unpublished([_file("a.glb", None, "a" * 64)], None) == {"a.glb"}
    assert smoke.unpublished([], first) == set()


def test_the_publish_rule_reads_the_files_of_check_outputs(tmp_path):
    files = {"wf_00001_.png": _png(_picture()), "wf_mesh_00001_.glb": bytes(2 * MIB)}
    res = _checked(tmp_path, files, MESH_WORKFLOW, unsent=("wf_mesh_00001_.glb",))
    assert smoke.unpublished(res["files"], None) == {"wf_mesh_00001_.glb"}
    assert smoke.unpublished(res["files"], res["files"]) == set()


# -- the module ----------------------------------------------------------------------
def test_the_limits_and_types_are_the_documented_ones():
    assert smoke.LIMITS["image"] == {"min_side": 64, "min_std": 1.0, "max_flat_share": 0.995}
    assert smoke.LIMITS["text"] == {"max_bytes": MIB, "min_chars": 8, "min_distinct_chars": 4,
                                    "repeat_words": 20, "max_top_word_share": 0.5}
    assert smoke.LIMITS["audio"] == {"min_rate": 8000, "max_rate": 192000, "max_channels": 8,
                                     "min_seconds": 1.0, "seconds_tolerance": 0.05,
                                     "min_seconds_slack": 0.5, "min_bytes_per_second": 4000}
    assert smoke.LIMITS["mesh"] == {"min_vertices": 100, "min_faces": 100, "min_extent": 1e-4}
    assert smoke.LIMITS["video"] == {"min_frames": 2, "min_side": 64, "min_frame_bytes": 100,
                                     "min_frame_bytes_per_mp": 500}
    assert smoke.TYPES == {".png": "image", ".txt": "text", ".flac": "audio", ".glb": "mesh",
                           ".mp4": "video"}
    assert set(smoke.SINKS) == {"SaveImage", "SaveText", "SaveAudio", "SaveAudioAdvanced",
                                "SaveGLB", "SaveVideo"}
    assert (smoke.FILES_MAX, smoke.PROBLEMS_MAX, smoke.UNSENT_MIN_BYTES,
            smoke.PUBLISH_MAX_BYTES) == (50, 20, 1024, MIB)


def test_importing_the_module_needs_only_the_standard_library():
    """numpy and Pillow are loaded by the checks that need them, so a script
    that only reads published results can import the module anywhere."""
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import smoke; "
            "assert not {'numpy', 'PIL'} & set(sys.modules), sorted(sys.modules); "
            "smoke.check_outputs(None, None, None, None, 'no_outputs'); "
            "smoke.unpublished([], None); smoke.main_sha({}, 'wf', 'image'); "
            "smoke.expected_outputs({}); smoke.flac_streaminfo(b'')")
    for flags in (["-S"], []):
        done = subprocess.run([sys.executable, *flags, "-c", code,
                               str(Path(smoke.__file__).parent)], capture_output=True)
        assert done.returncode == 0, done.stderr.decode("utf-8", "replace")[-500:]
