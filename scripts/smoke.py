"""Smoke checks: did a run write its files, and is each of them usable?

A smoke-level workflow has no approved reference. It passes when it finished,
every save node wrote its file, and every file of a known type passes the
checks of its type. This module holds those checks and what a run publishes
about them. run_regression.py calls it for an entry with `"level": "smoke"`.

A file's type comes from its extension (TYPES). A file of another extension is
listed and not checked. Each check reads one file and returns what it measured
and what is wrong with it, with the limits in LIMITS:

  image  .png   Pillow decodes it fully. Each side has at least 64 px. It is
                not blank (the 8-bit grey image has a standard deviation of at
                least 1.0) and not one flat colour with a speck (at most
                99.5 % of the pixels share one grey level).
  text   .txt   UTF-8, at most 1 MiB. At least 8 characters that are not
                blank and 4 different ones. With 20 words or more, the most
                frequent word is at most half of them.
  audio  .flac  A FLAC file that ends with a whole frame at the sample its
                header names. 8,000 to 192,000 Hz, 1 to 8 channels, at least
                1.0 s long, and not silent: at least 4,000 bytes of audio
                frames a second (FLAC stores silence in a few bytes).
  mesh   .glb   A GLB whose first primitive has positions in its binary
                chunk. At least 100 vertices and 100 faces, every position
                finite, and a bounding box larger than 0.0001 in at least two
                directions.
  video  .mp4   An MP4 with a video track. At least 2 frames, each side at
                least 64 px, and not blank: the track holds at least 1,500
                bytes a frame for each megapixel.

A manifest entry's `expect` can pin what its author knows: `width` and
`height` (PNG files), `frames` (video), `seconds` (audio, within 5 % and never
tighter than 0.5 s) and `min_chars` (text, in place of the 8).

No decoder is needed: FLAC, GLB and MP4 say enough in their headers and
tables. The checks read files that a broken run wrote, so a file that cannot
be read is a failure and never a crash: `check_outputs` turns every exception
of a check into a `decode` problem of that file.

Smoke level does not catch a wrong but plausible output: half a picture, a
quiet hiss where music should be, a mesh of the wrong object. That is what a
reference is for.

File names and the sha256 values come from the worker, so whoever shows a
problem sentence escapes it like any other name.

Standard library, numpy and Pillow (scripts/requirements.txt). numpy and
Pillow are imported inside the checks that need them: importing this module
needs the standard library only.
"""
from __future__ import annotations

import collections
import io
import json
import posixpath
import struct
from pathlib import Path

import output_kinds

LIMITS = {
    "image": {"min_side": 64, "min_std": 1.0, "max_flat_share": 0.995},
    "text": {"max_bytes": output_kinds.TEXT_MAX_BYTES, "min_chars": 8, "min_distinct_chars": 4,
             "repeat_words": 20, "max_top_word_share": 0.5},
    "audio": {"min_rate": 8000, "max_rate": 192000, "max_channels": 8, "min_seconds": 1.0,
              "seconds_tolerance": 0.05, "min_seconds_slack": 0.5,
              "min_bytes_per_second": 4000},
    "mesh": {"min_vertices": 100, "min_faces": 100, "min_extent": 1e-4},
    "video": {"min_frames": 2, "min_side": 64, "min_bytes_per_frame_mp": 1500},
}
TYPES = {".png": "image", ".txt": "text", ".flac": "audio", ".glb": "mesh", ".mp4": "video"}
# What SaveGLB writes: a mesh becomes a .glb, a 3D file keeps its own extension.
MESH_EXTENSIONS = (".glb", ".gltf", ".obj", ".fbx", ".stl", ".usdz", ".ply", ".splat", ".spz",
                   ".ksplat")
# Save node class -> the extensions of the file it must have written. Any other
# output node (VHS_VideoCombine) is expected to write nothing in particular.
SINKS = {"SaveImage": (".png",), "SaveText": (".txt",), "SaveAudioAdvanced": (".flac",),
         "SaveAudio": (".flac",), "SaveGLB": MESH_EXTENSIONS, "SaveVideo": (".mp4",)}
FILES_MAX = 50                    # entries in `files`
PROBLEMS_MAX = 20                 # entries in `problems`
UNSENT_MIN_BYTES = 1024           # a file the worker did not send passes on its size from here
PUBLISH_MAX_BYTES = 1024 * 1024   # a larger output file is published only when it repeats
_FLAC_TRIES = 8                   # frame checksums tried while looking for the last frame
_GLB_JSON, _GLB_BIN = 0x4E4F534A, 0x004E4942
_GLB_FLOAT = 5126


class Unreadable(ValueError):
    """A file that is not what its extension says, told in plain words."""


def _problem(check: str, text: str) -> dict:
    return {"check": check, "text": text}


def _expected(expect: dict | None, key: str):
    """The number an entry's `expect` holds for `key`; None when it holds none."""
    value = expect.get(key) if isinstance(expect, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return value


def _count(n: int, word: str, plural: str | None = None) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {plural or word + 's'}"


# -- image ---------------------------------------------------------------------------
def _grey(im):
    """An opened image as 8-bit grey levels. A 16-bit grey PNG is scaled:
    converted the usual way its bright half would all read as white."""
    import numpy as np
    if im.mode.startswith("I"):
        return (np.asarray(im).astype(np.float64) / 257.0).clip(0, 255).astype(np.uint8)
    return np.asarray(im.convert("L"), dtype=np.uint8)


def check_image(path: Path, expect: dict | None = None) -> tuple[dict, list[dict]]:
    """(facts, problems) of a PNG file. Facts: `width`, `height`, `std` (the
    standard deviation of the grey image) and `flat_share` (the share of the
    pixels that have the most common grey level)."""
    import numpy as np
    from PIL import Image, UnidentifiedImageError
    lim = LIMITS["image"]
    try:
        im = Image.open(io.BytesIO(Path(path).read_bytes()))
    except UnidentifiedImageError:
        raise Unreadable("not an image file") from None
    with im:
        im.load()
        facts = {"width": im.width, "height": im.height}
        grey = _grey(im)
    problems = []
    if min(facts["width"], facts["height"]) < lim["min_side"]:
        problems.append(_problem("size", f"{facts['width']}x{facts['height']} px: each side "
                                         f"must have at least {lim['min_side']} px"))
    for key in ("width", "height"):
        want = _expected(expect, key)
        if want is not None and facts[key] != want:
            problems.append(_problem("expect", f"{key} {facts[key]} px, {want:g} px expected"))
    std = float(grey.std())
    flat = float(np.bincount(grey.ravel(), minlength=256).max() / grey.size)
    facts.update(std=round(std, 2), flat_share=round(flat, 4))
    if std < lim["min_std"]:
        problems.append(_problem("blank", f"blank: brightness spread {std:.2f}, under "
                                          f"{lim['min_std']}"))
    elif flat > lim["max_flat_share"]:
        problems.append(_problem("flat", f"one flat colour with a speck: {flat * 100:.2f} % of "
                                         "the pixels have one brightness, over "
                                         f"{lim['max_flat_share'] * 100:.1f} %"))
    return facts, problems


# -- text ----------------------------------------------------------------------------
def check_text(path: Path, expect: dict | None = None) -> tuple[dict, list[dict]]:
    """(facts, problems) of a text file. Facts: `chars` (characters that are
    not blank, the number the limit is held against) and `words`."""
    lim = LIMITS["text"]
    size = Path(path).stat().st_size
    if size > lim["max_bytes"]:
        return {}, [_problem("size", f"{size} bytes: a text output may have at most "
                                     f"{lim['max_bytes']}")]
    words = Path(path).read_bytes().decode("utf-8-sig").split()
    body = "".join(words)
    facts = {"chars": len(body), "words": len(words)}
    problems = []
    own = _expected(expect, "min_chars")
    need = lim["min_chars"] if own is None else own
    if not body:
        problems.append(_problem("empty", "empty"))
    elif len(body) < need:
        problems.append(_problem("empty" if own is None else "expect",
                                 f"{_count(len(body), 'non-blank character')}, at least "
                                 f"{need:g} expected"))
    elif len(set(body)) < lim["min_distinct_chars"]:
        problems.append(_problem("repeat", "only "
                                           f"{_count(len(set(body)), 'different character')}"))
    if len(words) >= lim["repeat_words"]:
        top = collections.Counter(w.casefold() for w in words).most_common(1)[0][1]
        if top / len(words) > lim["max_top_word_share"]:
            problems.append(_problem("repeat", f"one word is {top} of the {len(words)} words"))
    return facts, problems


# -- audio (FLAC) --------------------------------------------------------------------
def _crc_table(poly: int, bits: int) -> tuple[int, ...]:
    top, mask, table = 1 << (bits - 1), (1 << bits) - 1, []
    for byte in range(256):
        crc = byte << (bits - 8)
        for _ in range(8):
            crc = ((crc << 1) ^ poly if crc & top else crc << 1) & mask
        table.append(crc)
    return tuple(table)


_CRC8, _CRC16 = _crc_table(0x07, 8), _crc_table(0x8005, 16)


def _crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc = _CRC8[crc ^ byte]
    return crc


def _crc16(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc = ((crc << 8) & 0xFFFF) ^ _CRC16[(crc >> 8) ^ byte]
    return crc


def _flac_header(data: bytes) -> tuple[dict, int, int] | None:
    """(STREAMINFO fields, the stream's block size in samples, where the audio
    frames start) of a FLAC file; None when `data` is not one, has no
    STREAMINFO block, or ends inside its metadata blocks."""
    if data[:4] != b"fLaC":
        return None
    found, pos = None, 4
    while pos + 4 <= len(data):
        last, kind = data[pos] & 0x80, data[pos] & 0x7F
        length = int.from_bytes(data[pos + 1:pos + 4], "big")
        body = data[pos + 4:pos + 4 + length]
        if len(body) < length:
            return None
        if kind == 0 and length >= 34 and found is None:
            packed = int.from_bytes(body[10:18], "big")
            md5 = body[18:34].hex()
            found = ({"sample_rate": packed >> 44, "channels": ((packed >> 41) & 0x7) + 1,
                      "bits": ((packed >> 36) & 0x1F) + 1, "samples": packed & ((1 << 36) - 1),
                      "pcm_md5": None if md5 == "0" * 32 else md5},
                     int.from_bytes(body[2:4], "big"))
        pos += 4 + length
        if last:
            return (*found, pos) if found else None
    return None


def flac_streaminfo(data: bytes) -> dict | None:
    """STREAMINFO of a FLAC file: `sample_rate`, `channels`, `bits` (of a
    sample), `samples` (in each channel; 0 when the writer did not know) and
    `pcm_md5`, the MD5 of the decoded samples (None when the writer left it
    out). None when `data` is not a FLAC file with that block. Two files with
    the same sound have the same values here, whatever else they hold."""
    head = _flac_header(data)
    return head[0] if head else None


def _flac_frame_end(data: bytes, at: int, block_size: int) -> int | None:
    """The sample the frame whose header starts at `at` ends before; None
    when no valid frame header starts there."""
    head = data[at:at + 16]
    if len(head) < 6 or head[0] != 0xFF or head[1] & 0xFE != 0xF8 or head[3] & 1:
        return None
    size_code, rate_code = head[2] >> 4, head[2] & 0x0F
    # The frame or sample number is coded like UTF-8: the leading ones of its
    # first byte count the bytes it has.
    ones = 8 - (head[4] ^ 0xFF).bit_length()
    if size_code == 0 or rate_code == 0x0F or ones in (1, 8):
        return None
    number, pos = head[4] & (0xFF >> (ones + 1)), 4 + max(ones, 1)
    for byte in head[5:pos]:
        if byte & 0xC0 != 0x80:
            return None
        number = (number << 6) | (byte & 0x3F)
    if size_code == 1:
        size = 192
    elif size_code <= 5:
        size = 576 << (size_code - 2)
    elif size_code <= 7:
        width = size_code - 5
        size, pos = int.from_bytes(head[pos:pos + width], "big") + 1, pos + width
    else:
        size = 256 << (size_code - 8)
    pos += {12: 1, 13: 2, 14: 2}.get(rate_code, 0)
    if pos >= len(head) or _crc8(head[:pos]) != head[pos]:
        return None
    # A stream of one block size numbers its frames, any other its samples.
    return (number if head[1] & 1 else number * block_size) + size


def _flac_end(data: bytes, start: int, info: dict, block_size: int) -> int:
    """The sample the audio of a FLAC file ends before, read from its last
    frame: the last frame header whose frame runs to the end of the file with
    a correct checksum. Unreadable when there is none. A cut file keeps its
    STREAMINFO, so nothing else tells that it was cut."""
    longest = block_size * info["channels"] * (info["bits"] // 8 + 1) + 64
    floor, at, tries = max(start, len(data) - longest), len(data), 0
    while tries < _FLAC_TRIES:
        at = data.rfind(b"\xff", floor, at)
        if at < 0:
            break
        end = _flac_frame_end(data, at, block_size)
        if end is None:
            continue
        tries += 1
        if _crc16(data[at:-2]) == int.from_bytes(data[-2:], "big"):
            return end
    raise Unreadable("the audio data does not end with a whole frame")


def check_audio(path: Path, expect: dict | None = None) -> tuple[dict, list[dict]]:
    """(facts, problems) of a FLAC file. Facts: `sample_rate`, `channels`,
    `bits`, `seconds`, `bytes_per_second` (the audio frames, without the
    metadata blocks) and `pcm_md5` (the MD5 of the decoded samples)."""
    lim = LIMITS["audio"]
    data = Path(path).read_bytes()
    head = _flac_header(data)
    if not head:
        raise Unreadable("not a FLAC file with a STREAMINFO block")
    info, block_size, start = head
    rate, channels, samples = info["sample_rate"], info["channels"], info["samples"]
    facts = {"sample_rate": rate, "channels": channels, "bits": info["bits"], "seconds": 0.0,
             "bytes_per_second": 0, "pcm_md5": info["pcm_md5"]}
    if not lim["min_rate"] <= rate <= lim["max_rate"] or channels > lim["max_channels"]:
        return facts, [_problem("decode", f"not usable audio: {rate} Hz and "
                                          f"{_count(channels, 'channel')}, {lim['min_rate']} to "
                                          f"{lim['max_rate']} Hz expected")]
    if not samples:
        return facts, [_problem("duration", "no samples: the header gives the length 0")]
    end = _flac_end(data, start, info, block_size)
    if end != samples:
        raise Unreadable(f"the audio ends at sample {end}, the header says {samples}")
    seconds = samples / rate
    per_second = (len(data) - start) / seconds
    facts.update(seconds=round(seconds, 3), bytes_per_second=round(per_second))
    problems = []
    want = _expected(expect, "seconds")
    if seconds < lim["min_seconds"]:
        problems.append(_problem("duration", f"{seconds:.2f} s long, at least "
                                             f"{lim['min_seconds']} s expected"))
    elif want is not None and abs(seconds - want) > max(lim["min_seconds_slack"],
                                                        want * lim["seconds_tolerance"]):
        problems.append(_problem("expect", f"{seconds:.1f} s long, {want:g} s expected"))
    if per_second < lim["min_bytes_per_second"]:
        problems.append(_problem("silent", f"silent or constant: {per_second:.0f} bytes of "
                                           "audio a second, under "
                                           f"{lim['min_bytes_per_second']}"))
    return facts, problems


# -- mesh (GLB) ----------------------------------------------------------------------
def _glb_chunks(data: bytes) -> tuple[dict, memoryview]:
    """(the glTF document, the binary chunk) of a GLB file."""
    if len(data) < 12 or data[:4] != b"glTF":
        raise Unreadable("not a GLB file")
    total = struct.unpack_from("<I", data, 8)[0]
    if total > len(data):
        raise Unreadable(f"the file has {len(data)} bytes, its header says {total}")
    chunks, pos = {}, 12
    while pos + 8 <= total:
        length, kind = struct.unpack_from("<II", data, pos)
        if pos + 8 + length > total:
            raise Unreadable("a chunk ends past the end of the file")
        chunks.setdefault(kind, memoryview(data)[pos + 8:pos + 8 + length])
        pos += 8 + length
    if _GLB_JSON not in chunks or _GLB_BIN not in chunks:
        raise Unreadable("no JSON chunk" if _GLB_JSON not in chunks else "no binary chunk")
    gltf = json.loads(bytes(chunks[_GLB_JSON]).decode("utf-8"))
    if not isinstance(gltf, dict):
        raise Unreadable("the JSON chunk is not a glTF document")
    return gltf, chunks[_GLB_BIN]


def _glb_first_primitive(gltf: dict, blob: memoryview):
    """(positions as an array of xyz rows, faces) of the first primitive of
    the first mesh."""
    import numpy as np
    try:
        prim = gltf["meshes"][0]["primitives"][0]
        accessor = gltf["accessors"][prim["attributes"]["POSITION"]]
        view = gltf["bufferViews"][accessor["bufferView"]]
        count = int(accessor["count"])
        corners = int(gltf["accessors"][prim["indices"]]["count"]) if "indices" in prim else count
        start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
        stride = int(view.get("byteStride") or 12)
        mode = prim.get("mode", 4)
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        raise Unreadable("no mesh with positions in the binary chunk") from None
    if accessor.get("componentType") != _GLB_FLOAT or accessor.get("type") != "VEC3":
        raise Unreadable("the positions are not triples of 32-bit floats")
    last = start + (count - 1) * stride + 12 if count else start
    if count < 0 or start < 0 or stride < 12 or last > len(blob):
        raise Unreadable("the positions end past the binary chunk")
    xyz = np.ndarray((count, 3), dtype="<f4", buffer=blob, offset=start, strides=(stride, 4))
    # Triangles, or a strip or a fan of them. Points and lines have no faces.
    faces = corners // 3 if mode == 4 else max(corners - 2, 0) if mode in (5, 6) else 0
    return xyz.astype(np.float64), faces


def check_mesh(path: Path, expect: dict | None = None) -> tuple[dict, list[dict]]:
    """(facts, problems) of a GLB file. Facts: `vertices` and `faces` of its
    first primitive, and `extent`, the three sides of the bounding box (None
    when a position is not finite)."""
    import numpy as np
    lim = LIMITS["mesh"]
    xyz, faces = _glb_first_primitive(*_glb_chunks(Path(path).read_bytes()))
    facts = {"vertices": len(xyz), "faces": faces, "extent": None}
    problems = []
    if len(xyz) < lim["min_vertices"] or faces < lim["min_faces"]:
        problems.append(_problem("size", f"{_count(len(xyz), 'vertex', 'vertices')} and "
                                         f"{_count(faces, 'face')}, at least "
                                         f"{lim['min_vertices']} and {lim['min_faces']} "
                                         "expected"))
    odd = int((~np.isfinite(xyz)).any(axis=1).sum())
    if odd:
        problems.append(_problem("nonfinite", "positions that are NaN or infinite: "
                                              f"{odd} of {len(xyz)}"))
        return facts, problems
    extent = xyz.max(axis=0) - xyz.min(axis=0) if len(xyz) else np.zeros(3)
    facts["extent"] = [round(float(side), 4) for side in extent]
    if int((extent > lim["min_extent"]).sum()) < 2:
        box = " x ".join(f"{side:g}" for side in facts["extent"])
        problems.append(_problem("geometry", f"no real geometry: the bounding box is {box}, "
                                             "a point or a line"))
    return facts, problems


# -- video (MP4) ---------------------------------------------------------------------
def _boxes(data: bytes, start: int, end: int):
    """(type, where the body starts, where the box ends) of each box in
    data[start:end]. Unreadable when a box ends past `end`: the file was cut."""
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack_from(">I4s", data, pos)
        head = 8
        if size == 1 and pos + 16 <= end:
            size, head = struct.unpack_from(">Q", data, pos + 8)[0], 16
        elif size == 0:
            size = end - pos
        if size < head or pos + size > end:
            name = f"the {kind.decode('ascii')}" if kind.isalnum() else "a"
            raise Unreadable(f"{name} box is cut short")
        yield kind, pos + head, pos + size
        pos += size


def _box(data: bytes, start: int, end: int, kind: bytes) -> tuple[int, int]:
    """(where the body starts, where the box ends) of the first `kind` box in
    data[start:end]."""
    for found, a, b in _boxes(data, start, end):
        if found == kind:
            return a, b
    raise Unreadable(f"no {kind.decode('ascii')} box")


def _fields(data: bytes, form: str, at: int, end: int) -> tuple:
    if at < 0 or at + struct.calcsize(form) > end:
        raise Unreadable("a box is too short for its fields")
    return struct.unpack_from(form, data, at)


def _video_track(data: bytes) -> tuple[int, int, int, float, int]:
    """(width, height, frames, seconds, bytes of the frames) of the first
    video track of an MP4 file. Only that track's bytes count, so an audio
    track cannot hide a blank picture."""
    # Every top-level box is walked: a cut file ends inside one of them.
    top = {kind: (a, b) for kind, a, b in _boxes(data, 0, len(data))}
    for kind in (b"ftyp", b"moov"):
        if kind not in top:
            raise Unreadable(f"no {kind.decode('ascii')} box")
    for kind, a, b in _boxes(data, *top[b"moov"]):
        if kind != b"trak":
            continue
        mdia = _box(data, a, b, b"mdia")
        handler = _box(data, *mdia, b"hdlr")
        if data[handler[0] + 8:min(handler[0] + 12, handler[1])] != b"vide":
            continue
        tkhd = _box(data, a, b, b"tkhd")
        width, height = _fields(data, ">II", max(tkhd[1] - 8, tkhd[0] + 76), tkhd[1])
        mdhd = _box(data, *mdia, b"mdhd")
        long_form = _fields(data, ">B", *mdhd)[0] == 1
        scale, duration = _fields(data, ">IQ" if long_form else ">II",
                                  mdhd[0] + (20 if long_form else 12), mdhd[1])
        stsz = _box(data, *_box(data, *_box(data, *mdia, b"minf"), b"stbl"), b"stsz")
        fixed, frames = _fields(data, ">II", stsz[0] + 4, stsz[1])
        size = fixed * frames if fixed else sum(_fields(data, f">{frames}I", stsz[0] + 12,
                                                        stsz[1]))
        return width >> 16, height >> 16, frames, duration / scale if scale else 0.0, size
    raise Unreadable("no video track")


def check_video(path: Path, expect: dict | None = None) -> tuple[dict, list[dict]]:
    """(facts, problems) of an MP4 file. Facts: `width`, `height`, `frames`,
    `seconds` and `bytes_per_frame_mp`, the bytes of the video track for each
    frame and megapixel."""
    lim = LIMITS["video"]
    width, height, frames, seconds, size = _video_track(Path(path).read_bytes())
    megapixels = width * height / 1e6
    density = round(size / frames / megapixels) if frames and megapixels else 0
    facts = {"width": width, "height": height, "frames": frames, "seconds": round(seconds, 3),
             "bytes_per_frame_mp": density}
    problems = []
    if frames < lim["min_frames"]:
        problems.append(_problem("frames", f"{_count(frames, 'frame')}, at least "
                                           f"{lim['min_frames']} expected"))
    if min(width, height) < lim["min_side"]:
        problems.append(_problem("size", f"{width}x{height} px: each side must have at least "
                                         f"{lim['min_side']} px"))
    want = _expected(expect, "frames")
    if want is not None and frames != want:
        problems.append(_problem("expect", f"{_count(frames, 'frame')}, {want:g} expected"))
    if frames and megapixels and density < lim["min_bytes_per_frame_mp"]:
        problems.append(_problem("blank", f"blank or still: {density} bytes a frame for each "
                                          f"megapixel, under {lim['min_bytes_per_frame_mp']}"))
    return facts, problems


_CHECKS = {"image": check_image, "text": check_text, "audio": check_audio, "mesh": check_mesh,
           "video": check_video}


# -- which files a run must have written ---------------------------------------------
def _prefix(node: dict) -> str | None:
    """The path a save node's files start with, from its literal
    `filename_prefix`; None when it cannot be known here (a link, a `%...%`
    placeholder that ComfyUI fills in, a path out of the output folder)."""
    inputs = node.get("inputs")
    prefix = inputs.get("filename_prefix") if isinstance(inputs, dict) else None
    if not isinstance(prefix, str) or not prefix or "%" in prefix:
        return None
    prefix = posixpath.normpath(prefix.replace("\\", "/"))
    return None if prefix.startswith(("/", "..")) or prefix == "." or ":" in prefix else prefix


def expected_outputs(workflow: dict) -> list[dict]:
    """What the save nodes of an API-format workflow must have written: one
    entry for each node of a class in SINKS, in the workflow's order.
    {"node": its id, "class": its class, "prefix": what its files start with
    (None: any name), "extensions": what they end in}. ComfyUI accepts a prompt
    in which one output branch is invalid and runs the others, so a run that
    finished may still lack a file."""
    found = []
    for node_id, node in (workflow.items() if isinstance(workflow, dict) else ()):
        sink = node.get("class_type") if isinstance(node, dict) else None
        if isinstance(sink, str) and sink in SINKS:
            found.append({"node": str(node_id), "class": sink, "prefix": _prefix(node),
                          "extensions": list(SINKS[sink])})
    return found


def _fits(name: str, prefix: str | None, extensions: tuple[str, ...]) -> bool:
    return (name.lower().endswith(extensions)
            and (prefix is None or name.startswith(prefix + "_")))


def _missing(expected: list[dict], names: list[str]) -> list[dict]:
    """A `missing` problem for each kind of file the save nodes wrote too few
    of. Nodes that share a prefix and extensions must have written a file
    each. A file counts for the longest prefix it starts with, so
    `wf_mesh_00001_.png` is never taken for a file of prefix `wf`."""
    nodes: dict[tuple, list[str]] = {}
    for sink in expected:
        nodes.setdefault((sink["prefix"], tuple(sink["extensions"])), []).append(sink["class"])
    have: collections.Counter = collections.Counter()
    for name in names:
        fits = [key for key in nodes if _fits(name, *key)]
        longest = max((len(key[0] or "") for key in fits), default=0)
        have.update(key for key in fits if len(key[0] or "") == longest)
    problems = []
    for (prefix, extensions), classes in nodes.items():
        if have[(prefix, extensions)] >= len(classes):
            continue
        what = f"{extensions[0] if len(extensions) == 1 else '3D'} file"
        where = f" with the prefix {prefix}" if prefix is not None else ""
        who = " and ".join(sorted(set(classes)))
        if len(classes) == 1:
            text = f"{who} wrote no {what}{where}"
        else:
            text = (f"{len(classes)} {who} nodes wrote "
                    f"{_count(have[(prefix, extensions)], what)}{where}, {len(classes)} expected")
        problems.append({"file": None, **_problem("missing", text)})
    return problems


# -- one run -------------------------------------------------------------------------
def _why(e: Exception) -> str:
    """One line on why a file could not be read. An error of the file system
    is cut down to its reason, so no path of the runner is published."""
    if isinstance(e, Unreadable):
        return str(e)
    line = e.strerror if isinstance(e, OSError) and e.strerror else str(e)
    line = (line.splitlines() or [""])[0].strip()
    return (f"{type(e).__name__}: {line}" if line else type(e).__name__)[:200]


def _on_runner(root: Path | None, name: str) -> Path:
    """Where the output file `name` is on the runner. The name comes from the
    worker: one that leaves the outputs folder is not opened."""
    if root is None:
        raise Unreadable("the run has no outputs folder")
    path = (root / name).resolve()
    if root.resolve() not in path.parents:
        raise Unreadable("its name leaves the outputs folder")
    return path


def _check_file(root: Path | None, rec: dict, expect: dict | None,
                notes: list[str]) -> tuple[dict, list[dict]]:
    """(the `files` entry, the problems) of one output file the worker listed."""
    name = rec["filename"]
    kind = TYPES.get(posixpath.splitext(name)[1].lower())
    entry = {"name": name, "type": kind, "bytes": rec.get("bytes"), "sha256": rec.get("sha256")}
    sent = not rec.get("truncated")
    if not sent:
        entry["sent"] = False
    if kind is None:
        return entry, []
    if not sent:
        size = rec.get("bytes")
        size = size if isinstance(size, int) and not isinstance(size, bool) else 0
        if size >= UNSENT_MIN_BYTES:
            notes.append(f"{name} was not sent (over the worker's size limit): checked by "
                         "size only")
            return entry, []
        return entry, [_problem("size", f"not sent, and too small to pass by its size: "
                                        f"{_count(size, 'byte')}, under {UNSENT_MIN_BYTES}")]
    try:
        facts, problems = _CHECKS[kind](_on_runner(root, name), expect)
    except Exception as e:  # a file that a broken run wrote: any error is the finding
        return entry, [_problem("decode", f"could not be read: {_why(e)}")]
    entry.update(facts)
    return entry, problems


def check_outputs(outputs_dir: Path | str | None, run_rec: dict | None, workflow: dict | None,
                  expect: dict | None, status: str | None) -> dict:
    """The `smoke` object of one run, from the files under `outputs_dir`, the
    run record that lists them, the workflow that was run (with its save
    nodes) and the entry's `expect`:

      files     one entry for each output file the worker listed, by name, at
                most FILES_MAX: `name`, `type` (None for an extension that is
                not checked), `bytes` and `sha256` as the worker reported
                them, `sent: False` for a file it listed and did not send,
                then what the check measured
      problems  why the run does not pass, at most PROBLEMS_MAX:
                {"file": name or None, "check": id, "text": sentence}
      notes     sentences that are not failures

    Only a run with worker status "ok" is checked: it passes when `problems`
    is empty. Status "no_outputs" (finished and wrote nothing) is one
    `missing` problem. Any other status means the workflow did not finish, so
    the three lists are empty and the status is the result.

    A file that was not sent cannot be opened. It passes on its size alone
    from UNSENT_MIN_BYTES, with a note. Check ids: missing, decode, size,
    expect, blank, flat, empty, repeat, duration, silent, nonfinite, geometry,
    frames."""
    smoke: dict = {"files": [], "problems": [], "notes": []}
    if status == "no_outputs":
        smoke["problems"].append({"file": None, **_problem(
            "missing", "the workflow finished and wrote no file")})
    if status != "ok":
        return smoke
    outputs = (run_rec or {}).get("outputs")
    listed = sorted((o for o in (outputs if isinstance(outputs, list) else [])
                     if isinstance(o, dict) and isinstance(o.get("filename"), str)
                     and o["filename"]), key=lambda o: o["filename"])
    root = Path(outputs_dir) if outputs_dir else None
    files, notes = [], []
    problems = _missing(expected_outputs(workflow), [o["filename"] for o in listed])
    for rec in listed:
        entry, found = _check_file(root, rec, expect, notes)
        files.append(entry)
        problems += [{"file": entry["name"], **p} for p in found]
    if len(files) > FILES_MAX:
        notes.append(f"{len(files)} output files: only the first {FILES_MAX} are listed")
    if len(problems) > PROBLEMS_MAX:
        notes.append(f"{len(problems)} problems: only the first {PROBLEMS_MAX} are listed")
    smoke.update(files=files[:FILES_MAX], problems=problems[:PROBLEMS_MAX], notes=notes)
    return smoke


def main_sha(run_rec: dict | None, wf_id: str, kind: str) -> str | None:
    """The sha256 that identifies a run's main output, from its record alone:
    the first PNG by name for an image workflow (as at reference level), the
    text rule of output_kinds for a text workflow, and for an audio workflow
    the first FLAC by name that starts with the workflow id. None when the
    run has no such file."""
    if kind == "text":
        return output_kinds.text_sha(run_rec)
    outputs = (run_rec or {}).get("outputs")
    names = sorted((str(o.get("filename", "")), i) for i, o in
                   enumerate(outputs if isinstance(outputs, list) else [])
                   if isinstance(o, dict))
    if kind == "image":
        found = [i for name, i in names if name.endswith(".png")]
    elif kind == "audio":
        found = [i for name, i in names
                 if name.lower().endswith(".flac") and name.startswith(wf_id)]
    else:
        found = []
    sha = outputs[found[0]].get("sha256") if found else None
    return sha if isinstance(sha, str) else None


def unpublished(files: list[dict], previous_files: list[dict] | None) -> set[str]:
    """The names of the output files a smoke run does not publish, from its
    `files` and those of the workflow's previous result on the lane. A file is
    published when it has at most PUBLISH_MAX_BYTES, or when it has the sha256
    the same name had in the previous result. So an output that repeats is
    stored from its second run on, and one that never repeats is never
    stored: git keeps equal files once, and smoke level is chosen exactly
    where an output may differ from run to run."""
    before = {f.get("name"): f.get("sha256")
              for f in (previous_files if isinstance(previous_files, list) else [])
              if isinstance(f, dict)}
    held = set()
    for f in files:
        name, size, sha = f.get("name"), f.get("bytes"), f.get("sha256")
        large = not isinstance(size, int) or size > PUBLISH_MAX_BYTES
        repeats = isinstance(sha, str) and bool(sha) and before.get(name) == sha
        if isinstance(name, str) and large and not repeats:
            held.add(name)
    return held
