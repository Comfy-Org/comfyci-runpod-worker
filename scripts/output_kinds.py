"""Output kinds: what a workflow's output is, and how two runs of it are compared.

A manifest entry names its kind in `output_kind`; an entry without the key is
an "image" workflow, compared the way compare.py always did. A comparator
answers, for one kind: how two output directories differ, which limits a
comparison with the reference is held to, whether it passes, what is rendered
next to the outputs, which sha256 identifies the output, and whether a
reference may be approved.

"audio" is a kind without a comparator so far: an audio workflow runs at
smoke level only (smoke.py), where no output is compared.

Standard library only, like lanes.py and regression_skip.py: run_report.py
imports the text helpers and renders without installing anything. The image
comparator needs numpy and Pillow, so it lives in run_regression.py next to
the functions it wraps.

Text. The files are the `.txt` outputs, paired by file name. The pass rule is
equality of the normalised text: line ends, blanks at the end of a line and
empty lines at the end of the file do not count; everything else does. A
difference is described by where it starts (line and column, 1-based), by a
similarity ratio and by the two lines there. A reference that differs from its
own second run never loosens the rule by itself: one changed token moves the
rest of a generated text, so "three times the noise" means nothing here.

Generated text is model output, so it is untrusted: nothing here prints it,
and `loggable` gives the metrics a log line may carry.
"""
from __future__ import annotations

import difflib
import hashlib
import itertools
import re
import textwrap
from pathlib import Path

KINDS = ("image", "text", "audio")
DEFAULT_KIND = "image"

TEXT_EXTENSIONS = (".txt",)
TEXT_MAX_BYTES = 1024 * 1024   # a larger text file is an error, never a slow diff
WORDS_MAX = 5000               # longer texts are scored by lines, not by words
LINES_MAX = 5000               # and texts with more lines by their shared start
DIFF_TOKENS_MAX = 6000         # longer texts get a line diff, not a word diff
MATCH_STEPS_MAX = 50_000_000   # the work one difflib match may take: about five seconds
UNMATCHED_LINES = 40           # lines of each text a line diff shows when no match was made
LINE_DIFF_HEAD = "--- reference\n+++ candidate\n"  # how `line_diff` starts: a report tells by it
BASIS_NOTES = {"lines": "by lines", "prefix": "by the shared start"}
EXCERPT_CHARS = 120
WRAP_COLUMNS = 100
_TOKEN = re.compile(r"\S+|\s+")


def kind_of(entry: dict | None) -> str:
    """The output kind a manifest entry declares (absent or null: image);
    ValueError when it is anything else than a known kind."""
    kind = (entry or {}).get("output_kind")
    if kind is None:
        return DEFAULT_KIND
    if not isinstance(kind, str) or kind not in KINDS:
        raise ValueError(f"unknown output_kind {kind!r} (known: {', '.join(KINDS)})")
    return kind


def metrics_kind(metrics: dict | None) -> str:
    """The kind a metrics dict (vs_golden, vs_previous, noise_floor.json)
    describes. Image metrics carry no `kind` key, as they always did."""
    kind = metrics.get("kind") if isinstance(metrics, dict) else None
    return kind if isinstance(kind, str) and kind in KINDS else DEFAULT_KIND


def loggable(metrics: dict) -> dict:
    """`metrics` as a log line may carry them: without the two lines of text
    at the first difference. Image metrics have none and come back as they are."""
    diff = metrics.get("first_diff") if isinstance(metrics, dict) else None
    if not isinstance(diff, dict):
        return metrics
    return {**metrics, "first_diff": {k: v for k, v in diff.items() if k not in ("ref", "cand")}}


def outputs_of(run_rec: dict | None, extensions: tuple[str, ...]) -> list[dict]:
    """A run record's output entries with one of `extensions`, by file name."""
    outputs = (run_rec or {}).get("outputs")
    found = [o for o in (outputs if isinstance(outputs, list) else [])
             if isinstance(o, dict) and str(o.get("filename", "")).lower().endswith(extensions)]
    return sorted(found, key=lambda o: str(o.get("filename")))


def text_sha(run_rec: dict | None) -> str | None:
    """The sha256 that identifies a run's text output, from its record alone:
    the text file's own sha256 as the worker reported it. With several text
    files, the sha256 of their "<sha256>  <name>" lines, so a change in any of
    them shows."""
    files = outputs_of(run_rec, TEXT_EXTENSIONS)
    if not files:
        return None
    if len(files) == 1:
        sha = files[0].get("sha256")
        return sha if isinstance(sha, str) else None
    listing = "".join(f"{o.get('sha256')}  {o.get('filename')}\n" for o in files)
    return hashlib.sha256(listing.encode("utf-8")).hexdigest()


# -- text ---------------------------------------------------------------------------
def normalise_text(data: bytes | str) -> str:
    """The text a comparison sees: UTF-8 (undecodable bytes replaced), no byte
    order mark, "\\n" line ends, no blanks at line ends, no empty lines at the end."""
    text = data.decode("utf-8", "replace") if isinstance(data, bytes) else data
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip(" \t") for line in text.split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _excerpt(line: str | None, col: int) -> str | None:
    """`line` cut to EXCERPT_CHARS around its 1-based `col`; cuts are marked."""
    if line is None:
        return None
    start = max(0, min(col - 1 - EXCERPT_CHARS // 2, len(line) - EXCERPT_CHARS))
    cut = line[start:start + EXCERPT_CHARS]
    return ("…" if start else "") + cut + ("…" if start + EXCERPT_CHARS < len(line) else "")


def first_difference(ref: str, cand: str) -> dict | None:
    """Where two normalised texts start to differ: 1-based line and column, the
    0-based character offset (the length of what they share from the start),
    and the two lines there (None for a text that has already ended). None
    when the texts are equal."""
    if ref == cand:
        return None
    a, b = ref.split("\n"), cand.split("\n")
    row = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    x = a[row] if row < len(a) else None
    y = b[row] if row < len(b) else None
    col = next((i for i, (p, q) in enumerate(zip(x or "", y or "")) if p != q),
               min(len(x or ""), len(y or "")))
    # One text has ended: the line break the other has before this line is not shared.
    ended = 1 if row == min(len(a), len(b)) else 0
    offset = sum(len(line) + 1 for line in a[:row]) + col - ended
    return {"line": row + 1, "col": col + 1, "offset": offset,
            "ref": _excerpt(x, col + 1), "cand": _excerpt(y, col + 1)}


class _TooSlow(Exception):
    """A match that would take more than MATCH_STEPS_MAX steps."""


class _Matcher(difflib.SequenceMatcher):
    """difflib's matcher of two token lists, stopped (_TooSlow) when its work
    would pass MATCH_STEPS_MAX. No limit on the length does that: when the
    lists share little more than one frequent token (a blank, a table's "|",
    an empty line), difflib's time grows with the cube of the length. The
    steps a search can take are known before it runs, so the same two texts
    stop at the same place on every machine. The blocks are difflib's own.
    `autojunk` is difflib's switch: on, a token that makes up over 1 % of a
    list of 200 or more starts no match, as in difflib.unified_diff."""

    def __init__(self, a: list[str], b: list[str], autojunk: bool = False):
        super().__init__(None, a, b, autojunk=autojunk)
        self._blocks: list[tuple[int, int, int]] | None = None

    def get_matching_blocks(self):  # ratio() and get_opcodes() are built on this
        if self._blocks is not None:
            return self._blocks
        a, b = self.a, self.b
        # A search of a[lo:hi] visits every place in b that difflib keeps for one of its tokens.
        upto = [0, *itertools.accumulate(len(self.b2j.get(token, ())) + 1 for token in a)]
        steps, todo, found = 0, [(0, len(a), 0, len(b))], []
        while todo:
            alo, ahi, blo, bhi = todo.pop()
            steps += upto[ahi] - upto[alo]
            if steps > MATCH_STEPS_MAX:
                raise _TooSlow
            i, j, size = self.find_longest_match(alo, ahi, blo, bhi)
            if size:
                found.append((i, j, size))
                if alo < i and blo < j:
                    todo.append((alo, i, blo, j))
                if i + size < ahi and j + size < bhi:
                    todo.append((i + size, ahi, j + size, bhi))
        self._blocks = [*sorted(found), (len(a), len(b), 0)]
        return self._blocks


def similarity(ref: str, cand: str) -> tuple[float, str]:
    """(ratio, basis): difflib's ratio over words (text split at blanks), or
    over lines for texts too long for that. Texts too long for either get the
    share of the longer text that the two have in common from the start
    ("prefix"). Too long is a count of words or lines, or a match that
    difflib cannot finish in MATCH_STEPS_MAX steps. A ratio by lines or by the
    shared start is never above what the words allow. 1.0 only for equal texts."""
    if ref == cand:
        return 1.0, "words"
    words = ref.split(), cand.split()
    ratio, basis = None, "prefix"
    for name, a, b, most in (("words", *words, WORDS_MAX),
                             ("lines", ref.split("\n"), cand.split("\n"), LINES_MAX)):
        if max(len(a), len(b)) > most:
            continue
        try:
            ratio, basis = _Matcher(a, b).ratio(), name
            break
        except _TooSlow:
            pass
    if ratio is None:
        ratio = (first_difference(ref, cand) or {}).get("offset", 0) / max(len(ref), len(cand))
    if basis != "words":
        # Equal lines around one long changed line would pass a limit that the
        # words fail. difflib's upper bound of the word ratio is one pass over them.
        ratio = min(ratio, difflib.SequenceMatcher(None, *words, autojunk=False).quick_ratio())
    return min(round(ratio, 6), 0.999999), basis


def basis_note(metrics: dict | None) -> str:
    """What follows a similarity that is not a share of words, where it is
    shown: " by lines" or " by the shared start". "" for a share of words, and
    for a basis this does not know: the value is read from results."""
    basis = metrics.get("similarity_basis") if isinstance(metrics, dict) else None
    return f" {BASIS_NOTES[basis]}" if isinstance(basis, str) and basis in BASIS_NOTES else ""


def compare_texts(ref: str, cand: str) -> dict:
    """Metrics of one normalised text against another."""
    ratio, basis = similarity(ref, cand)
    res = {"identical": ref == cand, "similarity": ratio, "similarity_basis": basis,
           "chars_ref": len(ref), "chars_cand": len(cand),
           "lines_ref": ref.count("\n") + 1 if ref else 0,
           "lines_cand": cand.count("\n") + 1 if cand else 0}
    diff = first_difference(ref, cand)
    if diff:
        res["first_diff"] = diff
    return res


def list_texts(d: Path) -> dict[str, Path]:
    """{relative name: path} of the text outputs under `d`."""
    d = Path(d)
    if not d.is_dir():
        return {}
    return {p.relative_to(d).as_posix(): p for p in sorted(d.rglob("*"))
            if p.is_file() and p.suffix.lower() in TEXT_EXTENSIONS}


def read_text(path: Path) -> str:
    return normalise_text(Path(path).read_bytes())


def compare_text_dirs(ref_dir: Path, cand_dir: Path) -> dict:
    """compare.compare_dirs for text outputs. Files are paired by name. The
    result counts every file, reports the lowest similarity of a pair and
    describes the first pair (by name) that differs."""
    ref, cand = list_texts(ref_dir), list_texts(cand_dir)
    if not ref or not cand:
        return {"kind": "text",
                "error": f"no text files to compare (ref={len(ref)}, cand={len(cand)})"}
    for name, path in [*ref.items(), *cand.items()]:
        if path.stat().st_size > TEXT_MAX_BYTES:
            return {"kind": "text", "error": f"text output over {TEXT_MAX_BYTES} bytes: {name}"}
    texts_ref = {name: read_text(path) for name, path in ref.items()}
    texts_cand = {name: read_text(path) for name, path in cand.items()}
    common = sorted(set(ref) & set(cand))
    res: dict = {"kind": "text", "files_compared": len(common), "file_count_ref": len(ref),
                 "file_count_cand": len(cand), "identical": sorted(ref) == sorted(cand),
                 "similarity": 1.0 if common else 0.0, "similarity_basis": "words",
                 "chars_ref": sum(map(len, texts_ref.values())),
                 "chars_cand": sum(map(len, texts_cand.values())),
                 "lines_ref": sum(t.count("\n") + 1 for t in texts_ref.values() if t),
                 "lines_cand": sum(t.count("\n") + 1 for t in texts_cand.values() if t)}
    empty = sum(1 for t in texts_ref.values() if not t)
    if empty:  # the counts above are totals: they hide an empty file next to a full one
        res["files_empty_ref"] = empty
    for name in common:
        one = compare_texts(texts_ref[name], texts_cand[name])
        if one["similarity"] < res["similarity"]:
            res["similarity"], res["similarity_basis"] = one["similarity"], one["similarity_basis"]
        if not one["identical"]:
            res["identical"] = False
            res.setdefault("first_diff", {"file": name, **one["first_diff"]})
    if not res["identical"] and res["similarity"] >= 1.0:
        res["similarity"] = 0.999999  # only the file lists differ
    return res


def percent(ratio: float) -> str:
    """A similarity below 1 as a percentage with one decimal, for display.
    Never "100.0": texts that differ are not shown as fully alike."""
    text = f"{ratio * 100:.1f}"
    return "over 99.9" if text == "100.0" and ratio < 1 else text


def wrap_lines(text: str, width: int = WRAP_COLUMNS) -> str:
    """`text` with its long lines folded at blanks, for display only."""
    out: list[str] = []
    for line in text.split("\n"):
        out += textwrap.wrap(line, width, break_long_words=False, break_on_hyphens=False,
                             replace_whitespace=False, drop_whitespace=True) or [""]
    return "\n".join(out)


def _marks(removed: str, added: str) -> str:
    return ("[-" + removed + "-]" if removed else "") + ("{+" + added + "+}" if added else "")


def _span(start: int, stop: int) -> str:
    """Lines `start` to `stop` (from 0, `stop` not included) as a unified diff names them."""
    count = stop - start
    return str(start + 1) if count == 1 else f"{start + 1 if count else start},{count}"


def line_diff(ref: str, cand: str) -> str:
    """A unified diff of the lines of two texts with one line of context, as
    difflib.unified_diff writes it, held to the step bound of `_Matcher`:
    difflib's own line match has none. Past the bound no line is matched.
    What lies between the start and the end that the two texts share is then
    one change, shown up to UNMATCHED_LINES lines of each text, and its
    header says when that is not all of it. For display only."""
    if ref == cand:
        return ""
    a, b = ref.split("\n"), cand.split("\n")
    note = ""
    try:
        groups = list(_Matcher(a, b, autojunk=True).get_grouped_opcodes(1))
    except _TooSlow:
        most = min(len(a), len(b))
        head = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), most)
        tail = min(most - head, next((i for i, (x, y) in enumerate(zip(reversed(a), reversed(b)))
                                      if x != y), most))
        gone, new = len(a) - tail - head, len(b) - tail - head
        if max(gone, new) > UNMATCHED_LINES:
            note = " too long to match line by line, only these lines are shown"
        groups = [[("replace", head, head + min(gone, UNMATCHED_LINES),
                    head, head + min(new, UNMATCHED_LINES))]]
    out = [LINE_DIFF_HEAD.rstrip("\n")]
    for group in groups:
        out.append(f"@@ -{_span(group[0][1], group[-1][2])} "
                   f"+{_span(group[0][3], group[-1][4])} @@{note}")
        for op, i1, i2, j1, j2 in group:
            if op == "equal":
                out += [" " + line for line in a[i1:i2]]
            else:
                out += ["-" + line for line in a[i1:i2]] + ["+" + line for line in b[j1:j2]]
    return "\n".join(out)


def word_diff(ref: str, cand: str) -> str:
    """The candidate text with its changes marked the way `git diff
    --word-diff` marks them: [-removed-] and {+added+}. Texts too long for a
    word diff (in tokens, or in the steps the match would take) get the
    unified line diff of `line_diff`, which starts with LINE_DIFF_HEAD. For
    display only."""
    a, b = _TOKEN.findall(ref), _TOKEN.findall(cand)
    ops = None
    if max(len(a), len(b)) <= DIFF_TOKENS_MAX:
        try:
            ops = _Matcher(a, b).get_opcodes()
        except _TooSlow:
            pass
    if ops is None:
        return line_diff(ref, cand)
    out: list[str] = []
    gone = new = ""
    for k, (op, i1, i2, j1, j2) in enumerate(ops):
        same = "".join(a[i1:i2]) if op == "equal" else None
        # Blanks alone between two changes are part of one change, as git shows
        # it: a text with little in common with its reference is not a wall of marks.
        if same is None or (same.isspace() and 0 < k < len(ops) - 1):
            gone, new = gone + "".join(a[i1:i2]), new + "".join(b[j1:j2])
            continue
        out += [_marks(gone, new), same]
        gone = new = ""
    return "".join([*out, _marks(gone, new)])


# -- the interface --------------------------------------------------------------------
class Comparator:
    """One output kind. A `compare` result always carries `error` or `identical`."""
    kind = ""
    has_figures = False   # `render` writes files under figures/ when outputs differ
    bless_gate = False    # `blessable` has something to say about this kind

    def compare(self, ref_dir: Path, cand_dir: Path) -> dict:
        raise NotImplementedError

    def thresholds(self, entry: dict, defaults: dict, noise_floor: dict | None) -> dict | None:
        """The limits a comparison with the reference is held to (None: none,
        the outputs must be identical)."""
        raise NotImplementedError

    def passes(self, metrics: dict, thresholds: dict | None) -> bool:
        raise NotImplementedError

    def render(self, ref_dir: Path, cand_dir: Path, figures_dir: Path, label: str,
               ref_name: str, metrics: dict, figures: bool = True) -> dict:
        """Write what shows the difference under `figures_dir` (unless
        `figures` is False or there is nothing to show); returns the metrics."""
        return dict(metrics)

    def thumbnails(self, outputs_dir: Path) -> str | None:
        """Write previews next to the outputs; the first one's file name."""
        return None

    def primary_sha(self, run_rec: dict) -> str | None:
        """The sha256 that identifies a run's output, from its record alone."""
        raise NotImplementedError

    def zero_floor(self) -> dict:
        """The noise floor of a reference that was never produced twice."""
        raise NotImplementedError

    def transfer_error(self, run_rec: dict) -> str | None:
        """Why the output cannot be compared at all (a file the worker listed
        and did not send); None when everything the comparison needs arrived."""
        return None

    def describe(self, outputs_dir: Path) -> dict:
        """What a noise floor says about the output itself, read from
        `outputs_dir`, for a reference that is taken from a run and not produced
        twice. Carries `error` when these outputs cannot be a reference."""
        return {}

    def left_behind(self, old_dir: Path, new_dir: Path) -> list[str]:
        """Compared files under `old_dir` that `new_dir` has no file for. A
        reference written over an earlier one adds files and removes none, so
        these would stay in it."""
        return []

    def blessable(self, noise_floor: dict | None, entry: dict) -> str | None:
        """Why a reference must not be approved; None when it may."""
        return None


class TextComparator(Comparator):
    kind = "text"
    bless_gate = True

    def compare(self, ref_dir: Path, cand_dir: Path) -> dict:
        return compare_text_dirs(ref_dir, cand_dir)

    def thresholds(self, entry: dict, defaults: dict, noise_floor: dict | None) -> dict | None:
        """None (identical text) unless the entry sets a similarity floor."""
        own = (entry.get("thresholds") or {}).get("min_text_similarity")
        return {"min_text_similarity": float(own)} if own is not None else None

    @staticmethod
    def _same_files(metrics: dict) -> bool:
        """Both sides hold the same file names: as many files, and each one paired."""
        count = metrics.get("file_count_ref")
        return (count == metrics.get("file_count_cand")
                and metrics.get("files_compared", count) == count)

    def passes(self, metrics: dict, thresholds: dict | None) -> bool:
        if metrics.get("error"):
            return False
        if metrics.get("identical"):
            return True
        if not self._same_files(metrics):
            return False  # a file that was never compared is not within any limit
        floor = (thresholds or {}).get("min_text_similarity")
        return floor is not None and floor < 1.0 and (metrics.get("similarity") or 0.0) >= floor

    def primary_sha(self, run_rec: dict) -> str | None:
        return text_sha(run_rec)

    def zero_floor(self) -> dict:
        return {"kind": "text", "identical": True, "similarity": 1.0}

    def transfer_error(self, run_rec: dict) -> str | None:
        cut = [str(o.get("filename")) for o in outputs_of(run_rec, TEXT_EXTENSIONS)
               if o.get("truncated")]
        if cut:
            return ("output not transferred, over the worker's inline size limit: "
                    + ", ".join(cut[:5]))
        return None

    def describe(self, outputs_dir: Path) -> dict:
        own = self.compare(outputs_dir, outputs_dir)
        if own.get("error"):
            return own
        return {k: v for k, v in own.items() if k.startswith(("file", "chars_", "lines_"))}

    def left_behind(self, old_dir: Path, new_dir: Path) -> list[str]:
        return sorted(set(list_texts(old_dir)) - set(list_texts(new_dir)))

    def blessable(self, noise_floor: dict | None, entry: dict) -> str | None:
        """The second run must pass against the first by the rule every later
        run is held to: a limit in the entry opens the gate only that far."""
        floor = noise_floor or {}
        if floor.get("error"):
            return f"its two runs could not be compared ({floor['error']})"
        if floor.get("chars_ref") == 0:
            return "the generated text is empty"
        if floor.get("files_empty_ref"):
            return "it has an empty text file"
        if floor and metrics_kind(floor) != self.kind:
            return "its noise floor is not a text comparison, so it is a golden of another kind"
        limit = self.thresholds(entry, {}, floor)
        if not floor.get("measured") or self.passes(floor, limit):
            return None
        if not self._same_files(floor):
            return ("its two runs did not write the same text files, so the reference would "
                    "fail at random: make the workflow deterministic")
        said = ("its two runs produced different text (similarity "
                f"{floor.get('similarity')}{basis_note(floor)})")
        if limit is None:
            return (f"{said}, so an exact-match reference would fail at random: make the "
                    "workflow deterministic, or set thresholds.min_text_similarity in its "
                    "manifest entry")
        return (f"{said}, which does not pass thresholds.min_text_similarity "
                f"{limit['min_text_similarity']:g} of its manifest entry, so the reference "
                "would fail at random: make the workflow deterministic, or lower the limit")
