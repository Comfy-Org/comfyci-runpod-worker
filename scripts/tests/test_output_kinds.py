"""output_kinds: the text comparison, and a text workflow through a run."""
import difflib
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

import output_kinds
import run_regression

TEXT = "The first line.\nThe second line has seven words here.\nThe third."
CMP = output_kinds.TextComparator()
TEXT_FLOOR = {"kind": "text", "identical": True, "similarity": 1.0, "chars_ref": 40}


def _dir(tmp_path, name, **files):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    for stem, text in files.items():
        (d / f"{stem}.txt").write_bytes(text.encode("utf-8"))
    return d


def test_kind_of_defaults_to_image_and_refuses_what_it_does_not_know():
    assert output_kinds.kind_of({}) == output_kinds.kind_of(None) == "image"
    assert output_kinds.kind_of({"output_kind": None}) == "image"
    assert output_kinds.kind_of({"output_kind": "text"}) == "text"
    with pytest.raises(ValueError, match="unknown output_kind 'audio'"):
        output_kinds.kind_of({"output_kind": "audio"})
    # Only an absent key or null means image: a typo must not run as one.
    for odd in ("", "Text", False, 0, ["text"]):
        with pytest.raises(ValueError, match="unknown output_kind"):
            output_kinds.kind_of({"output_kind": odd})
    # Image metrics never carried a kind; anything odd reads as image too.
    assert output_kinds.metrics_kind({"identical": True}) == "image"
    assert output_kinds.metrics_kind({"kind": "text"}) == "text"
    assert output_kinds.metrics_kind({"kind": ["text"]}) == "image"
    assert output_kinds.metrics_kind(None) == output_kinds.metrics_kind("text") == "image"


def test_the_module_and_the_report_need_only_the_standard_library():
    """The "All lanes" job renders the report without installing anything:
    -S leaves site-packages out, so any other import would fail here."""
    code = "import sys; sys.path.insert(0, sys.argv[1]); import run_report, output_kinds"
    done = subprocess.run([sys.executable, "-S", "-c", code,
                           str(Path(output_kinds.__file__).parent)], capture_output=True)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")[-500:]


def test_normalised_text_ignores_line_ends_and_trailing_blanks_only():
    n = output_kinds.normalise_text
    assert n(b"\xef\xbb\xbfa \r\nb\t\r\n\r\n") == n("a\nb") == "a\nb"
    assert n("a\rb") == "a\nb"
    assert n(" a") != n("a") and n("a\n\nb") != n("a\nb") and n("A") != n("a")
    assert n(b"ok \xff") == "ok �"  # bytes that are not UTF-8 never raise
    assert n("a\n \n\t\n") == "a" and n("") == n("\n\n") == ""


def test_identical_texts(tmp_path):
    ref = _dir(tmp_path, "ref", wf_00001=TEXT + "\n")
    cand = _dir(tmp_path, "cand", wf_00001=TEXT.replace("\n", "\r\n") + "  \n\n")
    res = CMP.compare(ref, cand)
    assert res == {"kind": "text", "files_compared": 1, "file_count_ref": 1, "file_count_cand": 1,
                   "identical": True, "similarity": 1.0, "similarity_basis": "words",
                   "chars_ref": len(TEXT), "chars_cand": len(TEXT), "lines_ref": 3, "lines_cand": 3}
    assert CMP.passes(res, None) and CMP.thresholds({"thresholds": None}, {}, None) is None


def test_a_difference_names_its_first_line_and_column(tmp_path):
    ref = _dir(tmp_path, "ref", wf_00001=TEXT)
    cand = _dir(tmp_path, "cand", wf_00001=TEXT.replace("seven", "eight"))
    res = CMP.compare(ref, cand)
    assert res["identical"] is False and not CMP.passes(res, None)
    assert res["first_diff"] == {"file": "wf_00001.txt", "line": 2, "col": 21, "offset": 36,
                                 "ref": "The second line has seven words here.",
                                 "cand": "The second line has eight words here."}
    assert TEXT[36:41] == "seven" and 0.8 < res["similarity"] < 1.0
    # A text that ends early differs where it ends.
    short = CMP.compare(ref, _dir(tmp_path, "short", wf_00001="The first line."))
    assert (short["first_diff"]["line"], short["first_diff"]["col"]) == (2, 1)
    assert short["first_diff"]["cand"] is None and short["lines_cand"] == 1
    # The offset is the length of what the two share, also when one of them has
    # ended: the line break that only the longer text has is not shared.
    assert short["first_diff"]["offset"] == len("The first line.")
    for a, b, shared in (("a", "a\nb", 1), ("a\nb", "a", 1), ("", "\nx", 0), ("a\nb", "a\nc", 2),
                         ("line1\nline2", "line1\nline2\n\nline4", 11)):
        assert output_kinds.first_difference(a, b)["offset"] == shared
    # An empty candidate differs at the very start and shares nothing.
    empty = CMP.compare(ref, _dir(tmp_path, "empty", wf_00001=""))
    assert (empty["first_diff"]["line"], empty["first_diff"]["col"]) == (1, 1)
    assert empty["similarity"] == 0.0 and empty["chars_cand"] == 0


def test_a_blank_only_change_is_a_difference_with_similarity_below_one(tmp_path):
    res = CMP.compare(_dir(tmp_path, "a", wf_00001="a b"), _dir(tmp_path, "b", wf_00001="a  b"))
    assert res["identical"] is False and res["similarity"] == 0.999999


def test_long_lines_are_cut_around_the_difference():
    line = "x" * 500
    diff = output_kinds.first_difference(line + "A" + line, line + "B" + line)
    assert diff["col"] == 501 and len(diff["ref"]) == output_kinds.EXCERPT_CHARS + 2
    assert diff["ref"].startswith("…") and diff["ref"].endswith("…") and "A" in diff["ref"]


def test_files_pair_by_name_and_a_missing_one_fails(tmp_path):
    ref = _dir(tmp_path, "ref", wf_00001=TEXT, wf_thinking_00001="why")
    same = CMP.compare(ref, _dir(tmp_path, "same", wf_00001=TEXT, wf_thinking_00001="why"))
    assert same["identical"] and same["files_compared"] == 2
    fewer = CMP.compare(ref, _dir(tmp_path, "fewer", wf_00001=TEXT))
    assert fewer["identical"] is False and fewer["similarity"] == 0.999999
    assert not CMP.passes(fewer, {"min_text_similarity": 0.5})  # a file count never passes
    assert "first_diff" not in fewer
    other = CMP.compare(ref, _dir(tmp_path, "other", wf_00001=TEXT, wf_thinking_00001="because"))
    assert other["first_diff"]["file"] == "wf_thinking_00001.txt" and other["similarity"] == 0.0
    # As many files, one under another name: the pair that is left is alike, and
    # says nothing about the file that was never compared.
    renamed = CMP.compare(ref, _dir(tmp_path, "renamed", wf_00001=TEXT, wf_notes_00001="why"))
    assert (renamed["files_compared"], renamed["file_count_cand"]) == (1, 2)
    assert renamed["similarity"] == 0.999999 and "first_diff" not in renamed
    assert not CMP.passes(renamed, {"min_text_similarity": 0.5})
    # No name in common: nothing was compared, whatever the similarity says.
    none = CMP.compare(ref, _dir(tmp_path, "none", a_00001=TEXT, b_00001="why"))
    assert (none["files_compared"], none["similarity"]) == (0, 0.0)
    assert not CMP.passes({**none, "similarity": 0.9}, {"min_text_similarity": 0.5})


def test_errors_instead_of_guesses(tmp_path):
    ref = _dir(tmp_path, "ref", wf_00001=TEXT)
    assert CMP.compare(ref, tmp_path / "missing") == {
        "kind": "text", "error": "no text files to compare (ref=1, cand=0)"}
    (tmp_path / "png").mkdir()
    (tmp_path / "png" / "wf_00001_.png").write_bytes(b"\x89PNG")
    assert "no text files" in CMP.compare(ref, tmp_path / "png")["error"]
    big = _dir(tmp_path, "big", wf_00001="x" * (output_kinds.TEXT_MAX_BYTES + 1))
    assert CMP.compare(ref, big)["error"].startswith("text output over")
    assert not CMP.passes({"kind": "text", "error": "x"}, None)


def test_a_similarity_floor_is_the_only_limit_and_only_when_the_entry_sets_it():
    entry = {"thresholds": {"min_text_similarity": 0.9}}
    assert CMP.thresholds(entry, {"thresholds": {"max_mean_mse": 2.0}}, None) == {
        "min_text_similarity": 0.9}
    near = {"identical": False, "similarity": 0.95, "file_count_ref": 1, "file_count_cand": 1}
    assert CMP.passes(near, {"min_text_similarity": 0.9})
    assert not CMP.passes({**near, "similarity": 0.85}, {"min_text_similarity": 0.9})
    assert not CMP.passes(near, None) and not CMP.passes(near, {"min_text_similarity": 1.0})
    # A reference that differs from its own second run loosens nothing by itself.
    noisy = {"kind": "text", "identical": False, "similarity": 0.5, "measured": True}
    assert CMP.thresholds({"thresholds": None}, {}, noisy) is None


def test_the_sha_is_the_text_files_own_or_one_over_all_of_them():
    one = {"outputs": [{"filename": "wf_00001.txt", "sha256": "a" * 64},
                       {"filename": "wf_00001_.png", "sha256": "b" * 64}]}
    assert output_kinds.text_sha(one) == CMP.primary_sha(one) == "a" * 64
    two = {"outputs": [{"filename": "wf_thinking_00001.txt", "sha256": "c" * 64}, *one["outputs"]]}
    listing = f"{'a' * 64}  wf_00001.txt\n{'c' * 64}  wf_thinking_00001.txt\n"
    assert output_kinds.text_sha(two) == hashlib.sha256(listing.encode()).hexdigest()
    assert output_kinds.text_sha({"outputs": []}) is None and output_kinds.text_sha({}) is None
    # A record that is not shaped like one identifies nothing.
    assert output_kinds.text_sha({"outputs": "nope"}) is None
    assert output_kinds.text_sha(None) is None
    assert output_kinds.text_sha({"outputs": [{"filename": "wf_00001.txt", "sha256": 7}]}) is None
    assert output_kinds.text_sha({"outputs": ["wf_00001.txt", {"sha256": "a"}]}) is None


def test_a_text_the_worker_did_not_send_is_an_error_not_a_change():
    rec = {"outputs": [{"filename": "wf_00001.txt", "sha256": "a", "truncated": True}]}
    assert CMP.transfer_error(rec) == (
        "output not transferred, over the worker's inline size limit: wf_00001.txt")
    assert CMP.transfer_error({"outputs": [{"filename": "wf_00001.txt"}]}) is None


def test_which_references_may_be_approved(tmp_path):
    ok = {"kind": "text", "identical": True, "chars_ref": 10, "measured": True}
    assert CMP.blessable(ok, {}) is None and CMP.blessable(None, {}) is None
    assert "could not be compared" in CMP.blessable({"kind": "text", "error": "no text files"}, {})
    assert "empty" in CMP.blessable({**ok, "chars_ref": 0}, {})
    noisy = {**ok, "identical": False, "similarity": 0.8}
    assert "different text (similarity 0.8)" in CMP.blessable(noisy, {})
    assert CMP.blessable(noisy, {"thresholds": {"min_text_similarity": 0.7}}) is None
    # A limit opens the gate only as far as the second run passes under it.
    for limit in (0.9, 1, 1.0):
        why = CMP.blessable(noisy, {"thresholds": {"min_text_similarity": limit}})
        assert f"does not pass thresholds.min_text_similarity {limit:g} of its" in why
    files = {**noisy, "similarity": 0.999999, "file_count_ref": 2, "file_count_cand": 1}
    assert "did not write the same text files" in CMP.blessable(
        files, {"thresholds": {"min_text_similarity": 0.7}})
    # A floor taken over from an earlier reference was not measured: it gates nothing.
    assert CMP.blessable({**noisy, "measured": False}, {}) is None
    assert CMP.zero_floor() == {"kind": "text", "identical": True, "similarity": 1.0}
    # What such a floor says about the text itself is read from the promoted outputs.
    assert CMP.describe(_dir(tmp_path, "one", wf_00001=TEXT)) == {
        "files_compared": 1, "file_count_ref": 1, "file_count_cand": 1, "chars_ref": len(TEXT),
        "chars_cand": len(TEXT), "lines_ref": 3, "lines_cand": 3}
    assert "empty" in CMP.blessable(CMP.describe(_dir(tmp_path, "empty", wf_00001="")), {})
    assert "no text files" in CMP.describe(tmp_path / "missing")["error"]
    # An empty file next to one with text: the totals hide it, the gate does not.
    two = _dir(tmp_path, "two", wf_00001="", wf_thinking_00001="why")
    floor = CMP.compare(two, two)
    assert (floor["identical"], floor["chars_ref"], floor["files_empty_ref"]) == (True, 3, 1)
    assert CMP.blessable({**floor, "measured": True}, {}) == "it has an empty text file"
    assert CMP.blessable({**TEXT_FLOOR, **CMP.describe(two), "measured": False}, {}) == (
        "it has an empty text file")
    # The floor of an image golden says nothing about a text: the entry changed its kind.
    image = {"identical": True, "mean_mse": 0.0, "measured": True}
    assert CMP.blessable(image, {}) == (
        "its noise floor is not a text comparison, so it is a golden of another kind")
    # A similarity that is not a share of words is named as what it is.
    assert "(similarity 0.8 by lines)" in CMP.blessable({**noisy, "similarity_basis": "lines"}, {})
    assert "(similarity 0.8 by the shared start)," in CMP.blessable(
        {**noisy, "similarity_basis": "prefix"}, {"thresholds": {"min_text_similarity": 0.9}})
    # Writing a reference over an earlier one removes no file: what would stay behind.
    one = _dir(tmp_path, "one", wf_00001=TEXT)
    assert CMP.left_behind(two, one) == ["wf_thinking_00001.txt"]
    assert CMP.left_behind(one, two) == [] == CMP.left_behind(tmp_path / "missing", one)


def test_word_diff_marks_what_was_removed_and_added():
    assert output_kinds.word_diff("a b c", "a x c") == "a [-b-]{+x+} c"
    assert output_kinds.word_diff("a b", "a b") == "a b"
    assert output_kinds.word_diff("", "new") == "{+new+}"
    # Changes with only blanks between them are one change, as in git's word diff.
    assert output_kinds.word_diff("a bright red hat", "a deep crimson hat") == (
        "a [-bright red-]{+deep crimson+} hat")
    assert output_kinds.word_diff("one two\nthree", "uno dos\ntres") == (
        "[-one two\nthree-]{+uno dos\ntres+}")
    assert output_kinds.word_diff("a b", "a  b") == "a[- -]{+  +}b"
    long = "\n".join(f"line {i}" for i in range(output_kinds.DIFF_TOKENS_MAX))
    assert output_kinds.word_diff(long, long + "\nmore").splitlines()[:2] == ["--- reference",
                                                                           "+++ candidate"]
    assert output_kinds.wrap_lines("word " * 40).count("\n") == 1
    assert output_kinds.wrap_lines("a\n\nb") == "a\n\nb"


def test_long_texts_are_scored_by_lines_and_very_long_ones_by_their_shared_start():
    """difflib is quadratic on repeated lines: a model stuck in a loop must not
    hang the run."""
    words = " ".join(f"w{i}" for i in range(output_kinds.WORDS_MAX + 1))
    assert output_kinds.similarity(words + "\nlast", words + "\nother") == (0.5, "lines")
    assert output_kinds.similarity("a b", "a c") == (0.5, "words")
    loop = "again\n" * (output_kinds.LINES_MAX + 1)
    ratio, basis = output_kinds.similarity(loop + "end", loop + "END")
    assert basis == "prefix" and 0.999 < ratio <= 0.999999
    assert output_kinds.similarity("x\n" + loop, "y\n" + loop) == (0.0, "prefix")
    assert output_kinds.similarity(loop, loop) == (1.0, "words")


def test_a_match_that_would_take_too_long_is_stopped_by_its_steps(monkeypatch):
    """No limit on the length keeps difflib quick: its time grows with the cube
    of the length when two texts share little more than one frequent token.
    The bound counts steps, not seconds, so a result never depends on the machine."""
    ref = "\n".join(f"| r{i} | rv{i} |" for i in range(40))    # a table, every cell changed
    cand = "\n".join(f"| c{i} | cv{i} |" for i in range(40))
    a, b = ref.split(), cand.split()
    # Within the bound, the ratio and the marks are difflib's own.
    plain = difflib.SequenceMatcher(None, a, b, autojunk=False)
    assert output_kinds.similarity(ref, cand) == (round(plain.ratio(), 6), "words")
    assert plain.ratio() == 0.6
    bounded = output_kinds._Matcher(a, b)
    assert bounded.get_opcodes() == plain.get_opcodes()
    assert bounded.get_matching_blocks() == [tuple(m) for m in plain.get_matching_blocks()]
    assert output_kinds.word_diff(ref, cand).startswith("| [-r0-]{+c0+} | [-rv0-]{+cv0+} |\n")
    # Past it: the next basis. No line is common here, which is quick to see.
    monkeypatch.setattr(output_kinds, "MATCH_STEPS_MAX", 5000)
    with pytest.raises(output_kinds._TooSlow):
        output_kinds._Matcher(a, b).ratio()
    assert output_kinds.similarity(ref, cand) == (0.0, "lines")
    assert output_kinds.word_diff(ref, cand).splitlines()[:2] == ["--- reference", "+++ candidate"]
    # Too slow by words and by lines: what the two share from the start.
    ref = "\n".join(f"same\nr{i}" for i in range(80))
    cand = "\n".join(f"same\nc{i}" for i in range(80))
    assert output_kinds.similarity(ref, cand) == (round(len("same\n") / len(ref), 6), "prefix")
    assert not CMP.passes({"identical": False, "similarity": 0.0, "file_count_ref": 1,
                           "file_count_cand": 1}, {"min_text_similarity": 0.5})


def test_a_similarity_by_lines_is_never_above_what_the_words_allow(monkeypatch):
    """Twenty equal lines and one long line with every item changed: 95 % of the
    lines, and two thirds of the words at most. When the word match is stopped,
    the step down to lines must not pass a limit that the words fail."""
    monkeypatch.setattr(output_kinds, "MATCH_STEPS_MAX", 5000)
    head = "\n".join(f"step {i} done" for i in range(20))
    ref = head + "\n" + " ".join(f"r{i} ," for i in range(60))
    cand = head + "\n" + " ".join(f"c{i} ," for i in range(60))
    with pytest.raises(output_kinds._TooSlow):
        output_kinds._Matcher(ref.split(), cand.split()).ratio()
    by_lines = difflib.SequenceMatcher(None, ref.split("\n"), cand.split("\n")).ratio()
    by_words = difflib.SequenceMatcher(None, ref.split(), cand.split(), autojunk=False)
    assert round(by_lines, 6) == 0.952381 and round(by_words.quick_ratio(), 6) == 0.666667
    assert output_kinds.similarity(ref, cand) == (0.666667, "lines")
    one = output_kinds.compare_texts(ref, cand)
    assert not CMP.passes({**one, "file_count_ref": 1, "file_count_cand": 1},
                          {"min_text_similarity": 0.9})
    # Where the lines say less than the words could, the number is the lines' own.
    assert output_kinds.similarity(ref, cand.replace("step", "STEP")) == (0.0, "lines")


def test_a_line_diff_is_difflibs_own_and_is_held_to_the_step_bound(monkeypatch):
    """difflib.unified_diff matches lines with no bound: its time grows with
    the cube of the line count when many lines recur, each too rare for
    difflib to set aside. The line diff of a long text goes through the
    bounded matcher instead, and reads the same as long as the match ends."""
    def plain(a, b):
        return "\n".join(difflib.unified_diff(a.split("\n"), b.split("\n"), "reference",
                                              "candidate", lineterm="", n=1))

    text = "\n".join(f"line {i}" for i in range(30))
    loop = "again\n" * 300  # over 1 % of 200 lines or more: difflib sets such a line aside
    for a, b in ((text, text.replace("line 0", "LINE 0")), (text, text + "\nmore"),
                 (text, "new\n" + text), ("", text), (text, ""), ("one", "two"), ("a", "a\nb"),
                 (text, text.replace("line 12\n", "").replace("line 20", "x\ny")),
                 ("first\n" + loop + "end", "FIRST\n" + loop + "end"),
                 (loop + "old\n" + loop, loop + loop), ("a\nb\n" * 150, "b\na\n" * 150)):
        assert output_kinds.line_diff(a, b) == plain(a, b) != ""
        assert output_kinds.line_diff(a, b).startswith(output_kinds.LINE_DIFF_HEAD)
    assert output_kinds.line_diff(text, text) == plain(text, text) == ""

    # Past the bound no line is matched. What lies between the start and the end
    # that the two texts share is one change.
    monkeypatch.setattr(output_kinds, "MATCH_STEPS_MAX", 50)
    ref = "\n".join(["top", *(f"same\nr{i}" for i in range(5)), "end"])
    cand = "\n".join(["top", *(f"same\nc{i}" for i in range(5)), "end"])
    with pytest.raises(output_kinds._TooSlow):
        output_kinds._Matcher(ref.split("\n"), cand.split("\n"), autojunk=True).get_opcodes()
    gone = ["r0", "same", "r1", "same", "r2", "same", "r3", "same", "r4"]
    assert output_kinds.line_diff(ref, cand).split("\n") == [
        "--- reference", "+++ candidate", "@@ -3,9 +3,9 @@", *("-" + line for line in gone),
        *("+" + line.replace("r", "c") for line in gone)]
    assert output_kinds.word_diff(ref, cand) == output_kinds.line_diff(ref, cand)
    # One changed line in a text that repeats itself is still one changed line.
    loop = "again\n" * 30
    assert output_kinds.line_diff(loop + "old\n" + loop, loop + "new\n" + loop).split("\n") == [
        "--- reference", "+++ candidate", "@@ -31 +31 @@", "-old", "+new"]
    assert output_kinds.line_diff(loop + "old\n" + loop, loop + loop).split("\n") == [
        "--- reference", "+++ candidate", "@@ -31 +30,0 @@", "-old"]
    assert output_kinds.line_diff(loop, loop) == ""
    # A long change is shown from its start, and its header says that this is not all.
    monkeypatch.setattr(output_kinds, "UNMATCHED_LINES", 3)
    assert output_kinds.line_diff(ref, cand).split("\n") == [
        "--- reference", "+++ candidate",
        "@@ -3,3 +3,3 @@ too long to match line by line, only these lines are shown",
        "-r0", "-same", "-r1", "+c0", "+same", "+c1"]


def test_what_a_log_line_may_carry_and_how_a_similarity_is_shown():
    metrics = {"kind": "text", "identical": False, "similarity": 0.5,
               "first_diff": {"file": "wf_00001.txt", "line": 2, "col": 3, "offset": 9,
                              "ref": "model output", "cand": "other model output"}}
    assert output_kinds.loggable(metrics) == {
        "kind": "text", "identical": False, "similarity": 0.5,
        "first_diff": {"file": "wf_00001.txt", "line": 2, "col": 3, "offset": 9}}
    assert "model output" in metrics["first_diff"]["ref"]  # the metrics themselves are untouched
    image = {"identical": False, "mean_mse": 3.4}
    assert output_kinds.loggable(image) is image
    # Texts that differ never read as fully alike.
    assert output_kinds.percent(0.864865) == "86.5" and output_kinds.percent(0.0) == "0.0"
    assert output_kinds.percent(0.999999) == "over 99.9" and output_kinds.percent(1.0) == "100.0"
    # A share of lines or of the shared start is said next to the number. The basis
    # is read from results: one that is not known adds nothing.
    assert output_kinds.basis_note({"similarity_basis": "lines"}) == " by lines"
    assert output_kinds.basis_note({"similarity_basis": "prefix"}) == " by the shared start"
    for other in ("words", "<b>lines</b>", ["lines"], 3, None):
        assert output_kinds.basis_note({"similarity_basis": other}) == ""
    assert output_kinds.basis_note({}) == output_kinds.basis_note(None) == ""


def test_the_image_comparator_is_the_old_functions(tmp_path, monkeypatch):
    image = run_regression.COMPARATORS["image"]
    assert run_regression.comparator_for({}) is image and image.kind == "image"
    assert run_regression.comparator_for({"output_kind": "text"}).kind == "text"
    calls = []
    monkeypatch.setattr(run_regression.compare, "compare_dirs",
                        lambda ref, cand: calls.append(("dirs", ref, cand)) or {"identical": True})
    monkeypatch.setattr(run_regression.compare, "compare_with_figures",
                        lambda *a, **kw: calls.append(("figures", a, kw)) or {"m": 1})
    assert image.compare("r", "c") == {"identical": True}
    assert image.render("r", "c", "f", "golden", "golden v1", {"x": 1}, figures=False) == {"m": 1}
    assert calls == [("dirs", "r", "c"),
                     ("figures", ("r", "c", "f", "golden", "golden v1"),
                      {"metrics": {"x": 1}, "figures": False})]
    defaults = {"thresholds": {"max_mean_mse": 2.0, "min_mean_psnr_db": 45.0,
                               "max_pct_pixels_changed": 1.0}}
    assert image.thresholds({}, defaults, None) == run_regression.derive_thresholds(
        {}, defaults, None)
    assert image.passes({"identical": True}, defaults["thresholds"])
    rec = {"outputs": [{"filename": "a_00001_.png", "sha256": "p"},
                       {"filename": "a.txt", "sha256": "t"}]}
    assert image.primary_sha(rec) == "p" and image.transfer_error(rec) is None
    assert image.blessable({"error": "x"}, {}) is None and not image.bless_gate
    assert image.left_behind(tmp_path / "old", tmp_path / "new") == []
    assert image.zero_floor() == {"identical": True, "mean_mse": 0.0, "mean_psnr_db": None,
                                  "min_psnr_db": None, "max_abs_diff": 0,
                                  "mean_pct_pixels_changed": 0.0}


class _Store:
    """A results store for one text workflow: a golden, and maybe a previous run."""

    def __init__(self, golden_text, previous_text=None, previous_verdict="pass"):
        self.golden_text, self.previous_text = golden_text, previous_text
        self.previous_verdict = previous_verdict
        self.published = None

    def _write(self, dest, text):
        Path(dest).mkdir(parents=True, exist_ok=True)
        (Path(dest) / "wf_00001.txt").write_bytes(text.encode("utf-8"))
        return 1

    def latest_pointer(self, branch):
        return {"commit": "prev"} if self.previous_text is not None else None

    def fetch_run_outputs(self, branch, commit, wf_id, dest):
        return self._write(dest, self.previous_text)

    def run_summary(self, branch, commit):
        return {"workflows": {"wf": {"verdict": self.previous_verdict, "golden_tag": "v1"}}}

    def golden_current(self, wf_id):
        return {"tag": "v1"} if self.golden_text is not None else None

    def fetch_golden_outputs(self, wf_id, tag, dest):
        return self._write(dest, self.golden_text)

    def noise_floor(self, wf_id, tag):
        return {"kind": "text", "identical": True, "measured": True}

    def publish_workflow_run(self, branch, commit, wf_id, local_dir):
        self.published = sorted(p.relative_to(local_dir).as_posix()
                                for p in Path(local_dir).rglob("*") if p.is_file())


def _run(tmp_path, monkeypatch, store, text, truncated=False, thresholds=None):
    def job(client, wf_id, workflow, workdir=None, **kw):
        out = Path(workdir) / "outputs"
        out.mkdir(parents=True)
        if not truncated:
            (out / "wf_00001.txt").write_bytes(text.encode("utf-8"))
        return {"status": "ok", "outputs_dir": str(out), "timings": {"prompt_exec_s": 3.0},
                "outputs": [{"filename": "wf_00001.txt", "bytes": len(text), "sha256": "s" * 64,
                             **({"truncated": True} if truncated else {})}]}

    monkeypatch.setattr(run_regression, "run_workflow_job", job)
    monkeypatch.setattr(run_regression, "RunPodClient", lambda endpoint_id: None)
    monkeypatch.setattr(run_regression.packs, "records_for", lambda entry, root: [])
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "wf_api.json").write_text("{}", encoding="utf-8")
    entry = {"workflow_path": "wf_api.json", "output_kind": "text", "models": [],
             "thresholds": thresholds}
    defaults = {"timeout_s": 600, "comfy_flags": [], "thresholds": {"max_mean_mse": 2.0}}
    args = types.SimpleNamespace(workdir=str(tmp_path / "work"), endpoint_id="ep", lane_id="lane",
                                 commit="new", repo_url=None, branch="master", skip_publish=False)
    return run_regression.process_workflow("wf", entry, defaults, args, tmp_path, store)


def test_a_text_workflow_that_matches_its_reference_passes(tmp_path, monkeypatch):
    store = _Store(TEXT, previous_text=TEXT)
    cell = _run(tmp_path, monkeypatch, store, TEXT + "\n")
    assert (cell["verdict"], cell["detail"], cell["output_kind"]) == ("pass", None, "text")
    assert cell["vs_golden"]["kind"] == "text" and cell["vs_golden"]["identical"] is True
    assert cell["vs_previous"]["identical"] is True
    # No image limits, no figures, no thumbnail; the sha is the text file's.
    assert cell["thresholds_used"] is None and cell["figures"] is False
    assert cell["thumbnail"] is None and cell["output_sha256"] == "s" * 64
    assert store.published == ["comparison.json", "outputs/wf_00001.txt", "run.json"]
    comparison = json.loads((tmp_path / "work" / "wf" / "comparison.json").read_text())
    # The one new key comes last, in the file and in the cell: nothing before it moved.
    assert comparison["output_kind"] == "text" and list(comparison)[-1] == "output_kind"
    assert list(cell)[-2:] == ["custom_nodes", "output_kind"] and len(cell) == 23


def test_a_changed_text_fails_as_new_then_as_inherited(tmp_path, monkeypatch):
    changed = TEXT.replace("seven", "eight")
    new = _run(tmp_path / "a", monkeypatch, _Store(TEXT, previous_text=TEXT), changed)
    assert (new["verdict"], new["detail"]) == ("fail", "new_drift")
    assert new["vs_golden"]["first_diff"]["line"] == 2 and new["figures"] is False
    again = _run(tmp_path / "b", monkeypatch,
                 _Store(TEXT, previous_text=changed, previous_verdict="fail"), changed)
    assert (again["verdict"], again["detail"]) == ("fail", "inherited")
    assert again["vs_previous"]["identical"] is True and again["figures"] is False


def test_a_changed_text_passes_only_within_a_similarity_limit_the_entry_sets(
        tmp_path, monkeypatch):
    changed = TEXT.replace("seven", "eight")
    near = _run(tmp_path / "a", monkeypatch, _Store(TEXT, previous_text=TEXT), changed,
                thresholds={"min_text_similarity": 0.8})
    assert (near["verdict"], near["detail"]) == ("pass", None)
    assert near["thresholds_used"] == {"min_text_similarity": 0.8}
    assert near["vs_golden"]["identical"] is False and near["figures"] is False
    far = _run(tmp_path / "b", monkeypatch, _Store(TEXT, previous_text=TEXT), changed,
               thresholds={"min_text_similarity": 0.99})
    assert (far["verdict"], far["detail"]) == ("fail", "new_drift")
    # An empty text is a text: it differs from the reference at its first character.
    empty = _run(tmp_path / "c", monkeypatch, _Store(TEXT), "")
    assert empty["verdict"] == "fail" and empty["vs_golden"]["chars_cand"] == 0
    assert (empty["vs_golden"]["first_diff"]["line"], empty["vs_golden"]["similarity"]) == (1, 0.0)


def test_a_text_workflow_without_a_reference_and_one_whose_text_was_not_sent(tmp_path, monkeypatch):
    cell = _run(tmp_path / "a", monkeypatch, _Store(None), TEXT)
    assert (cell["verdict"], cell["vs_golden"]) == ("no_baseline", None)
    assert cell["output_kind"] == "text"
    lost = _run(tmp_path / "b", monkeypatch, _Store(TEXT, previous_text=TEXT), TEXT,
                truncated=True)
    assert lost["verdict"] == "fail" and lost["vs_golden"]["kind"] == "text"
    assert "not transferred" in lost["vs_golden"]["error"]
    # The previous run is not compared either: "no text files" would blame the run.
    assert lost["vs_previous"] == lost["vs_golden"]


def test_an_unknown_kind_stops_the_run_before_any_gpu_job(tmp_path, monkeypatch):
    import index_builder
    import storage
    lanes = {"primary": "main", "lanes": {"main": {"cadence": "per-commit", "layout": 1,
                                                   "prefix": "regression", "endpoint_id": "ep"}}}
    monkeypatch.setattr(index_builder, "load_lanes", lambda path=None: lanes)
    manifest = tmp_path / "manifest" / "workflows.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({
        "defaults": {"timeout_s": 600, "comfy_flags": [], "thresholds": {}},
        "workflows": {"wf": {"enabled": True, "workflow_path": "x", "models": [],
                             "output_kind": "audio"}}}))

    class Store:
        root = "regression"

        def latest_pointer(self, branch):
            return None

        def download_json(self, path):
            return None

        published_summary = attempt_marker = lambda self, branch, commit: None

    monkeypatch.setattr(storage, "from_args", lambda args: Store())
    monkeypatch.setattr(run_regression, "process_workflow",
                        lambda *a: pytest.fail("no GPU job for a kind nothing can compare"))
    monkeypatch.setattr(sys, "argv", ["run_regression.py", "--manifest", str(manifest), "--commit",
                                      "new", "--branch", "master", "--skip-publish"])
    with pytest.raises(SystemExit, match=r"\[wf\] unknown output_kind 'audio'"):
        run_regression.main()
