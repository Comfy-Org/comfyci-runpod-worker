"""compare.py: metrics and figure determinism."""
import numpy as np
from PIL import Image

import compare


def _png(path, seed, size=(32, 24), delta=0):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    arr[0, 0, 0] = delta  # pixel (0,0) red channel is the controlled difference
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def test_identical_frames(tmp_path):
    _png(tmp_path / "a" / "x_00001_.png", 1)
    _png(tmp_path / "b" / "x_00001_.png", 1)
    res = compare.compare_dirs(tmp_path / "a", tmp_path / "b")
    assert res["identical"] is True
    assert res["mean_mse"] == 0.0 and res["max_abs_diff"] == 0


def test_changed_frame_metrics(tmp_path):
    _png(tmp_path / "a" / "x_00001_.png", 1)
    _png(tmp_path / "b" / "x_00001_.png", 1, delta=100)
    res = compare.compare_dirs(tmp_path / "a", tmp_path / "b")
    assert res["identical"] is False
    assert res["max_abs_diff"] == 100
    assert 0 < res["mean_pct_pixels_changed"] < 1


def test_figures_are_byte_identical_for_identical_candidates(tmp_path):
    """Two candidates with the same pixels must render the same figure bytes,
    so git stores them once regardless of which commit produced them."""
    _png(tmp_path / "ref" / "x_00001_.png", 1)
    _png(tmp_path / "c1" / "x_00001_.png", 1, delta=50)
    _png(tmp_path / "c2" / "x_00001_.png", 1, delta=50)
    r1 = compare.compare_with_figures(tmp_path / "ref", tmp_path / "c1", tmp_path / "f1",
                                      "golden", "golden v0.1")
    r2 = compare.compare_with_figures(tmp_path / "ref", tmp_path / "c2", tmp_path / "f2",
                                      "golden", "golden v0.1")
    assert r1 == r2 and not r1.get("identical")
    for name in ("side_by_side_golden.png", "diff_heatmap_golden.png"):
        assert (tmp_path / "f1" / name).read_bytes() == (tmp_path / "f2" / name).read_bytes()


def test_figures_can_be_skipped_and_metrics_reused(tmp_path):
    _png(tmp_path / "ref" / "x_00001_.png", 1)
    _png(tmp_path / "c" / "x_00001_.png", 1, delta=50)
    metrics = compare.compare_dirs(tmp_path / "ref", tmp_path / "c")
    res = compare.compare_with_figures(tmp_path / "ref", tmp_path / "c", tmp_path / "f",
                                       "golden", "golden v0.1", metrics=metrics, figures=False)
    assert res == metrics
    assert not (tmp_path / "f").exists()
