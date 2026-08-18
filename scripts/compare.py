"""Fixed-seed output comparison, ported from the pr15056 QA harness.

Compares the PNGs of a candidate run against a reference run (golden baseline
or previous master run): per-frame MSE, PSNR (mean/min), max abs channel diff,
% pixels changed, plus an `identical` flag. Also renders a side-by-side strip
and an abs-diff heatmap of the worst frame for the dashboard.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def list_pngs(d: Path) -> list[Path]:
    return sorted(p for p in Path(d).rglob("*.png"))


def compare_dirs(ref_dir: Path, cand_dir: Path) -> dict:
    ref_frames, cand_frames = list_pngs(ref_dir), list_pngs(cand_dir)
    if not ref_frames or not cand_frames:
        return {"error": f"no PNGs to compare (ref={len(ref_frames)}, cand={len(cand_frames)})"}
    n = min(len(ref_frames), len(cand_frames))
    per_frame_mse, per_frame_psnr = [], []
    max_abs = 0
    pct_changed_acc = 0.0
    identical = len(ref_frames) == len(cand_frames)
    for i in range(n):
        a = np.asarray(Image.open(ref_frames[i]).convert("RGB"), dtype=np.int16)
        b = np.asarray(Image.open(cand_frames[i]).convert("RGB"), dtype=np.int16)
        if a.shape != b.shape:
            return {"error": f"shape mismatch on frame {i}: {a.shape} vs {b.shape}"}
        diff = np.abs(a - b)
        mse = float(np.mean((a - b).astype(np.float64) ** 2))
        per_frame_mse.append(mse)
        per_frame_psnr.append(10 * math.log10(255.0 ** 2 / mse) if mse > 0 else float("inf"))
        max_abs = max(max_abs, int(diff.max()))
        pct_changed_acc += float((diff.max(axis=2) > 0).mean()) * 100
        if mse > 0:
            identical = False
    finite = [p for p in per_frame_psnr if math.isfinite(p)]
    return {
        "frames_compared": n,
        "frame_count_ref": len(ref_frames),
        "frame_count_cand": len(cand_frames),
        "identical": identical,
        "mean_mse": round(float(np.mean(per_frame_mse)), 6),
        "mean_psnr_db": round(float(np.mean(finite)), 2) if finite else None,
        "min_psnr_db": round(min(finite), 2) if finite else None,
        "max_abs_diff": max_abs,
        "mean_pct_pixels_changed": round(pct_changed_acc / n, 3),
        "worst_frame_index": int(np.argmax(per_frame_mse)),
    }


def save_figures(ref_dir: Path, cand_dir: Path, figures_dir: Path, label: str,
                 ref_name: str, cand_name: str, worst_idx: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    ref_frames, cand_frames = list_pngs(ref_dir), list_pngs(cand_dir)
    n = min(len(ref_frames), len(cand_frames))
    picks = sorted({0, n // 2, n - 1})

    fig, axes = plt.subplots(3, len(picks), figsize=(4.2 * len(picks), 8.5), squeeze=False)
    for col, fi in enumerate(picks):
        a = np.asarray(Image.open(ref_frames[fi]).convert("RGB"))
        b = np.asarray(Image.open(cand_frames[fi]).convert("RGB"))
        d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
        axes[0][col].imshow(a); axes[0][col].set_title(f"{ref_name} frame {fi}", fontsize=9)
        axes[1][col].imshow(b); axes[1][col].set_title(f"{cand_name} frame {fi}", fontsize=9)
        im = axes[2][col].imshow(d, cmap="inferno", vmin=0, vmax=max(1, d.max()))
        axes[2][col].set_title(f"abs diff (max={d.max()})", fontsize=9)
        fig.colorbar(im, ax=axes[2][col], fraction=0.03)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"{ref_name} vs {cand_name} (seed-locked)", fontsize=12)
    fig.tight_layout()
    fig.savefig(figures_dir / f"side_by_side_{label}.png", dpi=110)
    plt.close(fig)

    a = np.asarray(Image.open(ref_frames[worst_idx]).convert("RGB"))
    b = np.asarray(Image.open(cand_frames[worst_idx]).convert("RGB"))
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(d, cmap="inferno", vmin=0, vmax=max(1, d.max()))
    ax.set_title(f"worst frame ({worst_idx}) abs diff: {ref_name} vs {cand_name}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(figures_dir / f"diff_heatmap_{label}.png", dpi=110)
    plt.close(fig)


def compare_with_figures(ref_dir: Path, cand_dir: Path, figures_dir: Path, label: str,
                         ref_name: str, cand_name: str) -> dict:
    res = compare_dirs(ref_dir, cand_dir)
    if not res.get("error") and not res.get("identical"):
        try:
            save_figures(ref_dir, cand_dir, figures_dir, label, ref_name, cand_name,
                         res.get("worst_frame_index") or 0)
        except Exception as e:
            res["figures_error"] = str(e)[:500]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--label", default="cmp")
    ap.add_argument("--ref-name", default="reference")
    ap.add_argument("--cand-name", default="candidate")
    ap.add_argument("--figures-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.figures_dir:
        res = compare_with_figures(Path(args.ref), Path(args.cand), Path(args.figures_dir),
                                   args.label, args.ref_name, args.cand_name)
    else:
        res = compare_dirs(Path(args.ref), Path(args.cand))
    text = json.dumps(res, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
