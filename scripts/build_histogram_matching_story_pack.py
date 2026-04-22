"""Build slide-ready figures for the histogram-matching story.

This script reuses the example outputs already present in this repository.

Outputs:
- 01_blue_channel_hist_cdf_lut.png
- 02_cohort_pooling_schematic.png
- 03_single_vs_multi_examples.png
- 04_distribution_dashboard.png
- story_pack_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Rectangle
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC_DIR))

from vahadane_trichrome import HistogramMatchingNormalizer
from vahadane_trichrome import build_histogram_specification_lut
from vahadane_trichrome import get_tissue_mask
from vahadane_trichrome import load_rgb_uint8


DEFAULT_SINGLE_METADATA = (
    PROJECT_ROOT / "outputs" / "histogram_matching_single_target_from_multi_example" / "run_metadata.json"
)
DEFAULT_MULTI_METADATA = (
    PROJECT_ROOT / "outputs" / "histogram_matching_multi_source_target_example" / "run_metadata.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "histogram_matching_story_pack"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_existing_path(path_or_name: str | Path, search_roots: list[Path]) -> Path:
    candidate = Path(path_or_name)
    if candidate.exists():
        return candidate

    basename = candidate.name
    for root in search_roots:
        matches = list(root.rglob(basename))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not resolve path: {path_or_name}")


def _to_short_label(value: str) -> str:
    stem = Path(value).stem
    return stem.replace("BU_level4_", "").replace("NW_level4_", "")[:28]


def _hist_and_cdf(values: np.ndarray, levels: int = 256) -> tuple[np.ndarray, np.ndarray]:
    hist = np.bincount(np.asarray(values, dtype=np.uint8).ravel(), minlength=levels).astype(np.float64)
    cdf = hist.cumsum()
    cdf /= cdf[-1]
    return hist, cdf


def _masked_channel_values(image: np.ndarray, channel_index: int, luminosity_threshold: float) -> np.ndarray:
    mask = get_tissue_mask(image, luminosity_threshold=luminosity_threshold)
    return image[..., channel_index][mask]


def _make_histogram_math_figure(
    source: np.ndarray,
    target: np.ndarray,
    output_path: Path,
    *,
    channel_index: int = 2,
    luminosity_threshold: float = 0.65,
) -> dict[str, float]:
    source_values = _masked_channel_values(source, channel_index, luminosity_threshold)
    target_values = _masked_channel_values(target, channel_index, luminosity_threshold)
    lut = build_histogram_specification_lut(source_values.reshape(1, -1), target_values.reshape(1, -1))

    source_hist, source_cdf = _hist_and_cdf(source_values)
    target_hist, target_cdf = _hist_and_cdf(target_values)
    x = np.arange(256)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(x, source_hist / source_hist.sum(), color="#1f77b4", linewidth=2, label="source BU")
    axes[0].plot(x, target_hist / target_hist.sum(), color="#d62728", linewidth=2, label="target NW")
    axes[0].set_title("Blue-channel histogram")
    axes[0].set_xlabel("pixel value")
    axes[0].set_ylabel("density")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend(loc="best")

    axes[1].plot(x, source_cdf, color="#1f77b4", linewidth=2, label="source CDF")
    axes[1].plot(x, target_cdf, color="#d62728", linewidth=2, label="target CDF")
    for quantile in (0.25, 0.5, 0.75):
        source_intensity = int(np.searchsorted(source_cdf, quantile))
        target_intensity = int(lut[source_intensity])
        axes[1].axhline(quantile, color="0.7", linestyle="--", linewidth=0.8)
        axes[1].axvline(source_intensity, color="#1f77b4", linestyle=":", linewidth=0.9)
        axes[1].axvline(target_intensity, color="#d62728", linestyle=":", linewidth=0.9)
    axes[1].set_title("CDFs: percentile matching")
    axes[1].set_xlabel("pixel value")
    axes[1].set_ylabel("cumulative probability")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(loc="lower right")

    axes[2].plot(x, lut, color="#2ca02c", linewidth=2.2, label="learned LUT")
    axes[2].plot(x, x, color="0.5", linestyle="--", linewidth=1.0, label="identity")
    axes[2].set_title("Lookup table")
    axes[2].set_xlabel("source intensity")
    axes[2].set_ylabel("mapped target intensity")
    axes[2].grid(True, alpha=0.2)
    axes[2].legend(loc="best")

    fig.suptitle("Histogram matching as percentile transfer in one channel", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    return {
        "source_blue_mean": float(np.mean(source_values)),
        "target_blue_mean": float(np.mean(target_values)),
        "mapped_blue_mean": float(np.mean(lut[source_values])),
    }


def _thumbnail(ax, image: np.ndarray, title: str | None = None) -> None:
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)


def _make_cohort_pooling_figure(
    source_paths: list[Path],
    target_paths: list[Path],
    matched_paths: list[Path],
    output_path: Path,
) -> None:
    sources = [load_rgb_uint8(path) for path in source_paths]
    targets = [load_rgb_uint8(path) for path in target_paths]
    matched = [load_rgb_uint8(path) for path in matched_paths]

    fig = plt.figure(figsize=(16, 9))
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.axis("off")

    source_box = Rectangle((0.04, 0.56), 0.26, 0.34, facecolor="#f4f8ff", edgecolor="#7aa6d8", linewidth=2)
    target_box = Rectangle((0.04, 0.12), 0.26, 0.34, facecolor="#fff6f1", edgecolor="#d89a7a", linewidth=2)
    lut_box = Rectangle((0.39, 0.31), 0.22, 0.28, facecolor="#f8f8f8", edgecolor="#6b6b6b", linewidth=2)
    apply_box = Rectangle((0.70, 0.31), 0.26, 0.28, facecolor="#f2fbf3", edgecolor="#7caf83", linewidth=2)
    for patch in (source_box, target_box, lut_box, apply_box):
        canvas.add_patch(patch)

    canvas.text(0.17, 0.88, "Source cohort", ha="center", va="center", fontsize=16, weight="bold")
    canvas.text(0.17, 0.84, "pool BU tissue pixels", ha="center", va="center", fontsize=11)
    canvas.text(0.17, 0.44, "Target cohort", ha="center", va="center", fontsize=16, weight="bold")
    canvas.text(0.17, 0.40, "pool NW tissue pixels", ha="center", va="center", fontsize=11)
    canvas.text(0.50, 0.54, "Learn one cohort LUT", ha="center", va="center", fontsize=16, weight="bold")
    canvas.text(
        0.50,
        0.45,
        "per channel:\nmatch source-cohort percentiles\nto target-cohort percentiles",
        ha="center",
        va="center",
        fontsize=12,
    )
    canvas.text(0.83, 0.54, "Apply same LUT", ha="center", va="center", fontsize=16, weight="bold")
    canvas.text(
        0.83,
        0.45,
        "transform each BU slide\nwith the same cohort-level mapping",
        ha="center",
        va="center",
        fontsize=12,
    )

    arrow_style = dict(arrowstyle="->", mutation_scale=18, linewidth=2, color="#4f4f4f")
    canvas.add_patch(FancyArrowPatch((0.30, 0.73), (0.39, 0.52), **arrow_style))
    canvas.add_patch(FancyArrowPatch((0.30, 0.29), (0.39, 0.38), **arrow_style))
    canvas.add_patch(FancyArrowPatch((0.61, 0.45), (0.70, 0.45), **arrow_style))

    source_positions = [(0.06, 0.62, 0.07, 0.17), (0.145, 0.62, 0.07, 0.17), (0.23, 0.62, 0.07, 0.17)]
    target_positions = [(0.06, 0.18, 0.07, 0.17), (0.145, 0.18, 0.07, 0.17), (0.23, 0.18, 0.07, 0.17)]
    output_positions = [(0.72, 0.36, 0.07, 0.17), (0.805, 0.36, 0.07, 0.17), (0.89, 0.36, 0.07, 0.17)]

    for image, (x0, y0, w, h) in zip(sources, source_positions, strict=True):
        ax = fig.add_axes([x0, y0, w, h])
        _thumbnail(ax, image)
    for image, (x0, y0, w, h) in zip(targets, target_positions, strict=True):
        ax = fig.add_axes([x0, y0, w, h])
        _thumbnail(ax, image)
    for image, (x0, y0, w, h) in zip(matched, output_positions, strict=True):
        ax = fig.add_axes([x0, y0, w, h])
        _thumbnail(ax, image)

    mini = fig.add_axes([0.43, 0.35, 0.14, 0.14])
    x = np.arange(256)
    mini.plot(x, np.sqrt(x / 255.0), color="#1f77b4", linewidth=2, label="source CDF")
    mini.plot(x, (x / 255.0) ** 1.5, color="#d62728", linewidth=2, label="target CDF")
    mini.set_xticks([0, 128, 255])
    mini.set_yticks([0.0, 0.5, 1.0])
    mini.tick_params(labelsize=8)
    mini.set_title("schematic CDFs", fontsize=9)

    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _make_single_vs_multi_examples_figure(
    comparison_rows: list[dict[str, str]],
    chosen_target_path: Path,
    output_path: Path,
) -> None:
    chosen_target = load_rgb_uint8(chosen_target_path)
    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(len(comparison_rows), 4, figure=fig, wspace=0.03, hspace=0.08)
    column_titles = ["Source BU", "Chosen single target", "Single-target output", "Multi-source+target output"]

    for row_index, row in enumerate(comparison_rows):
        images = [
            load_rgb_uint8(Path(row["source_png"])),
            chosen_target,
            load_rgb_uint8(Path(row["single_matched_png"])),
            load_rgb_uint8(Path(row["prior_multi_matched_png"])),
        ]
        for col_index, image in enumerate(images):
            ax = fig.add_subplot(gs[row_index, col_index])
            _thumbnail(ax, image)
            if row_index == 0:
                ax.set_title(column_titles[col_index], fontsize=12, weight="bold")
            if col_index == 0:
                ax.set_ylabel(f"BU slide {row_index + 1}", fontsize=12)

    fig.suptitle("Real example outputs: one target slide vs cohort-level fit", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _make_distribution_dashboard(rgb_plot: Path, lab_plot: Path, output_path: Path) -> None:
    rgb = plt.imread(rgb_plot)
    lab = plt.imread(lab_plot)

    fig, axes = plt.subplots(2, 1, figsize=(14, 18))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB cohort histograms and CDFs", fontsize=14)
    axes[0].axis("off")
    axes[1].imshow(lab)
    axes[1].set_title("Lab cohort histograms and CDFs", fontsize=14)
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _benchmark_png_workflow(
    source_paths: list[Path],
    target_path: Path,
    target_paths: list[Path],
    *,
    luminosity_threshold: float = 0.65,
) -> dict[str, float]:
    sources = [load_rgb_uint8(path) for path in source_paths]
    single_target = load_rgb_uint8(target_path)
    multi_targets = [load_rgb_uint8(path) for path in target_paths]

    single = HistogramMatchingNormalizer(luminosity_threshold=luminosity_threshold)
    t0 = time.perf_counter()
    single.fit(single_target)
    t1 = time.perf_counter()
    single.transform(sources[0])
    t2 = time.perf_counter()

    multi = HistogramMatchingNormalizer(luminosity_threshold=luminosity_threshold)
    t3 = time.perf_counter()
    multi.fit_multi_source_target(sources, multi_targets)
    t4 = time.perf_counter()
    for image in sources:
        multi.transform(image, apply_source_tissue_mask=False)
    t5 = time.perf_counter()

    return {
        "single_fit_s": round(t1 - t0, 3),
        "single_transform_s": round(t2 - t1, 3),
        "multi_fit_s": round(t4 - t3, 3),
        "multi_transform_total_s": round(t5 - t4, 3),
        "multi_transform_avg_s": round((t5 - t4) / len(sources), 3),
    }


def build_story_pack(
    single_metadata_path: Path,
    multi_metadata_path: Path,
    output_dir: Path,
    *,
    example_index: int = 0,
) -> dict:
    search_roots = [
        PROJECT_ROOT / "outputs" / "histogram_matching_single_target_from_multi_example",
        PROJECT_ROOT / "outputs" / "histogram_matching_multi_source_target_example",
        PROJECT_ROOT / "outputs",
    ]

    single_meta = _load_json(single_metadata_path)
    multi_meta = _load_json(multi_metadata_path)

    comparison_rows = list(single_meta["outputs"])
    chosen_target_path = _resolve_existing_path(single_meta["target_png"], search_roots)

    source_paths = [_resolve_existing_path(value, search_roots) for value in single_meta["source_pngs"]]
    local_target_paths = sorted(
        (PROJECT_ROOT / "outputs" / "histogram_matching_multi_source_target_example" / "snapshots" / "targets").glob(
            "*.png"
        )
    )
    if local_target_paths:
        target_paths = local_target_paths
    else:
        target_paths = [_resolve_existing_path(value, search_roots) for value in multi_meta["target_pngs"]]

    matched_paths = [
        _resolve_existing_path(item["prior_multi_matched_png"], search_roots)
        for item in comparison_rows
    ]

    example_row = comparison_rows[example_index]
    example_source = load_rgb_uint8(_resolve_existing_path(example_row["source_png"], search_roots))
    example_target = load_rgb_uint8(chosen_target_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    hist_summary = _make_histogram_math_figure(
        example_source,
        example_target,
        output_dir / "01_blue_channel_hist_cdf_lut.png",
    )
    _make_cohort_pooling_figure(
        source_paths=source_paths[:3],
        target_paths=target_paths[:3],
        matched_paths=matched_paths[:3],
        output_path=output_dir / "02_cohort_pooling_schematic.png",
    )
    _make_single_vs_multi_examples_figure(
        comparison_rows=comparison_rows,
        chosen_target_path=chosen_target_path,
        output_path=output_dir / "03_single_vs_multi_examples.png",
    )
    _make_distribution_dashboard(
        rgb_plot=_resolve_existing_path(
            single_meta["cohort_plot_outputs"]["rgb"],
            search_roots,
        ),
        lab_plot=_resolve_existing_path(
            single_meta["cohort_plot_outputs"]["lab"],
            search_roots,
        ),
        output_path=output_dir / "04_distribution_dashboard.png",
    )

    benchmark = _benchmark_png_workflow(source_paths, chosen_target_path, target_paths[:3])
    summary = {
        "single_metadata_path": str(single_metadata_path),
        "multi_metadata_path": str(multi_metadata_path),
        "story_pack_dir": str(output_dir),
        "selected_source_labels": [_to_short_label(path.name) for path in source_paths],
        "selected_target_labels": [_to_short_label(path.name) for path in target_paths],
        "example_index": example_index,
        "histogram_figure_summary": hist_summary,
        "png_level4_benchmark": benchmark,
        "generated_files": [
            "01_blue_channel_hist_cdf_lut.png",
            "02_cohort_pooling_schematic.png",
            "03_single_vs_multi_examples.png",
            "04_distribution_dashboard.png",
        ],
    }
    (output_dir / "story_pack_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-metadata", type=Path, default=DEFAULT_SINGLE_METADATA)
    parser.add_argument("--multi-metadata", type=Path, default=DEFAULT_MULTI_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--example-index", type=int, default=0)
    args = parser.parse_args()

    summary = build_story_pack(
        single_metadata_path=args.single_metadata,
        multi_metadata_path=args.multi_metadata,
        output_dir=args.output_dir,
        example_index=args.example_index,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
