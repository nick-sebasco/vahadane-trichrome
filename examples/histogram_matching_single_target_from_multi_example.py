"""Single-target histogram matching on the same BU slides from a prior multi-target run.

Run:
python /home/sebasn/vahadane-trichrome/examples/histogram_matching_single_target_from_multi_example.py

Workflow:
1. Read the previous multi-source/target histogram-matching metadata.
2. Reuse the exact same 3 BU source zarrs from that run.
3. Pick one of the previously used NW targets as the single reference.
4. Export fresh level-4 PNG snapshots.
5. Apply classic single-target histogram matching to each BU source.
6. Save image comparisons plus histogram/CDF plots for multi vs single review.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LOCAL.LOCAL_create_test_images import Img
from vahadane_trichrome import HistogramMatchingNormalizer
from vahadane_trichrome import load_rgb_uint8
from vahadane_trichrome import plot_cohort_feature_distributions


OUTPUT_LEVEL = 4
TARGET_INDEX = 0
PREVIOUS_RUN_ROOT = PROJECT_ROOT / "outputs" / "histogram_matching_multi_source_target_example"
PREVIOUS_METADATA_PATH = PREVIOUS_RUN_ROOT / "run_metadata.json"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "histogram_matching_single_target_from_multi_example"
SNAPSHOT_DIR = OUTPUT_ROOT / "snapshots"
SOURCE_SNAPSHOT_DIR = SNAPSHOT_DIR / "sources"
TARGET_SNAPSHOT_DIR = SNAPSHOT_DIR / "target"
TRANSFORM_DIR = OUTPUT_ROOT / "transformed"
PLOT_DIR = OUTPUT_ROOT / "plots"


def _export_png(cohort: str, zarr_name: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{cohort}_level{OUTPUT_LEVEL}_{zarr_name.removesuffix('.zarr')}.png"
    Img(cohort, zarr_name, level=OUTPUT_LEVEL).save_to_png(str(output_path))
    return output_path


def _save_comparison_figure(
    source: np.ndarray,
    target: np.ndarray,
    single_matched: np.ndarray,
    multi_matched: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(source)
    axes[0].set_title("Source BU")
    axes[1].imshow(target)
    axes[1].set_title("Chosen NW target")
    axes[2].imshow(single_matched)
    axes[2].set_title("Single-target matched")
    axes[3].imshow(multi_matched)
    axes[3].set_title("Prior multi-target matched")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_rgb_histogram_overlay(
    source: np.ndarray,
    target: np.ndarray,
    single_matched: np.ndarray,
    multi_matched: np.ndarray,
    output_path: Path,
) -> None:
    labels_and_images = (
        ("source", source),
        ("target", target),
        ("single", single_matched),
        ("multi", multi_matched),
    )
    colors = ("tab:red", "tab:green", "tab:blue")
    channel_names = ("R", "G", "B")

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    bins = np.arange(257)
    for channel_idx, (ax, color, channel_name) in enumerate(zip(axes, colors, channel_names, strict=True)):
        for label, image in labels_and_images:
            channel = image[..., channel_idx].ravel()
            ax.hist(
                channel,
                bins=bins,
                range=(0, 255),
                density=True,
                histtype="step",
                linewidth=1.4,
                alpha=0.95,
                label=label,
            )
        ax.set_ylabel("density")
        ax.set_title(f"{channel_name} channel")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("pixel value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _load_previous_metadata() -> dict:
    if not PREVIOUS_METADATA_PATH.exists():
        raise FileNotFoundError(f"Previous multi-target metadata not found: {PREVIOUS_METADATA_PATH}")
    return json.loads(PREVIOUS_METADATA_PATH.read_text(encoding="utf-8"))


def main() -> None:
    previous_metadata = _load_previous_metadata()
    source_names = list(previous_metadata["selected_source_zarrs"])
    target_names = list(previous_metadata["selected_target_zarrs"])
    target_name = target_names[TARGET_INDEX]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    SOURCE_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    TRANSFORM_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loaded prior multi-target metadata.", flush=True)
    print("Exporting BU source snapshots...", flush=True)
    source_pngs = [_export_png("BU", name, SOURCE_SNAPSHOT_DIR) for name in source_names]
    print("Exporting chosen NW target snapshot...", flush=True)
    target_png = _export_png("NW", target_name, TARGET_SNAPSHOT_DIR)

    print("Loading exported PNGs...", flush=True)
    source_images = [load_rgb_uint8(path) for path in source_pngs]
    target_image = load_rgb_uint8(target_png)

    print("Fitting single-target histogram normalizer...", flush=True)
    normalizer = HistogramMatchingNormalizer(
        luminosity_threshold=0.65,
        use_connected_components=True,
        min_component_size_fraction=5e-4,
        min_component_size_relative_to_largest=1e-2,
        cumulative_foreground_coverage=0.995,
        connected_components_connectivity=2,
        connected_components_fail_safe=True,
    ).fit(target_image)

    transformed_outputs: list[dict[str, str]] = []
    single_matched_paths: list[str] = []
    prior_multi_matched_paths: list[str] = []

    for source_name, source_png, source_image in zip(source_names, source_pngs, source_images, strict=True):
        print(f"Transforming {source_name}...", flush=True)
        run_stem = source_png.stem.replace("BU_level4_", "")
        output_dir = TRANSFORM_DIR / run_stem
        output_dir.mkdir(parents=True, exist_ok=True)

        prior_multi_matched_path = PREVIOUS_RUN_ROOT / "transformed" / run_stem / "matched.png"
        if not prior_multi_matched_path.exists():
            raise FileNotFoundError(f"Prior multi-target matched image not found: {prior_multi_matched_path}")
        prior_multi_matched = load_rgb_uint8(prior_multi_matched_path)

        single_matched = normalizer.transform(source_image, apply_source_tissue_mask=True)
        source_copy_path = output_dir / "source.png"
        target_copy_path = output_dir / "target.png"
        single_path = output_dir / "single_matched.png"
        multi_copy_path = output_dir / "multi_matched.png"
        comparison_path = output_dir / "comparison.png"
        histogram_path = output_dir / "rgb_histograms.png"

        plt.imsave(source_copy_path, source_image)
        plt.imsave(target_copy_path, target_image)
        plt.imsave(single_path, single_matched)
        plt.imsave(multi_copy_path, prior_multi_matched)
        _save_comparison_figure(source_image, target_image, single_matched, prior_multi_matched, comparison_path)
        _save_rgb_histogram_overlay(source_image, target_image, single_matched, prior_multi_matched, histogram_path)

        single_matched_paths.append(str(single_path))
        prior_multi_matched_paths.append(str(prior_multi_matched_path))
        transformed_outputs.append(
            {
                "source_zarr": source_name,
                "source_png": str(source_png),
                "single_target_png": str(target_png),
                "single_matched_png": str(single_path),
                "prior_multi_matched_png": str(prior_multi_matched_path),
                "comparison_png": str(comparison_path),
                "rgb_histograms_png": str(histogram_path),
            }
        )

    print("Saving cohort histogram/CDF plots...", flush=True)
    rgb_plot = plot_cohort_feature_distributions(
        {
            "BU raw": [str(path) for path in source_pngs],
            "BU single matched": single_matched_paths,
            "BU multi matched": prior_multi_matched_paths,
            "NW chosen target": [str(target_png)],
        },
        output_path=PLOT_DIR / "rgb_cohort_hist_cdf.png",
        feature_domain="rgb",
        luminosity_threshold=0.65,
        use_connected_components=True,
        max_pixels_per_image=50_000,
        random_state=0,
        plot_kind="both",
        bins=128,
        density=True,
    )
    lab_plot = plot_cohort_feature_distributions(
        {
            "BU raw": [str(path) for path in source_pngs],
            "BU single matched": single_matched_paths,
            "BU multi matched": prior_multi_matched_paths,
            "NW chosen target": [str(target_png)],
        },
        output_path=PLOT_DIR / "lab_cohort_hist_cdf.png",
        feature_domain="lab",
        luminosity_threshold=0.65,
        use_connected_components=True,
        max_pixels_per_image=50_000,
        random_state=0,
        plot_kind="both",
        bins=128,
        density=True,
    )

    metadata = {
        "level_key": f"0/{OUTPUT_LEVEL}",
        "fit_mode": "single_target",
        "source_cohort": "BU",
        "target_cohort": "NW",
        "previous_multi_run_metadata": str(PREVIOUS_METADATA_PATH),
        "selected_source_zarrs": source_names,
        "available_previous_target_zarrs": target_names,
        "selected_single_target_index": TARGET_INDEX,
        "selected_single_target_zarr": target_name,
        "source_pngs": [str(path) for path in source_pngs],
        "target_png": str(target_png),
        "outputs": transformed_outputs,
        "cohort_plot_outputs": {
            "rgb": rgb_plot.output_path,
            "lab": lab_plot.output_path,
        },
    }
    metadata_path = OUTPUT_ROOT / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Single-target histogram matching comparison complete.")
    print(f"Saved metadata: {metadata_path}")
    print("Reused BU source zarrs:")
    for name in source_names:
        print(f"  - {name}")
    print(f"Chosen NW single target [{TARGET_INDEX}]: {target_name}")


if __name__ == "__main__":
    main()
