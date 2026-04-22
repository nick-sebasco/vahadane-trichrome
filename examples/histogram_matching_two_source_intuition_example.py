"""Intuition experiment: two very different sources, one chosen target.

Run:
poetry run python examples/histogram_matching_two_source_intuition_example.py

What it does:
1. Export two specified NW level-5 source snapshots and one chosen BU target snapshot.
2. Apply classic single-target histogram matching to each source separately.
3. Fit one shared multi-source-target histogram mapping using both sources and the same target.
4. Save visual comparisons, cohort distribution plots, and quantitative distance summaries.

Why this exists:
This is meant to build intuition for the difference between:
- fitting each source separately to the same chosen target image
- learning one fixed shared mapping from a diverse source set to that same target

With deliberately different sources, the shared mapping usually looks more
conservative than fitting each source independently to the same target.
"""

from __future__ import annotations

import json
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
from vahadane_trichrome import cohort_wasserstein_matrix
from vahadane_trichrome import load_rgb_uint8
from vahadane_trichrome import plot_cohort_feature_distributions
from vahadane_trichrome import sample_image_features
from vahadane_trichrome import wasserstein_distance_1d


OUTPUT_LEVEL = 5
SOURCE_COHORT = "NW"
TARGET_COHORT = "BU"
SOURCE_NAMES = [
    "NW_Trichrome_Box_1_LF_RF_SSCMH02_BASE_LF_01-14-09_-_2022-11-10_17.23.59.zarr",
    "NW_Trichrome_Box_2_LA_SSCMH23_24MO_LA_10-29-12_-_2022-11-16_13.43.49.zarr",
]
TARGET_NAME = "3_-_2024-08-23_08.40.24.zarr"

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "histogram_matching_two_source_intuition_example"
SNAPSHOT_DIR = OUTPUT_ROOT / "snapshots"
SOURCE_SNAPSHOT_DIR = SNAPSHOT_DIR / "sources"
TARGET_SNAPSHOT_DIR = SNAPSHOT_DIR / "target"
TRANSFORM_DIR = OUTPUT_ROOT / "transformed"
PLOT_DIR = OUTPUT_ROOT / "plots"

MASK_KWARGS = dict(
    luminosity_threshold=0.65,
    use_connected_components=True,
    min_component_size_fraction=5e-4,
    min_component_size_relative_to_largest=1e-2,
    cumulative_foreground_coverage=0.995,
    connected_components_connectivity=2,
    connected_components_fail_safe=True,
)
EVAL_MASK_KWARGS = dict(
    luminosity_threshold=MASK_KWARGS["luminosity_threshold"],
    use_connected_components=MASK_KWARGS["use_connected_components"],
)


def _export_png(cohort: str, zarr_name: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{cohort}_level{OUTPUT_LEVEL}_{zarr_name.removesuffix('.zarr')}.png"
    Img(cohort, zarr_name, level=OUTPUT_LEVEL).save_to_png(output_path)
    return output_path


def _save_source_target_overview(
    source_images: list[np.ndarray],
    source_names: list[str],
    target_image: np.ndarray,
    target_name: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, (ax, image, name) in enumerate(zip(axes[:2], source_images, source_names, strict=True)):
        ax.imshow(image)
        ax.set_title(f"Source {idx + 1} ({SOURCE_COHORT})\n{name[:52]}")
        ax.axis("off")
    axes[2].imshow(target_image)
    axes[2].set_title(f"Chosen target ({TARGET_COHORT}, same target for both fits)\n{target_name[:52]}")
    axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_case_comparison(
    source: np.ndarray,
    target: np.ndarray,
    single_matched: np.ndarray,
    multi_matched: np.ndarray,
    output_path: Path,
    *,
    source_label: str,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    axes[0].imshow(source)
    axes[0].set_title(f"{source_label}\nraw")
    axes[1].imshow(target)
    axes[1].set_title("Chosen BU target")
    axes[2].imshow(single_matched)
    axes[2].set_title("Per-source fit\nto same BU target")
    axes[3].imshow(multi_matched)
    axes[3].set_title("Shared fit from\nboth sources to same BU target")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _mean_lab_wasserstein(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left_features = sample_image_features(
        left,
        feature_domain="lab",
        max_pixels=30_000,
        random_state=0,
        **EVAL_MASK_KWARGS,
    )
    right_features = sample_image_features(
        right,
        feature_domain="lab",
        max_pixels=30_000,
        random_state=0,
        **EVAL_MASK_KWARGS,
    )
    channels = ("L", "a", "b")
    per_channel = {
        channel_name: float(wasserstein_distance_1d(left_features[:, idx], right_features[:, idx]))
        for idx, channel_name in enumerate(channels)
    }
    per_channel["mean"] = float(np.mean(list(per_channel.values())))
    return per_channel


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    SOURCE_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    TRANSFORM_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("Exporting source and target snapshots...", flush=True)
    source_pngs = [_export_png(SOURCE_COHORT, name, SOURCE_SNAPSHOT_DIR) for name in SOURCE_NAMES]
    target_png = _export_png(TARGET_COHORT, TARGET_NAME, TARGET_SNAPSHOT_DIR)

    print("Loading exported PNGs...", flush=True)
    source_images = [load_rgb_uint8(path) for path in source_pngs]
    target_image = load_rgb_uint8(target_png)

    print("Fitting single-target model...", flush=True)
    single_normalizer = HistogramMatchingNormalizer(**MASK_KWARGS).fit(target_image)

    print("Fitting shared multi-source-target model...", flush=True)
    multi_normalizer = HistogramMatchingNormalizer(**MASK_KWARGS).fit_multi_source_target(
        source_images,
        [target_image],
    )

    print("Saving image comparisons...", flush=True)
    single_paths: list[str] = []
    multi_paths: list[str] = []
    outputs: list[dict[str, object]] = []

    overview_path = OUTPUT_ROOT / "source_target_overview.png"
    _save_source_target_overview(source_images, SOURCE_NAMES, target_image, TARGET_NAME, overview_path)

    for idx, (source_name, source_png, source_image) in enumerate(
        zip(SOURCE_NAMES, source_pngs, source_images, strict=True),
        start=1,
    ):
        run_stem = source_png.stem.replace(f"{SOURCE_COHORT}_level{OUTPUT_LEVEL}_", "")
        output_dir = TRANSFORM_DIR / run_stem
        output_dir.mkdir(parents=True, exist_ok=True)

        single_matched = single_normalizer.transform(source_image, apply_source_tissue_mask=True)
        multi_matched = multi_normalizer.transform(source_image, apply_source_tissue_mask=True)

        source_copy_path = output_dir / "source.png"
        target_copy_path = output_dir / "target.png"
        single_path = output_dir / "single_target_matched.png"
        multi_path = output_dir / "multi_source_matched.png"
        comparison_path = output_dir / "comparison.png"

        plt.imsave(source_copy_path, source_image)
        plt.imsave(target_copy_path, target_image)
        plt.imsave(single_path, single_matched)
        plt.imsave(multi_path, multi_matched)
        _save_case_comparison(
            source_image,
            target_image,
            single_matched,
            multi_matched,
            comparison_path,
            source_label=f"Source {idx}",
        )

        single_paths.append(str(single_path))
        multi_paths.append(str(multi_path))
        outputs.append(
            {
                "source_zarr": source_name,
                "source_png": str(source_png),
                "single_target_png": str(target_png),
                "single_matched_png": str(single_path),
                "multi_source_matched_png": str(multi_path),
                "comparison_png": str(comparison_path),
                "raw_to_target_lab_wasserstein": _mean_lab_wasserstein(source_image, target_image),
                "single_to_target_lab_wasserstein": _mean_lab_wasserstein(single_matched, target_image),
                "multi_to_target_lab_wasserstein": _mean_lab_wasserstein(multi_matched, target_image),
            }
        )

    print("Saving cohort distribution plots...", flush=True)
    rgb_plot = plot_cohort_feature_distributions(
        {
            "NW raw": [str(path) for path in source_pngs],
            "NW per-source fit": single_paths,
            "NW shared two-source fit": multi_paths,
            "BU target": [str(target_png)],
        },
        output_path=PLOT_DIR / "rgb_cohort_hist_cdf.png",
        feature_domain="rgb",
        max_pixels_per_image=50_000,
        random_state=0,
        plot_kind="both",
        bins=128,
        density=True,
        **EVAL_MASK_KWARGS,
    )
    lab_plot = plot_cohort_feature_distributions(
        {
            "NW raw": [str(path) for path in source_pngs],
            "NW per-source fit": single_paths,
            "NW shared two-source fit": multi_paths,
            "BU target": [str(target_png)],
        },
        output_path=PLOT_DIR / "lab_cohort_hist_cdf.png",
        feature_domain="lab",
        max_pixels_per_image=50_000,
        random_state=0,
        plot_kind="both",
        bins=128,
        density=True,
        **EVAL_MASK_KWARGS,
    )

    print("Computing cohort distance summaries...", flush=True)
    cohort_distances = cohort_wasserstein_matrix(
        {
            "NW raw": [str(path) for path in source_pngs],
            "NW per-source fit": single_paths,
            "NW shared two-source fit": multi_paths,
            "BU target": [str(target_png)],
        },
        feature_domain="lab",
        max_pixels_per_image=50_000,
        random_state=0,
        **EVAL_MASK_KWARGS,
    )

    distance_matrix = {
        row_name: {
            col_name: float(cohort_distances.distance_matrix[row_idx, col_idx])
            for col_idx, col_name in enumerate(cohort_distances.cohort_names)
        }
        for row_idx, row_name in enumerate(cohort_distances.cohort_names)
    }

    summary = {
        "level_key": f"0/{OUTPUT_LEVEL}",
        "source_cohort": SOURCE_COHORT,
        "target_cohort": TARGET_COHORT,
        "selected_source_zarrs": SOURCE_NAMES,
        "selected_target_zarrs": [TARGET_NAME],
        "source_target_overview_png": str(overview_path),
        "source_pair_raw_lab_wasserstein": _mean_lab_wasserstein(source_images[0], source_images[1]),
        "outputs": outputs,
        "cohort_plot_outputs": {
            "rgb": rgb_plot.output_path,
            "lab": lab_plot.output_path,
        },
        "cohort_lab_wasserstein_matrix": distance_matrix,
    }

    metadata_path = OUTPUT_ROOT / "run_metadata.json"
    metadata_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Two-source intuition histogram-matching example complete.")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
