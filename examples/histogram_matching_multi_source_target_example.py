"""Cohort histogram-matching example with multiple sources and multiple targets.

Run:
python /home/sebasn/vahadane-trichrome/examples/histogram_matching_multi_source_target_example.py

Workflow:
1. Randomly select 3 BU source zarrs and 3 NW target zarrs.
2. Export level-4 PNG snapshots using the LOCAL image helper.
3. Learn one fixed histogram-matching LUT from the pooled BU and NW cohorts.
4. Apply that learned mapping to each BU source image individually.
5. Save the chosen image list, PNG snapshots, normalized outputs, and triptychs.
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


OUTPUT_LEVEL = 4
RANDOM_SEED = 20260330
N_SOURCES = 3
N_TARGETS = 3
OUTPUT_ROOT = Path("/flashscratch/sebasn/vahadane_runs/histogram_matching_multi_source_target_example")
SNAPSHOT_DIR = OUTPUT_ROOT / "snapshots"
SOURCE_SNAPSHOT_DIR = SNAPSHOT_DIR / "sources"
TARGET_SNAPSHOT_DIR = SNAPSHOT_DIR / "targets"
TRANSFORM_DIR = OUTPUT_ROOT / "transformed"


def _list_zarr_names(cohort: str) -> list[str]:
    image_dir = Path(Img.sources[cohort])
    return sorted(name for name in os.listdir(image_dir) if name.endswith(".zarr"))


def _random_pick(cohort: str, count: int, rng: np.random.Generator) -> list[str]:
    candidates = _list_zarr_names(cohort)
    if len(candidates) < count:
        raise ValueError(f"Requested {count} {cohort} images but only found {len(candidates)}.")
    indices = rng.choice(len(candidates), size=count, replace=False)
    return [candidates[int(idx)] for idx in np.sort(indices)]


def _export_pngs(cohort: str, names: list[str], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for name in names:
        output_path = output_dir / f"{cohort}_level{OUTPUT_LEVEL}_{name.removesuffix('.zarr')}.png"
        Img(cohort, name, level=OUTPUT_LEVEL).save_to_png(str(output_path))
        output_paths.append(output_path)
    return output_paths


def _save_triptych(source: np.ndarray, target: np.ndarray, matched: np.ndarray, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(source)
    axes[0].set_title("Source")
    axes[1].imshow(target)
    axes[1].set_title("Representative target")
    axes[2].imshow(matched)
    axes[2].set_title("Matched")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    SOURCE_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    TRANSFORM_DIR.mkdir(parents=True, exist_ok=True)

    source_names = _random_pick("BU", N_SOURCES, rng)
    target_names = _random_pick("NW", N_TARGETS, rng)

    source_pngs = _export_pngs("BU", source_names, SOURCE_SNAPSHOT_DIR)
    target_pngs = _export_pngs("NW", target_names, TARGET_SNAPSHOT_DIR)

    source_images = [load_rgb_uint8(path) for path in source_pngs]
    target_images = [load_rgb_uint8(path) for path in target_pngs]

    normalizer = HistogramMatchingNormalizer(
        luminosity_threshold=0.65,
        use_connected_components=True,
        min_component_size_fraction=5e-4,
        min_component_size_relative_to_largest=1e-2,
        cumulative_foreground_coverage=0.995,
        connected_components_connectivity=2,
        connected_components_fail_safe=True,
    ).fit_multi_source_target(source_images, target_images)

    representative_target = target_images[0]
    transformed_outputs: list[dict[str, str]] = []
    for source_name, source_png, source_image in zip(source_names, source_pngs, source_images, strict=True):
        run_stem = source_png.stem.replace("BU_level4_", "")
        output_dir = TRANSFORM_DIR / run_stem
        output_dir.mkdir(parents=True, exist_ok=True)

        transformed = normalizer.transform(source_image, apply_source_tissue_mask=True)
        transformed_path = output_dir / "matched.png"
        comparison_path = output_dir / "comparison.png"
        source_copy_path = output_dir / "source.png"
        target_copy_path = output_dir / "representative_target.png"

        plt.imsave(source_copy_path, source_image)
        plt.imsave(target_copy_path, representative_target)
        plt.imsave(transformed_path, transformed)
        _save_triptych(source_image, representative_target, transformed, comparison_path)

        transformed_outputs.append(
            {
                "source_zarr": source_name,
                "source_png": str(source_png),
                "transformed_png": str(transformed_path),
                "comparison_png": str(comparison_path),
            }
        )

    metadata = {
        "random_seed": RANDOM_SEED,
        "level_key": f"0/{OUTPUT_LEVEL}",
        "fit_mode": "multi_source_target",
        "source_cohort": "BU",
        "target_cohort": "NW",
        "selected_source_zarrs": source_names,
        "selected_target_zarrs": target_names,
        "source_pngs": [str(path) for path in source_pngs],
        "target_pngs": [str(path) for path in target_pngs],
        "outputs": transformed_outputs,
    }
    metadata_path = OUTPUT_ROOT / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Multi-source/target histogram matching example complete.")
    print(f"Saved metadata: {metadata_path}")
    print("Selected BU source zarrs:")
    for name in source_names:
        print(f"  - {name}")
    print("Selected NW target zarrs:")
    for name in target_names:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
