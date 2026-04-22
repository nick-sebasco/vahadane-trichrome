"""Cohort Wasserstein evaluation example on three real cohort images.

Run:
python /home/sebasn/vahadane-trichrome/examples/cohort_evaluation_example.py

This example does not perform stain normalization. It demonstrates the
evaluation workflow on raw images from three cohorts by:

1. computing a cohort Wasserstein distance matrix
2. saving pooled cohort feature distribution plots in OD space
3. writing a small text summary to the outputs directory

For a true before/after study, run the same script pattern once on raw images
and again on separately normalized images, then compare the outputs.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vahadane_trichrome import cohort_wasserstein_matrix
from vahadane_trichrome import plot_cohort_feature_distributions


TEST_DATA_DIR = PROJECT_ROOT / "examples" / "example_data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "examples" / "cohort_evaluation"

COHORTS = {
    "BU": [
        TEST_DATA_DIR / "BU_6_-_2024-08-23_08.50.20.png",
    ],
    "KD": [
        TEST_DATA_DIR / "KD_V_tri_center.png",
    ],
    "NW": [
        TEST_DATA_DIR
        / "NW_NW_Trichrome_Box_1_LF_RF_SSCMH27_BASE_LF_04-27-11_-_2022-11-10_16.02.34_left.png",
    ],
}


def _format_distance_matrix(result) -> str:
    names = result.cohort_names
    header = ["cohort", *names]
    rows = ["\t".join(header)]
    for row_idx, row_name in enumerate(names):
        formatted_values = [f"{result.distance_matrix[row_idx, col_idx]:.6f}" for col_idx in range(len(names))]
        rows.append("\t".join([row_name, *formatted_values]))
    return "\n".join(rows)


def _format_channel_distances(result) -> str:
    lines = ["Per-pair channel distances:"]
    seen_pairs: set[tuple[str, str]] = set()
    for left_name in result.cohort_names:
        for right_name in result.cohort_names:
            if left_name == right_name:
                continue
            if (right_name, left_name) in seen_pairs:
                continue
            distances = result.channel_distances[(left_name, right_name)]
            channel_text = ", ".join(
                f"{channel}={distance:.6f}"
                for channel, distance in zip(result.channels, distances, strict=True)
            )
            lines.append(f"- {left_name} vs {right_name}: {channel_text}")
            seen_pairs.add((left_name, right_name))
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    missing_paths = [str(path) for paths in COHORTS.values() for path in paths if not path.exists()]
    if missing_paths:
        missing = "\n".join(missing_paths)
        raise FileNotFoundError(f"Missing example image(s):\n{missing}")

    result = cohort_wasserstein_matrix(
        COHORTS,
        feature_domain="od",
        luminosity_threshold=0.9,
        max_pixels_per_image=50_000,
        random_state=0,
    )

    plot_result = plot_cohort_feature_distributions(
        COHORTS,
        output_path=OUTPUT_DIR / "cohort_od_distributions.png",
        feature_domain="od",
        luminosity_threshold=0.9,
        max_pixels_per_image=50_000,
        random_state=0,
        plot_kind="both",
        bins=128,
    )

    matrix_path = OUTPUT_DIR / "cohort_wasserstein_matrix.tsv"
    matrix_path.write_text(_format_distance_matrix(result) + "\n", encoding="utf-8")

    summary_path = OUTPUT_DIR / "summary.txt"
    summary_lines = [
        "Cohort Wasserstein evaluation summary",
        "",
        "Inputs:",
        *(f"- {cohort}: {paths[0]}" for cohort, paths in COHORTS.items()),
        "",
        f"Feature domain: {result.feature_domain}",
        f"Channels: {', '.join(result.channels)}",
        f"Saved distribution plot: {plot_result.output_path}",
        f"Saved matrix: {matrix_path}",
        "",
        _format_distance_matrix(result),
        "",
        _format_channel_distances(result),
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("Cohort Wasserstein evaluation example complete.")
    print(f"Saved summary: {summary_path}")
    print(f"Saved matrix: {matrix_path}")
    print(f"Saved distribution plot: {plot_result.output_path}")
    print()
    print(_format_distance_matrix(result))
    print()
    print(_format_channel_distances(result))


if __name__ == "__main__":
    main()
