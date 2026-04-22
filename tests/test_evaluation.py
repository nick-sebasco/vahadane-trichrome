"""Tests for normalization validation helpers."""

from __future__ import annotations

import numpy as np

from vahadane_trichrome import cohort_wasserstein_matrix
from vahadane_trichrome import paired_structural_similarity
from vahadane_trichrome import plot_cohort_feature_distributions
from vahadane_trichrome import structural_similarity_score
from vahadane_trichrome import summarize_reference_cohort_improvement
from vahadane_trichrome import wasserstein_distance_1d


def _make_tissue_image(
    tissue_rgb: tuple[int, int, int],
    *,
    seed: int,
    offset: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((72, 72, 3), 255, dtype=np.uint8)

    image[12 + offset : 56 + offset, 10:54] = np.array(tissue_rgb, dtype=np.uint8)
    image[24 + offset : 44 + offset, 40:64] = np.array(
        [max(0, channel - 30) for channel in tissue_rgb],
        dtype=np.uint8,
    )
    noise = rng.integers(-6, 7, size=image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return image


def test_wasserstein_distance_is_zero_for_identical_distributions() -> None:
    values = np.array([0.1, 0.4, 0.8, 1.2], dtype=np.float64)
    assert wasserstein_distance_1d(values, values) == 0.0


def test_cohort_wasserstein_matrix_detects_reference_alignment_improvement() -> None:
    reference = [
        _make_tissue_image((120, 90, 70), seed=1, offset=0),
        _make_tissue_image((118, 92, 68), seed=2, offset=1),
    ]
    shifted = [
        _make_tissue_image((185, 145, 115), seed=3, offset=0),
        _make_tissue_image((182, 148, 112), seed=4, offset=1),
    ]
    normalized = [
        _make_tissue_image((124, 94, 72), seed=5, offset=0),
        _make_tissue_image((121, 91, 69), seed=6, offset=1),
    ]

    before = cohort_wasserstein_matrix(
        {"ref": reference, "external": shifted},
        feature_domain="od",
        luminosity_threshold=0.95,
        max_pixels_per_image=4000,
        random_state=7,
    )
    after = cohort_wasserstein_matrix(
        {"ref": reference, "external": normalized},
        feature_domain="od",
        luminosity_threshold=0.95,
        max_pixels_per_image=4000,
        random_state=7,
    )
    improvement = summarize_reference_cohort_improvement(
        before,
        after,
        reference_cohort="ref",
    )

    assert before.distance_matrix.shape == (2, 2)
    assert after.distance_matrix[0, 1] < before.distance_matrix[0, 1]
    assert improvement.improved_cohorts == ("external",)
    assert improvement.deltas["external"] < 0.0


def test_structural_similarity_prefers_color_shift_over_geometric_shift() -> None:
    source = _make_tissue_image((135, 92, 78), seed=10, offset=0)
    recolored = _make_tissue_image((145, 98, 82), seed=10, offset=0)
    shifted_geometry = _make_tissue_image((145, 98, 82), seed=10, offset=6)

    recolored_score = structural_similarity_score(
        source,
        recolored,
        luminosity_threshold=0.95,
    )
    shifted_score = structural_similarity_score(
        source,
        shifted_geometry,
        luminosity_threshold=0.95,
    )

    assert recolored_score > 0.95
    assert recolored_score > shifted_score


def test_paired_structural_similarity_returns_mean_and_per_pair_scores() -> None:
    sources = [
        _make_tissue_image((128, 90, 76), seed=20, offset=0),
        _make_tissue_image((132, 94, 80), seed=21, offset=2),
    ]
    transformed = [
        _make_tissue_image((134, 95, 80), seed=20, offset=0),
        _make_tissue_image((138, 99, 84), seed=21, offset=2),
    ]

    result = paired_structural_similarity(
        sources,
        transformed,
        luminosity_threshold=0.95,
    )

    assert result.scores.shape == (2,)
    assert 0.95 < result.mean_score <= 1.0
    assert result.std_score >= 0.0


def test_plot_cohort_feature_distributions_saves_figure(tmp_path) -> None:
    cohorts = {
        "ref": [
            _make_tissue_image((120, 90, 70), seed=31, offset=0),
        ],
        "external": [
            _make_tissue_image((170, 130, 102), seed=32, offset=1),
        ],
    }

    output_path = tmp_path / "cohort_distributions.png"
    result = plot_cohort_feature_distributions(
        cohorts,
        output_path=output_path,
        feature_domain="od",
        luminosity_threshold=0.95,
        max_pixels_per_image=3000,
        random_state=4,
        plot_kind="both",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert result.output_path == str(output_path)
    assert result.cohort_names == ("ref", "external")
    assert result.channels == ("od_r", "od_g", "od_b")
