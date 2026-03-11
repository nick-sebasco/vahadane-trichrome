"""Tests for multi-target reference fitting and aggregation."""

import numpy as np

from vahadane_trichrome import VahadaneTrichromeNormalizer
from vahadane_trichrome.core import _aggregate_stain_matrices
from vahadane_trichrome.core import _align_stain_matrices_to_anchor
from vahadane_trichrome.core import _match_source_rows_to_target


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def _synthesize_rgb_from_od_mixture(
    concentrations: np.ndarray,
    stain_matrix: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    od = concentrations @ stain_matrix
    rgb = 255.0 * np.exp(-od)
    return np.clip(rgb, 0, 255).reshape(height, width, 3).astype(np.uint8)


def test_align_and_aggregate_stain_matrices_median_recovers_common_basis() -> None:
    """Median aggregation should recover a common basis after explicit alignment."""
    base = _unit_rows(
        np.array(
            [
                [0.85, 0.22, 0.15],
                [0.12, 0.90, 0.18],
                [0.20, 0.28, 0.88],
            ],
            dtype=np.float64,
        )
    )
    matrices = [
        base[[2, 0, 1]] * np.array([[1.10], [0.95], [1.05]], dtype=np.float64),
        base[[1, 2, 0]] * np.array([[0.90], [1.15], [1.00]], dtype=np.float64),
        base[[0, 1, 2]] * np.array([[1.05], [1.00], [0.92]], dtype=np.float64),
    ]

    aligned, _, _ = _align_stain_matrices_to_anchor(matrices)
    aggregated = _aggregate_stain_matrices(aligned, method="median")

    aggregated_aligned_to_base = _match_source_rows_to_target(aggregated, base)
    assert np.allclose(_unit_rows(aggregated_aligned_to_base), base, atol=5e-2)


def test_fit_multi_target_aligns_reference_matrices_and_scale_statistics() -> None:
    """Multi-target fit should align stain matrices and aggregate per-channel p99 stats."""

    class _SequentialFixedExtractor:
        def __init__(self, stain_matrices: list[np.ndarray], tissue_mask: np.ndarray) -> None:
            self._stain_matrices = [matrix.copy() for matrix in stain_matrices]
            self._mask = tissue_mask
            self._call_index = 0
            self._last_mask = None

        @property
        def last_tissue_mask(self):
            if self._last_mask is None:
                return None
            return self._last_mask.copy()

        def get_stain_matrix(self, _img: np.ndarray) -> np.ndarray:
            matrix = self._stain_matrices[self._call_index]
            self._call_index += 1
            self._last_mask = self._mask
            return matrix.copy()

    rng = np.random.default_rng(20260311)
    height, width = 24, 24
    n_pixels = height * width
    tissue_mask = np.ones((height, width), dtype=bool)

    base = _unit_rows(
        np.array(
            [
                [0.85, 0.22, 0.15],
                [0.12, 0.90, 0.18],
                [0.20, 0.28, 0.88],
            ],
            dtype=np.float64,
        )
    )

    stain_matrices = [
        base[[2, 0, 1]] * np.array([[1.10], [0.95], [1.05]], dtype=np.float64),
        base[[1, 2, 0]] * np.array([[0.90], [1.15], [1.00]], dtype=np.float64),
        base[[0, 1, 2]] * np.array([[1.05], [1.00], [0.92]], dtype=np.float64),
    ]
    stain_matrices = [_unit_rows(matrix) for matrix in stain_matrices]

    concentrations = [
        rng.uniform(0.20, 1.10, size=(n_pixels, 3)),
        rng.uniform(0.15, 0.95, size=(n_pixels, 3)),
        rng.uniform(0.25, 1.20, size=(n_pixels, 3)),
    ]
    target_images = [
        _synthesize_rgb_from_od_mixture(conc, matrix, height, width)
        for conc, matrix in zip(concentrations, stain_matrices)
    ]

    extractor = _SequentialFixedExtractor(stain_matrices=stain_matrices, tissue_mask=tissue_mask)
    normalizer = VahadaneTrichromeNormalizer(extractor=extractor)
    normalizer.fit_multi_target(target_images, aggregation="median", max_workers=1)

    aligned, permutations, _ = _align_stain_matrices_to_anchor(stain_matrices)
    expected_matrix = _aggregate_stain_matrices(aligned, method="median")

    scale_vectors = [
        np.percentile(
            VahadaneTrichromeNormalizer.get_concentrations(image, matrix),
            99,
            axis=0,
            keepdims=True,
        )
        for image, matrix in zip(target_images, stain_matrices)
    ]
    aligned_scale_vectors = [
        scale[:, list(perm)]
        for scale, perm in zip(scale_vectors, permutations)
    ]
    expected_scale = np.median(np.stack(aligned_scale_vectors, axis=0), axis=0)

    assert normalizer.stain_matrix_target is not None
    assert normalizer.max_c_target is not None
    assert np.allclose(normalizer.stain_matrix_target, expected_matrix, atol=1e-6)
    assert np.allclose(normalizer.max_c_target, expected_scale, atol=1e-6)
    assert normalizer.fit_metadata is not None
    assert normalizer.fit_metadata["fit_mode"] == "multi_target"
    assert normalizer.fit_metadata["aggregation"] == "median"
