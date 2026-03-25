"""Tests for histogram specification / histogram matching."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from vahadane_trichrome import HistogramMatchingNormalizer
from vahadane_trichrome import histogram_specification
from vahadane_trichrome.cli import _build_parser
from vahadane_trichrome.cli import run_cli
from vahadane_trichrome.histogram_matching import build_histogram_specification_lut


def _channel_histograms(image: np.ndarray) -> np.ndarray:
    """Return per-channel 256-bin histograms."""
    if image.ndim == 2:
        return np.bincount(image.ravel(), minlength=256)[None, :]
    return np.stack(
        [np.bincount(image[..., idx].ravel(), minlength=256) for idx in range(image.shape[-1])],
        axis=0,
    )


def test_histogram_specification_is_stable_when_source_equals_target() -> None:
    rng = np.random.default_rng(20260325)
    image = rng.integers(0, 256, size=(32, 40, 3), dtype=np.uint8)

    matched = histogram_specification(image, image)

    max_delta = int(np.max(np.abs(matched.astype(np.int16) - image.astype(np.int16))))
    mean_delta = float(np.mean(np.abs(matched.astype(np.float32) - image.astype(np.float32))))
    assert max_delta <= 2
    assert mean_delta < 0.2


def test_build_histogram_specification_lut_preserves_two_level_mapping() -> None:
    source = np.array(
        [
            [0, 0, 255, 255],
            [0, 0, 255, 255],
        ],
        dtype=np.uint8,
    )
    target = np.array(
        [
            [32, 32, 200, 200],
            [32, 32, 200, 200],
        ],
        dtype=np.uint8,
    )

    lut = build_histogram_specification_lut(source, target)
    matched = lut[source]

    assert lut[0] == 32
    assert lut[255] == 200
    np.testing.assert_array_equal(matched, target)


def test_histogram_specification_moves_histograms_toward_target() -> None:
    rng = np.random.default_rng(20260325)
    source = rng.integers(0, 80, size=(64, 64, 3), dtype=np.uint8)
    target = rng.integers(160, 256, size=(64, 64, 3), dtype=np.uint8)

    matched = histogram_specification(source, target)

    source_hist = _channel_histograms(source)
    matched_hist = _channel_histograms(matched)
    target_hist = _channel_histograms(target)

    source_distance = np.abs(source_hist - target_hist).sum(axis=1)
    matched_distance = np.abs(matched_hist - target_hist).sum(axis=1)
    assert np.all(matched_distance < source_distance)


def test_histogram_matching_normalizer_fit_transform_matches_function() -> None:
    rng = np.random.default_rng(20260325)
    source = rng.integers(0, 256, size=(24, 30, 3), dtype=np.uint8)
    target = rng.integers(0, 256, size=(24, 30, 3), dtype=np.uint8)

    normalizer = HistogramMatchingNormalizer().fit(target)
    transformed = normalizer.transform(source)

    np.testing.assert_array_equal(
        transformed,
        np.where(
            normalizer.source_tissue_mask[..., None],
            histogram_specification(
                source,
                target,
                source_mask=normalizer.source_tissue_mask,
                target_mask=normalizer.target_tissue_mask,
            ),
            255,
        ).astype(np.uint8),
    )


def test_histogram_matching_normalizer_requires_fit_first() -> None:
    source = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="Run fit\\(\\) first"):
        HistogramMatchingNormalizer().transform(source)


def test_histogram_specification_rejects_non_uint8_inputs() -> None:
    source = np.zeros((8, 8, 3), dtype=np.float32)
    target = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Expected uint8 image data"):
        histogram_specification(source, target)


def test_histogram_matching_uses_tissue_masks_to_ignore_white_background() -> None:
    source = np.full((24, 24, 3), 255, dtype=np.uint8)
    target = np.full((24, 24, 3), 255, dtype=np.uint8)
    source[6:18, 6:18] = np.array([180, 40, 40], dtype=np.uint8)
    target[6:18, 6:18] = np.array([60, 120, 180], dtype=np.uint8)

    normalizer = HistogramMatchingNormalizer(luminosity_threshold=0.95).fit(target)
    transformed = normalizer.transform(source, apply_source_tissue_mask=True)

    assert normalizer.target_tissue_mask is not None
    assert normalizer.source_tissue_mask is not None
    assert np.all(transformed[~normalizer.source_tissue_mask] == 255)
    tissue_mean = transformed[normalizer.source_tissue_mask].mean(axis=0)
    target_tissue_mean = target[6:18, 6:18].reshape(-1, 3).mean(axis=0)
    np.testing.assert_allclose(tissue_mean, target_tissue_mean, atol=10.0)


def test_histogram_matching_defaults_to_white_background_outside_source_tissue() -> None:
    source = np.full((24, 24, 3), 255, dtype=np.uint8)
    target = np.full((24, 24, 3), 255, dtype=np.uint8)
    source[6:18, 6:18] = np.array([180, 40, 40], dtype=np.uint8)
    target[6:18, 6:18] = np.array([60, 120, 180], dtype=np.uint8)

    transformed = HistogramMatchingNormalizer(luminosity_threshold=0.95).fit(target).transform(source)
    background = np.all(source == 255, axis=-1)

    assert np.all(transformed[background] == 255)


def test_run_cli_supports_histogram_matching_method(tmp_path: Path) -> None:
    rng = np.random.default_rng(20260325)
    source = rng.integers(0, 80, size=(24, 30, 3), dtype=np.uint8)
    target = rng.integers(160, 256, size=(24, 30, 3), dtype=np.uint8)

    source_path = tmp_path / "source.png"
    target_path = tmp_path / "target.png"
    output_path = tmp_path / "matched.png"
    artifact_dir = tmp_path / "artifacts"

    plt.imsave(source_path, source)
    plt.imsave(target_path, target)

    parser = _build_parser()
    args = parser.parse_args(
        [
            "--source",
            str(source_path),
            "--reference",
            str(target_path),
            "--output",
            str(output_path),
            "--artifact-dir",
            str(artifact_dir),
            "--method",
            "histogram_matching",
            "--apply-source-tissue-mask",
            "--luminosity-threshold",
            "0.95",
        ]
    )

    exit_code = run_cli(args)
    metadata = json.loads((artifact_dir / "run_metadata.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert output_path.is_file()
    assert metadata["method"] == "histogram_matching"
    assert metadata["fit_mode"] == "single_target"
    assert metadata["apply_source_tissue_mask"] is True
