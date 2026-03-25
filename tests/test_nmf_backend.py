"""Tests for the NMF stain-extraction backend."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from vahadane_trichrome import VahadaneTrichromeExtractor
from vahadane_trichrome import VahadaneTrichromeNormalizer
from vahadane_trichrome.cli import _build_parser
from vahadane_trichrome.cli import run_cli
import vahadane_trichrome.core as core_module
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


def _make_toy_tissue_image(
    height: int = 128,
    width: int = 128,
    *,
    square_rgb: tuple[int, int, int],
    line_rgb: tuple[int, int, int],
    dot_rgb: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    y0, y1 = int(0.2 * height), int(0.8 * height)
    x0, x1 = int(0.2 * width), int(0.8 * width)
    tissue_mask = np.zeros((height, width), dtype=bool)
    tissue_mask[y0:y1, x0:x1] = True
    img[y0:y1, x0:x1] = np.array(square_rgb, dtype=np.uint8)

    line_rows = np.linspace(y0 + 8, y1 - 8, num=4, dtype=int)
    for row, thickness in zip(line_rows, [1, 2, 3, 5]):
        r1 = max(y0, row - thickness // 2)
        r2 = min(y1, row + thickness // 2 + 1)
        img[r1:r2, x0 + 6:x1 - 6] = np.array(line_rgb, dtype=np.uint8)

    ys, xs = np.ogrid[:height, :width]
    for cy, cx, radius in [
        (y0 + 22, x0 + 24, 4),
        (y0 + 38, x0 + 66, 6),
        (y0 + 70, x0 + 40, 5),
        (y0 + 84, x0 + 88, 7),
    ]:
        circle = (ys - cy) ** 2 + (xs - cx) ** 2 <= radius**2
        img[circle & tissue_mask] = np.array(dot_rgb, dtype=np.uint8)

    return img, tissue_mask


def test_nmf_backend_recovers_nonnegative_unit_norm_stain_matrix_on_synthetic_mixture() -> None:
    rng = np.random.default_rng(20260318)
    height, width = 48, 40
    basis = _unit_rows(
        np.array(
            [
                [0.85, 0.22, 0.15],
                [0.12, 0.90, 0.18],
                [0.20, 0.28, 0.88],
            ],
            dtype=np.float64,
        )
    )
    concentrations = rng.uniform(0.15, 1.20, size=(height * width, 3))
    rgb = _synthesize_rgb_from_od_mixture(concentrations, basis, height, width)

    extractor = VahadaneTrichromeExtractor(
        backend="nmf",
        n_components=3,
        regularizer=1e-4,
        sort_mode="none",
        use_connected_components=False,
        nmf_max_iter=3000,
    )
    stain_matrix = extractor.get_stain_matrix(rgb)

    assert stain_matrix.shape == (3, 3)
    assert np.all(np.isfinite(stain_matrix))
    assert np.min(stain_matrix) >= 0.0
    np.testing.assert_allclose(np.linalg.norm(stain_matrix, axis=1), np.ones(3), atol=1e-5)

    aligned = _match_source_rows_to_target(stain_matrix, basis)
    cosine_diag = np.sum(_unit_rows(aligned) * basis, axis=1)
    assert np.min(cosine_diag) > 0.90


def test_nmf_backend_fit_transform_preserves_white_background_on_toy_image() -> None:
    source_img, expected_tissue_mask = _make_toy_tissue_image(
        square_rgb=(210, 40, 40),
        line_rgb=(30, 80, 220),
        dot_rgb=(20, 20, 20),
    )
    target_img, _ = _make_toy_tissue_image(
        square_rgb=(170, 70, 70),
        line_rgb=(80, 120, 200),
        dot_rgb=(35, 35, 35),
    )

    normalizer = VahadaneTrichromeNormalizer(
        backend="nmf",
        n_components=3,
        luminosity_threshold=0.85,
        sort_mode="none",
        nmf_max_iter=3000,
    )
    normalizer.fit(target_img)
    transformed = normalizer.transform(source_img, apply_source_tissue_mask=True)

    assert normalizer.stain_matrix_target is not None
    assert normalizer.stain_matrix_source_aligned is not None
    assert np.all(transformed[~expected_tissue_mask] == 255)
    assert np.any(transformed[expected_tissue_mask] < 255)


def test_run_cli_records_nmf_backend_metadata(tmp_path: Path) -> None:
    source_img, _ = _make_toy_tissue_image(
        square_rgb=(210, 40, 40),
        line_rgb=(30, 80, 220),
        dot_rgb=(20, 20, 20),
    )
    target_img, _ = _make_toy_tissue_image(
        square_rgb=(170, 70, 70),
        line_rgb=(80, 120, 200),
        dot_rgb=(35, 35, 35),
    )
    source_path = tmp_path / "source.png"
    target_path = tmp_path / "target.png"
    output_path = tmp_path / "normalized.png"
    artifact_dir = tmp_path / "artifacts"

    plt.imsave(source_path, source_img)
    plt.imsave(target_path, target_img)

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
            "--save-swatches",
            "--backend",
            "nmf",
            "--nmf-max-iter",
            "3000",
        ]
    )

    exit_code = run_cli(args)
    metadata = json.loads((artifact_dir / "run_metadata.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert output_path.is_file()
    assert metadata["backend"] == "nmf"
    assert metadata["nmf_max_iter"] == 3000
    assert metadata["regularizer"] == 1e-4


def test_nmf_backend_rejects_cd_solver_with_non_frobenius_loss() -> None:
    with pytest.raises(ValueError, match="solver='cd' only supports beta_loss='frobenius'"):
        VahadaneTrichromeExtractor(
            backend="nmf",
            nmf_solver="cd",
            nmf_beta_loss="kullback-leibler",
        )


def test_nmf_backend_raises_on_degenerate_factorization(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeNMF:
        def __init__(self, *args, **kwargs) -> None:
            self.components_ = np.array(
                [
                    [1.0, 0.5, 0.25],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )

        def fit_transform(self, img_od: np.ndarray) -> np.ndarray:
            return np.ones((img_od.shape[0], 3), dtype=np.float64)

    monkeypatch.setattr(core_module, "NMF", _FakeNMF)

    rgb = np.full((16, 16, 3), 180, dtype=np.uint8)
    extractor = VahadaneTrichromeExtractor(
        backend="nmf",
        n_components=3,
        regularizer=1e-4,
        use_connected_components=False,
    )

    with pytest.raises(RuntimeError, match="degenerate stain matrix"):
        extractor.get_stain_matrix(rgb)