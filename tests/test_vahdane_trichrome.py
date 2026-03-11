"""Scientific tests for Vahadane trichrome algorithm.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import matplotlib.pyplot as plt

from vahadane_trichrome import VahadaneTrichromeNormalizer
from vahadane_trichrome import _match_source_rows_to_target


# Helper functions for tests
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
	height: int = 192,
	width: int = 192,
	*,
	square_rgb: tuple[int, int, int] = (210, 40, 40),
	line_rgb: tuple[int, int, int] = (30, 80, 220),
	dot_rgb: tuple[int, int, int] = (20, 20, 20),
) -> tuple[np.ndarray, np.ndarray]:
	"""Create a paint-style synthetic image: white background + red square + blue lines + black dots.

	Returns the RGB image and a boolean mask for the red-square tissue region.
	"""
	img = np.full((height, width, 3), 255, dtype=np.uint8)

	y0, y1 = int(0.2 * height), int(0.8 * height)
	x0, x1 = int(0.2 * width), int(0.8 * width)
	tissue_mask = np.zeros((height, width), dtype=bool)
	tissue_mask[y0:y1, x0:x1] = True

	img[y0:y1, x0:x1] = np.array(square_rgb, dtype=np.uint8)

	line_thicknesses = [1, 2, 3, 5]
	line_rows = np.linspace(y0 + 8, y1 - 8, num=len(line_thicknesses), dtype=int)
	for row, thickness in zip(line_rows, line_thicknesses):
		r1 = max(y0, row - thickness // 2)
		r2 = min(y1, row + thickness // 2 + 1)
		img[r1:r2, x0 + 6:x1 - 6] = np.array(line_rgb, dtype=np.uint8)

	centers = [
		(y0 + 22, x0 + 24, 4),
		(y0 + 38, x0 + 66, 6),
		(y0 + 70, x0 + 40, 5),
		(y0 + 84, x0 + 88, 7),
	]
	ys, xs = np.ogrid[:height, :width]
	for cy, cx, radius in centers:
		circle = (ys - cy) ** 2 + (xs - cx) ** 2 <= radius**2
		circle = circle & tissue_mask
		img[circle] = np.array(dot_rgb, dtype=np.uint8)

	return img, tissue_mask


def _resolve_test_output_dir(tmp_path: Path) -> Path:
	"""Choose persistent test output directory when present, else use pytest tmp_path."""
	persistent_dir = Path(__file__).resolve().parent / "test_outputs"
	if persistent_dir.is_dir():
		return persistent_dir
	return tmp_path


def _simple_edge_map(gray: np.ndarray, threshold: float = 18.0) -> np.ndarray:
	"""Compute a lightweight edge map from finite differences.

	No external dependencies are used; this is sufficient for toy-geometry
	structure checks (lines/dots/square boundaries).
	"""
	gray = gray.astype(np.float32)
	gx = np.zeros_like(gray, dtype=np.float32)
	gy = np.zeros_like(gray, dtype=np.float32)
	gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
	gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
	grad_mag = np.sqrt(gx * gx + gy * gy)
	return grad_mag >= threshold


# Tests
@pytest.mark.parametrize(
	"seed,height,width,concentration_low,concentration_high,mae_threshold",
	[
		(7, 40, 36, 0.00, 1.25, 0.09),
		(21, 48, 32, 0.15, 1.10, 0.08),
		(99, 24, 24, 0.30, 1.60, 0.10),
	],
)
def test_get_concentrations_recovers_physically_valid_solution_with_low_od_error(
	seed: int,
	height: int,
	width: int,
	concentration_low: float,
	concentration_high: float,
	mae_threshold: float,
) -> None:
	"""Validate concentration recovery against synthetic ground-truth mixtures.

	Scientific rationale:
		A core Vahadane step solves a linear inverse problem in optical density
		(OD) space:

			OD \approx C @ W

		where:
		- ``W`` is the stain basis matrix (rows are stain vectors),
		- ``C`` is the per-pixel concentration matrix.

		This test builds synthetic data from known nonnegative concentrations and
		a known 3-stain basis, converts it to RGB via Beer-Lambert, and then
		runs ``get_concentrations`` to recover concentrations.

	What this audits:
		1) Physical feasibility constraint:
		   With ``clip_non_negative=True``, recovered concentrations should be
		   nonnegative.
		2) Numerical validity:
		   Recovered concentrations are finite and have expected shape.
		3) Forward-model consistency:
		   Reprojecting recovered concentrations through the same basis should
		   reconstruct OD with low mean absolute error.

	Why parametrized:
		The inverse problem quality can vary with concentration range and image
		size (effective sample count), so we test several plausible operating
		regimes instead of one hand-picked case.
	"""
	rng = np.random.default_rng(seed)

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

	concentrations_true = rng.uniform(
		concentration_low,
		concentration_high,
		size=(height * width, 3),
	)
	rgb = _synthesize_rgb_from_od_mixture(concentrations_true, basis, height, width)

	concentrations_est = VahadaneTrichromeNormalizer.get_concentrations(
		rgb,
		basis,
		clip_non_negative=True,
	)

	assert concentrations_est.shape == (height * width, 3)
	assert np.all(np.isfinite(concentrations_est))
	assert np.min(concentrations_est) >= 0.0

	od_true = concentrations_true @ basis
	od_est = concentrations_est @ basis
	od_mae = float(np.mean(np.abs(od_true - od_est)))
	assert od_mae < mae_threshold


def test_get_concentrations_preserves_sparse_activation_pattern_on_sparse_synthetic_mixture() -> None:
	"""Audit sparsity behavior on a controlled sparse synthetic concentration field.

	Why this test:
		Sparsity is a core scientific claim in Vahadane-style formulations. For a
		unit-style scientific audit, a synthetic ground-truth sparse mixture is
		more reliable than a real slide because we know the true activation pattern.

	Setup:
		- Build a 3-stain basis in OD space.
		- Generate concentrations where most pixels activate only one stain
		  (and a minority activate two stains).
		- Convert to RGB via Beer-Lambert and recover concentrations with
		  ``get_concentrations(..., clip_non_negative=True)``.

	Assertions:
		1) Recovered concentrations remain substantially sparse (high near-zero rate).
		2) Estimated sparsity is close to true sparsity within tolerance.
		3) Physical nonnegativity constraint is respected.
	"""
	rng = np.random.default_rng(123)
	height, width = 48, 48
	n_pixels = height * width

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

	concentrations_true = np.zeros((n_pixels, 3), dtype=np.float64)
	for i in range(n_pixels):
		if rng.random() < 0.82:
			k = int(rng.integers(0, 3))
			concentrations_true[i, k] = rng.uniform(0.45, 1.40)
		else:
			k1, k2 = rng.choice(3, size=2, replace=False)
			concentrations_true[i, k1] = rng.uniform(0.30, 1.10)
			concentrations_true[i, k2] = rng.uniform(0.20, 0.80)

	rgb = _synthesize_rgb_from_od_mixture(concentrations_true, basis, height, width)
	concentrations_est = VahadaneTrichromeNormalizer.get_concentrations(
		rgb,
		basis,
		clip_non_negative=True,
	)

	assert np.all(np.isfinite(concentrations_est))
	assert np.min(concentrations_est) >= 0.0

	zero_threshold = 0.05
	true_zero_fraction = float(np.mean(concentrations_true <= zero_threshold))
	est_zero_fraction = float(np.mean(concentrations_est <= zero_threshold))

	assert est_zero_fraction > 0.55
	assert abs(est_zero_fraction - true_zero_fraction) < 0.15


@pytest.mark.parametrize(
	"permutation,scale_factors",
	[
		([2, 0, 1], [1.30, 0.90, 1.10]),
		([1, 2, 0], [0.75, 1.40, 0.95]),
	],
)
def test_match_source_rows_to_target_recovers_correct_row_correspondence(
	permutation: list[int],
	scale_factors: list[float],
) -> None:
	"""Validate that source stain rows are aligned to target rows by similarity.

	Scientific rationale:
		Dictionary learning is permutation-invariant: the same stain basis can be
		returned with rows in a different order. For normalization, row ``i`` in
		source must correspond to row ``i`` in target, otherwise concentration
		channels are mixed and the reconstructed colors are scientifically invalid.

	What this test does:
		1) Build a known target 3-stain basis with unit-norm rows.
		2) Construct source basis by permuting rows and applying per-row scaling.
		3) Run ``_match_source_rows_to_target``.
		4) Compare aligned-vs-target after unit normalization (directional match).

	Why scaling is included:
		The alignment logic uses cosine similarity, which should be robust to row
		magnitude differences. This test confirms correspondence recovery under
		both permutation and realistic amplitude variation.
	"""
	target = _unit_rows(
		np.array(
			[
				[0.85, 0.22, 0.15],
				[0.12, 0.90, 0.18],
				[0.20, 0.28, 0.88],
			],
			dtype=np.float64,
		)
	)
	source = target[permutation] * np.asarray(scale_factors, dtype=np.float64)[:, None]

	aligned = _match_source_rows_to_target(source, target)

	aligned_unit = _unit_rows(aligned)
	target_unit = _unit_rows(target)
	assert np.allclose(aligned_unit, target_unit, atol=1e-10)


def test_toy_paint_image_sanity_white_background_is_wiped_out_with_source_mask(tmp_path: Path) -> None:
	"""Sanity-check masking behavior on an idealized, human-interpretable synthetic image.

	This test is intentionally not a full Beer-Lambert realism test. It validates a
	critical practical invariant: when tissue exists in a known square region and the
	rest of the image is pure white background, ``transform(..., apply_source_tissue_mask=True)``
	should leave background pixels white while preserving non-white content in tissue.

	Image design:
		- White background
		- Red square (toy tissue region)
		- Blue horizontal lines of varying thickness inside the square
		- Black circles of varying radius inside the square

	Assertions:
		1) Pixels outside the known tissue square are exactly white in output.
		2) Pixels inside the tissue square are not all white (signal preserved).
		3) Extracted source tissue mask overlaps strongly with known square region.
	"""
	source_img, expected_tissue_mask = _make_toy_tissue_image(
		height=128,
		width=128,
		square_rgb=(210, 40, 40),
		line_rgb=(30, 80, 220),
		dot_rgb=(20, 20, 20),
	)
	target_img, _ = _make_toy_tissue_image(
		square_rgb=(170, 70, 70),
		line_rgb=(80, 120, 200),
		dot_rgb=(35, 35, 35),
	)
	output_dir = _resolve_test_output_dir(tmp_path)

	normalizer = VahadaneTrichromeNormalizer(
		n_components=3,
		luminosity_threshold=0.85,
		regularizer=0.1,
		sort_mode="none",
	)
	normalizer.fit(target_img)
	transformed = normalizer.transform(source_img, apply_source_tissue_mask=True)

	source_path = output_dir / "toy_source.png"
	target_path = output_dir / "toy_target.png"
	transformed_path = output_dir / "toy_transformed_masked.png"
	plt.imsave(source_path, source_img)
	plt.imsave(target_path, target_img)
	plt.imsave(transformed_path, transformed)

	assert normalizer.source_tissue_mask is not None
	assert normalizer.target_tissue_mask is not None
	source_mask_path = output_dir / "toy_source_mask.png"
	target_mask_path = output_dir / "toy_target_mask.png"
	plt.imsave(source_mask_path, normalizer.source_tissue_mask.astype(np.uint8) * 255, cmap="inferno")
	plt.imsave(target_mask_path, normalizer.target_tissue_mask.astype(np.uint8) * 255, cmap="inferno")

	swatch_outputs = normalizer.save_stain_vector_swatches(
		output_dir=str(output_dir),
		prefix="toy",
		rgb=True,
	)

	print(f"Toy source saved: {source_path}")
	print(f"Toy target saved: {target_path}")
	print(f"Toy transformed (masked) saved: {transformed_path}")
	print(f"Mask artifact [source_mask]: {source_mask_path}")
	print(f"Mask artifact [target_mask]: {target_mask_path}")
	for name, path in swatch_outputs.items():
		print(f"Swatch artifact [{name}]: {path}")

	assert normalizer.stain_matrix_source_aligned is not None
	swatch_colors = VahadaneTrichromeNormalizer._stain_matrix_to_swatch_image(
		normalizer.stain_matrix_source_aligned,
		swatch_height=1,
		swatch_width=1,
		rgb=True,
	)[0, :, :]
	green_dominant = np.logical_and(
		swatch_colors[:, 1] > swatch_colors[:, 0] + 8,
		swatch_colors[:, 1] > swatch_colors[:, 2] + 8,
	)
	assert not np.any(green_dominant)

	background_mask = ~expected_tissue_mask
	assert np.all(transformed[background_mask] == 255)
	assert np.any(transformed[expected_tissue_mask] < 255)

	mask_overlap = np.mean(normalizer.source_tissue_mask == expected_tissue_mask)
	assert mask_overlap > 0.96


def test_toy_swatch_colors_change_when_square_changes_from_red_to_green(tmp_path: Path) -> None:
	"""Verify swatches are data-driven by changing toy square color from red to green.

	This test guards against "cheating" or fixed-color swatch behavior. We run two
	otherwise identical toy images (same lines/dots), only changing the square color:
	- Case A: red square
	- Case B: green square

	Expected behavior:
	- Green-dominant swatch count should increase for the green-square case.
	- Mean green channel intensity of swatches should be higher for green-square.
	"""
	output_dir = _resolve_test_output_dir(tmp_path)

	# Case A: red-square toy
	red_source, _ = _make_toy_tissue_image(
		height=128,
		width=128,
		square_rgb=(210, 40, 40),
		line_rgb=(30, 80, 220),
		dot_rgb=(20, 20, 20),
	)
	red_target, _ = _make_toy_tissue_image(
		square_rgb=(170, 70, 70),
		line_rgb=(80, 120, 200),
		dot_rgb=(35, 35, 35),
	)
	red_normalizer = VahadaneTrichromeNormalizer(
		n_components=3,
		luminosity_threshold=0.85,
		regularizer=0.1,
		sort_mode="none",
	)
	red_normalizer.fit(red_target)
	red_normalizer.transform(red_source, apply_source_tissue_mask=True)
	red_swatch_outputs = red_normalizer.save_stain_vector_swatches(
		output_dir=str(output_dir),
		prefix="toy_red_square",
		rgb=True,
	)

	# Case B: green-square toy
	green_source, _ = _make_toy_tissue_image(
		height=128,
		width=128,
		square_rgb=(40, 180, 40),
		line_rgb=(30, 80, 220),
		dot_rgb=(20, 20, 20),
	)
	green_target, _ = _make_toy_tissue_image(
		square_rgb=(70, 160, 70),
		line_rgb=(80, 120, 200),
		dot_rgb=(35, 35, 35),
	)
	green_normalizer = VahadaneTrichromeNormalizer(
		n_components=3,
		luminosity_threshold=0.85,
		regularizer=0.1,
		sort_mode="none",
	)
	green_normalizer.fit(green_target)
	green_normalizer.transform(green_source, apply_source_tissue_mask=True)
	green_swatch_outputs = green_normalizer.save_stain_vector_swatches(
		output_dir=str(output_dir),
		prefix="toy_green_square",
		rgb=True,
	)

	assert red_normalizer.stain_matrix_source_aligned is not None
	assert green_normalizer.stain_matrix_source_aligned is not None

	red_colors = VahadaneTrichromeNormalizer._stain_matrix_to_swatch_image(
		red_normalizer.stain_matrix_source_aligned,
		swatch_height=1,
		swatch_width=1,
		rgb=True,
	)[0, :, :].astype(np.int32)
	green_colors = VahadaneTrichromeNormalizer._stain_matrix_to_swatch_image(
		green_normalizer.stain_matrix_source_aligned,
		swatch_height=1,
		swatch_width=1,
		rgb=True,
	)[0, :, :].astype(np.int32)

	red_green_dominant = np.logical_and(
		red_colors[:, 1] > red_colors[:, 0] + 8,
		red_colors[:, 1] > red_colors[:, 2] + 8,
	)
	green_green_dominant = np.logical_and(
		green_colors[:, 1] > green_colors[:, 0] + 8,
		green_colors[:, 1] > green_colors[:, 2] + 8,
	)

	assert np.sum(green_green_dominant) >= np.sum(red_green_dominant)
	assert float(np.mean(green_colors[:, 1])) > float(np.mean(red_colors[:, 1]))

	print("Red-square swatches:")
	for name, path in red_swatch_outputs.items():
		print(f"  - {name}: {path}")
	print("Green-square swatches:")
	for name, path in green_swatch_outputs.items():
		print(f"  - {name}: {path}")


def test_end_to_end_identity_same_source_target_preserves_toy_structure(tmp_path: Path) -> None:
	"""Audit end-to-end stability: same source/target should preserve toy structure.

	Rationale:
		Before sparsity-focused checks, we verify the full normalization pipeline is
		stable when no cross-slide appearance shift is requested (source == target).
		In this setting, transform output should remain close to the source inside
		tissue and preserve geometric structures (square boundary, lines, dots).

	What is measured:
		1) Tissue-region photometric closeness (mean absolute RGB error).
		2) Structure preservation via edge-map IoU in tissue region.
		3) Background handling: outside expected tissue remains white when masked.
	"""
	source_img, expected_tissue_mask = _make_toy_tissue_image(
		square_rgb=(210, 40, 40),
		line_rgb=(30, 80, 220),
		dot_rgb=(20, 20, 20),
	)
	output_dir = _resolve_test_output_dir(tmp_path)

	normalizer = VahadaneTrichromeNormalizer(
		n_components=3,
		luminosity_threshold=0.85,
		regularizer=0.1,
		sort_mode="none",
	)
	normalizer.fit(source_img)
	transformed = normalizer.transform(source_img, apply_source_tissue_mask=True)

	identity_path = output_dir / "toy_identity_transformed_masked.png"
	plt.imsave(identity_path, transformed)
	print(f"Toy identity transformed (masked) saved: {identity_path}")

	tissue = expected_tissue_mask
	background = ~tissue

	inside_mae = float(np.mean(np.abs(transformed[tissue].astype(np.float32) - source_img[tissue].astype(np.float32))))
	assert inside_mae < 18.0

	source_gray = np.mean(source_img, axis=2)
	transformed_gray = np.mean(transformed, axis=2)
	edge_source = _simple_edge_map(source_gray)
	edge_transformed = _simple_edge_map(transformed_gray)

	edge_source_tissue = edge_source & tissue
	edge_transformed_tissue = edge_transformed & tissue
	intersection = np.sum(edge_source_tissue & edge_transformed_tissue)
	union = np.sum(edge_source_tissue | edge_transformed_tissue)
	edge_iou = 1.0 if union == 0 else float(intersection / union)
	assert edge_iou > 0.70

	assert np.all(transformed[background] == 255)


def test_fit_transform_percentile_scaling_aligns_source_to_target_statistics() -> None:
	"""Audit core scaling math: transform should align source concentration percentiles to target.

	This isolates the scientific scaling step used by ``fit``/``transform``:
		- ``fit`` stores ``max_c_target`` as per-channel 99th-percentile concentrations.
		- ``transform`` rescales source concentrations by ``max_c_target / max_c_source``.

	We construct source/target synthetic mixtures with known per-channel scaling,
	use a fixed extractor (same known basis for both), and verify transformed
	concentration 99th percentiles align to target statistics within tolerance.
	"""
	class _FixedExtractor:
		def __init__(self, stain_matrix: np.ndarray, tissue_mask: np.ndarray) -> None:
			self._stain_matrix = stain_matrix
			self._mask = tissue_mask
			self._last_mask = None

		@property
		def last_tissue_mask(self):
			if self._last_mask is None:
				return None
			return self._last_mask.copy()

		def get_stain_matrix(self, _img: np.ndarray) -> np.ndarray:
			self._last_mask = self._mask
			return self._stain_matrix.copy()

	rng = np.random.default_rng(2026)
	height, width = 56, 56
	n_pixels = height * width

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

	target_conc = rng.uniform(0.18, 1.15, size=(n_pixels, 3))
	source_channel_scale = np.array([1.55, 0.62, 1.28], dtype=np.float64)
	source_conc = target_conc * source_channel_scale[None, :]

	target_img = _synthesize_rgb_from_od_mixture(target_conc, basis, height, width)
	source_img = _synthesize_rgb_from_od_mixture(source_conc, basis, height, width)

	mask = np.ones((height, width), dtype=bool)
	extractor = _FixedExtractor(stain_matrix=basis, tissue_mask=mask)
	normalizer = VahadaneTrichromeNormalizer(extractor=extractor)

	normalizer.fit(target_img)
	transformed = normalizer.transform(source_img, apply_source_tissue_mask=False)

	target_est = VahadaneTrichromeNormalizer.get_concentrations(target_img, basis)
	transformed_est = VahadaneTrichromeNormalizer.get_concentrations(transformed, basis)

	target_p99 = np.percentile(target_est, 99, axis=0)
	transformed_p99 = np.percentile(transformed_est, 99, axis=0)

	rel_err = np.abs(transformed_p99 - target_p99) / np.maximum(target_p99, 1e-8)
	assert np.all(rel_err < 0.18)


def test_real_trichrome_integration_runs_and_logs_outputs(tmp_path: Path) -> None:
	"""Integration check on provided real trichrome image.

	This test focuses on practical pipeline robustness (not exact ground-truth
	physics): it verifies fit/transform executes on a real image, outputs are
	finite/valid, and artifacts are saved for visual QA.
	"""
	data_path = Path(__file__).resolve().parent / "test_data" / "BU_2_-_2024-08-23_08.38.00.png"
	if not data_path.exists():
		pytest.skip(f"Real trichrome test image not found: {data_path}")

	img = plt.imread(data_path)
	if img.ndim == 2:
		img = np.repeat(img[..., None], 3, axis=2)
	if img.ndim == 3 and img.shape[-1] == 4:
		img = img[..., :3]
	if img.dtype != np.uint8:
		img = np.clip(img * 255.0 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)

	max_dim = 512
	h, w = img.shape[:2]
	step = max(1, int(np.ceil(max(h, w) / max_dim)))
	img_small = img[::step, ::step]

	output_dir = _resolve_test_output_dir(tmp_path)
	normalizer = VahadaneTrichromeNormalizer(
		n_components=3,
		luminosity_threshold=0.85,
		regularizer=0.1,
		sort_mode="none",
	)

	normalizer.fit(img_small)
	transformed = normalizer.transform(img_small, apply_source_tissue_mask=True)

	assert transformed.shape == img_small.shape
	assert transformed.dtype == np.uint8
	assert np.all(np.isfinite(transformed))

	source_path = output_dir / "real_trichrome_source.png"
	transformed_path = output_dir / "real_trichrome_transformed_masked.png"
	plt.imsave(source_path, img_small)
	plt.imsave(transformed_path, transformed)

	swatch_outputs = normalizer.save_stain_vector_swatches(
		output_dir=str(output_dir),
		prefix="real_trichrome",
		rgb=True,
	)

	assert normalizer.source_tissue_mask is not None


def test_save_load_fit_state_round_trip_preserves_transform_behavior(tmp_path: Path) -> None:
	"""Round-trip fit-state persistence should preserve transform output.

	This verifies that saving and loading target fit state is sufficient to run
	``transform`` consistently without re-fitting the target image.
	"""
	target_img, _ = _make_toy_tissue_image(
		height=128,
		width=128,
		square_rgb=(170, 70, 70),
		line_rgb=(80, 120, 200),
		dot_rgb=(35, 35, 35),
	)
	source_img, _ = _make_toy_tissue_image(
		height=128,
		width=128,
		square_rgb=(210, 40, 40),
		line_rgb=(30, 80, 220),
		dot_rgb=(20, 20, 20),
	)

	metadata = {
		"fit_date": "2026-03-04",
		"fit_image": "toy_target",
		"note": "round_trip_test",
	}

	n1 = VahadaneTrichromeNormalizer(
		n_components=3,
		luminosity_threshold=0.85,
		regularizer=0.1,
	)
	n1.fit(target_img)
	out1 = n1.transform(source_img, apply_source_tissue_mask=True)

	state_path = tmp_path / "fit_state_round_trip.npz"
	n1.save_fit_state(str(state_path), metadata=metadata)

	n2 = VahadaneTrichromeNormalizer(
		n_components=3,
		luminosity_threshold=0.85,
		regularizer=0.1,
	)
	loaded_metadata = n2.load_fit_state(str(state_path))
	out2 = n2.transform(source_img, apply_source_tissue_mask=True)

	assert loaded_metadata == metadata
	assert n2.fit_metadata == metadata
	assert np.array_equal(out1, out2)
	assert n2.stain_matrix_target is not None
	assert n2.max_c_target is not None


def test_save_load_fit_state_metadata_optional_defaults_to_empty_dict(tmp_path: Path) -> None:
	"""Saving without metadata should load back as an empty metadata dictionary."""
	target_img, _ = _make_toy_tissue_image(
		height=96,
		width=96,
		square_rgb=(170, 70, 70),
		line_rgb=(80, 120, 200),
		dot_rgb=(35, 35, 35),
	)

	normalizer = VahadaneTrichromeNormalizer(
		n_components=3,
		luminosity_threshold=0.85,
		regularizer=0.1,
	)
	normalizer.fit(target_img)

	state_path = tmp_path / "fit_state_no_metadata.npz"
	normalizer.save_fit_state(str(state_path))

	loaded = VahadaneTrichromeNormalizer(
		n_components=3,
		luminosity_threshold=0.85,
		regularizer=0.1,
	)
	metadata = loaded.load_fit_state(str(state_path))

	assert metadata == {}
	assert loaded.fit_metadata == {}