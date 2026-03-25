"""Scientific tests for Vahadane trichrome algorithm."""

import numpy as np
import pytest

from vahadane_trichrome import VahadaneTrichromeNormalizer
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


def test_compute_channel_scale_factors_clips_runaway_percentile_ratios() -> None:
	"""Extreme target/source percentile ratios should be clipped safely.

	This guards against the black-tissue failure mode where a nearly absent source
	stain channel receives an arbitrarily large multiplicative scale factor.
	"""
	max_c_target = np.array([[1.2, 0.8, 1.5]], dtype=np.float64)
	max_c_source = np.array([[0.4, 0.2, 0.01]], dtype=np.float64)

	scale = VahadaneTrichromeNormalizer._compute_channel_scale_factors(
		max_c_target,
		max_c_source,
		max_scale_factor=4.0,
	)

	assert np.allclose(scale, np.array([[3.0, 4.0, 4.0]], dtype=np.float64))


def test_transform_caps_reconstructed_od_to_target_range() -> None:
	"""Transform should not exceed the target slide's observed OD range.

	Swatch images only visualize stain directions, not the concentration magnitudes
	that determine whether the final reconstruction saturates toward black.
	This regression test verifies that transform caps reconstructed OD using the
	target slide's own 99th-percentile OD statistics.
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

	identity_basis = np.eye(3, dtype=np.float64)
	mask = np.ones((1, 1), dtype=bool)
	normalizer = VahadaneTrichromeNormalizer(
		extractor=_FixedExtractor(identity_basis, mask),
		max_concentration_scale_factor=None,
	)
	normalizer.stain_matrix_target = np.eye(3, dtype=np.float64)
	normalizer.max_c_target = np.array([[100.0, 100.0, 100.0]], dtype=np.float64)
	normalizer.max_od_target = np.array([[0.4, 0.5, 0.6]], dtype=np.float64)

	img = np.array([[[10, 10, 10]]], dtype=np.uint8)
	transformed = normalizer.transform(img, apply_source_tissue_mask=False)

	expected = np.clip(255.0 * np.exp(-normalizer.max_od_target), 0, 255).astype(np.uint8)
	np.testing.assert_array_equal(transformed.reshape(1, 3), expected)

