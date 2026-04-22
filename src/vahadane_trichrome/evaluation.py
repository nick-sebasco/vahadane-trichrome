"""Evaluation helpers for cohort comparison and normalization assessment.

This module does not perform stain normalization. It only measures distances
or similarities on the images it is given. The intended workflow is:

1. Run evaluation on raw cohorts.
2. Normalize images separately with whatever settings you want.
3. Run evaluation again on the normalized cohorts.
4. Compare the before/after results outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from typing import Mapping
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from skimage import color
from skimage.metrics import structural_similarity

from .utils import get_tissue_mask
from .utils import rgb2od

ImageInput = np.ndarray | str | Path


@dataclass(frozen=True)
class CohortDistanceResult:
    """Pairwise cohort distances in a chosen feature domain."""

    cohort_names: tuple[str, ...]
    distance_matrix: np.ndarray
    channel_distances: dict[tuple[str, str], np.ndarray]
    feature_domain: str
    channels: tuple[str, ...]


@dataclass(frozen=True)
class StructuralSimilarityResult:
    """SSIM summary for aligned image pairs."""

    scores: np.ndarray
    mean_score: float
    std_score: float
    feature_domain: str


@dataclass(frozen=True)
class ReferenceCohortImprovement:
    """Before/after distance improvement relative to a reference cohort."""

    reference_cohort: str
    before: dict[str, float]
    after: dict[str, float]
    deltas: dict[str, float]
    improved_cohorts: tuple[str, ...]


@dataclass(frozen=True)
class CohortDistributionPlotResult:
    """Metadata for a saved cohort feature distribution plot."""

    output_path: str
    cohort_names: tuple[str, ...]
    feature_domain: str
    channels: tuple[str, ...]
    plot_kind: str


def _to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    image = np.asarray(arr)
    image = np.squeeze(image)

    if image.ndim == 2:
        rgb = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim == 3 and image.shape[-1] in (3, 4):
        rgb = image[..., :3]
    elif image.ndim == 3 and image.shape[0] in (3, 4):
        rgb = np.moveaxis(image[:3, ...], 0, -1)
    else:
        raise ValueError(f"Could not convert array with shape {image.shape} to RGB.")

    if rgb.dtype != np.uint8:
        scale = 255.0 if float(np.max(rgb)) <= 1.0 else 1.0
        rgb = np.clip(rgb * scale, 0, 255).astype(np.uint8)
    return rgb


def load_rgb_uint8(image: ImageInput) -> np.ndarray:
    """Load an RGB image from an array or path."""
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return _to_rgb_uint8(plt.imread(path))
    return _to_rgb_uint8(np.asarray(image))


def _channel_names(feature_domain: str) -> tuple[str, ...]:
    normalized = feature_domain.lower()
    if normalized == "od":
        return ("od_r", "od_g", "od_b")
    if normalized == "rgb":
        return ("r", "g", "b")
    if normalized == "lab":
        return ("lab_l", "lab_a", "lab_b")
    if normalized == "lab_l":
        return ("lab_l",)
    raise ValueError(
        f"Unsupported feature_domain={feature_domain!r}. Expected one of: od, rgb, lab, lab_l."
    )


def _extract_feature_image(image: np.ndarray, feature_domain: str) -> np.ndarray:
    """Map an RGB image into the feature space used for evaluation.

    Why this exists:
    Wasserstein distance compares numeric distributions, so we first need a
    per-pixel representation to compare. The choice of feature domain defines
    what "difference" means:

    - ``od``: optical density, often preferred for stain analysis because
      absorbance-like effects are better behaved than raw RGB.
    - ``rgb``: direct color comparison in display space.
    - ``lab``: perceptual color comparison.
    - ``lab_l``: intensity/lightness-only comparison.

    Assumptions:
    - The chosen feature domain is a meaningful proxy for the cohort effect you
      want to measure.
    - Comparing marginal channel distributions is acceptable for your use case.

    Limitations:
    - This does not isolate stain effects from biology, scanner effects, or
      section-thickness effects.
    - Spatial arrangement is discarded; only per-pixel values are retained.
    - In multichannel modes, downstream code compares channels independently,
      so cross-channel joint structure is not modeled.
    """
    normalized = feature_domain.lower()
    if normalized == "od":
        return rgb2od(image)
    if normalized == "rgb":
        return image.astype(np.float32) / 255.0
    if normalized == "lab":
        return color.rgb2lab(image.astype(np.float32) / 255.0).astype(np.float32)
    if normalized == "lab_l":
        lab = color.rgb2lab(image.astype(np.float32) / 255.0).astype(np.float32)
        return lab[..., :1]
    raise ValueError(
        f"Unsupported feature_domain={feature_domain!r}. Expected one of: od, rgb, lab, lab_l."
    )


def _resolve_mask(
    image: np.ndarray,
    *,
    use_tissue_mask: bool,
    tissue_mask: np.ndarray | None,
    luminosity_threshold: float,
    use_connected_components: bool,
) -> np.ndarray:
    if tissue_mask is not None:
        mask = np.asarray(tissue_mask, dtype=bool)
        if mask.shape != image.shape[:2]:
            raise ValueError(
                f"tissue_mask shape {mask.shape} does not match image shape {image.shape[:2]}."
            )
        return mask
    if not use_tissue_mask:
        return np.ones(image.shape[:2], dtype=bool)
    return get_tissue_mask(
        image,
        luminosity_threshold=luminosity_threshold,
        use_connected_components=use_connected_components,
    )


def sample_image_features(
    image: ImageInput,
    *,
    feature_domain: str = "od",
    tissue_mask: np.ndarray | None = None,
    use_tissue_mask: bool = True,
    luminosity_threshold: float = 0.8,
    use_connected_components: bool = True,
    max_pixels: int | None = 50_000,
    random_state: int | np.random.Generator | None = 0,
) -> np.ndarray:
    """Extract masked pixel features from one image for downstream evaluation.

    Why this exists:
    Whole-slide or patch images can differ greatly in size, whitespace content,
    and tissue location. This helper reduces an image to a set of tissue-pixel
    feature vectors so cohort comparisons are based on distributions of tissue
    values rather than raw image geometry.

    What it does:
    - load the input image as RGB
    - optionally compute or apply a tissue mask
    - convert masked pixels into the requested feature domain
    - optionally subsample pixels to control runtime and memory

    Assumptions:
    - The tissue mask is accurate enough that retained pixels are mostly tissue.
    - A random subset of pixels is representative when ``max_pixels`` is used.
    - Pixel-level color distributions are an appropriate summary for the image.

    Limitations:
    - It is not immune to cohort differences in tissue composition. If one
      cohort contains more collagen, lumen, muscle, necrosis, or artifact than
      another, the measured distance may reflect biology as much as staining.
    - It ignores where tissue appears in the image; two slides with the same
      color distribution but different structure can look identical here.
    - Masking errors can dominate the result by either excluding real tissue or
      including whitespace, pen marks, folds, or debris.
    - Subsampling reduces variance from huge images but introduces Monte Carlo
      noise; results are approximate unless all pixels are used.
    """
    rgb = load_rgb_uint8(image)
    mask = _resolve_mask(
        rgb,
        use_tissue_mask=use_tissue_mask,
        tissue_mask=tissue_mask,
        luminosity_threshold=luminosity_threshold,
        use_connected_components=use_connected_components,
    )
    if not np.any(mask):
        raise ValueError("No foreground pixels available after masking.")

    feature_image = _extract_feature_image(rgb, feature_domain)
    pixels = feature_image[mask]

    if max_pixels is not None and pixels.shape[0] > max_pixels:
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        indices = rng.choice(pixels.shape[0], size=max_pixels, replace=False)
        pixels = pixels[indices]
    return np.asarray(pixels, dtype=np.float32)


def _pool_cohort_features(
    cohorts: Mapping[str, Sequence[ImageInput]],
    *,
    feature_domain: str,
    use_tissue_mask: bool,
    luminosity_threshold: float,
    use_connected_components: bool,
    max_pixels_per_image: int | None,
    random_state: int | np.random.Generator | None,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, np.ndarray]]:
    if len(cohorts) < 2:
        raise ValueError("At least two cohorts are required.")

    channels = _channel_names(feature_domain)
    cohort_names = tuple(cohorts.keys())
    pooled_features: dict[str, np.ndarray] = {}

    for cohort_name, images in cohorts.items():
        if len(images) == 0:
            raise ValueError(f"Cohort {cohort_name!r} does not contain any images.")
        sampled = [
            sample_image_features(
                image,
                feature_domain=feature_domain,
                use_tissue_mask=use_tissue_mask,
                luminosity_threshold=luminosity_threshold,
                use_connected_components=use_connected_components,
                max_pixels=max_pixels_per_image,
                random_state=random_state,
            )
            for image in images
        ]
        pooled_features[cohort_name] = np.concatenate(sampled, axis=0)

    return cohort_names, channels, pooled_features


def wasserstein_distance_1d(u_values: np.ndarray, v_values: np.ndarray) -> float:
    """Compute the 1D Wasserstein distance using empirical quantiles."""
    u = np.sort(np.asarray(u_values, dtype=np.float64).ravel())
    v = np.sort(np.asarray(v_values, dtype=np.float64).ravel())
    if u.size == 0 or v.size == 0:
        raise ValueError("wasserstein_distance_1d requires non-empty inputs.")

    quantiles = np.linspace(0.0, 1.0, max(u.size, v.size), endpoint=True)
    u_interp = np.quantile(u, quantiles, method="linear")
    v_interp = np.quantile(v, quantiles, method="linear")
    return float(np.mean(np.abs(u_interp - v_interp)))


def cohort_wasserstein_matrix(
    cohorts: Mapping[str, Sequence[ImageInput]],
    *,
    feature_domain: str = "od",
    use_tissue_mask: bool = True,
    luminosity_threshold: float = 0.8,
    use_connected_components: bool = True,
    max_pixels_per_image: int | None = 50_000,
    random_state: int | np.random.Generator | None = 0,
) -> CohortDistanceResult:
    """Compute pairwise cohort Earth Mover's Distance from pooled image features.

    This function does not fit or apply any stain normalization model.
    It simply:

    1. Loads each image as RGB.
    2. Optionally keeps only tissue pixels.
    3. Converts those pixels into the requested feature domain.
    4. Optionally subsamples pixels per image.
    5. Pools sampled pixels within each cohort.
    6. Computes 1D Wasserstein distance per channel between every cohort pair.
    7. Stores the mean channel distance in the returned matrix.

    The returned matrix is therefore a cohort-to-cohort distance summary for
    the images exactly as provided. To compare raw vs normalized data, call this
    function once on the raw cohorts and again on the separately normalized
    cohorts, then compare the two results.

    Assumptions:
    - Pooling sampled pixels across all images in a cohort yields a meaningful
      cohort-level distribution.
    - Channel-wise 1D Wasserstein distances are sufficient for the question at
      hand, even though joint multivariate color structure is ignored.
    - Cohorts are comparable enough that changes in color distribution are
      informative rather than being entirely driven by different tissue mixes.

    Limitations:
    - This is a cohort-level distributional metric, not a slide-pair metric.
      It will not tell you which individual slides are responsible for a shift.
    - Different amounts or types of tissue across cohorts can change the result
      even when stain normalization is working correctly.
    - Because features are pooled, large or heterogeneous cohorts can dominate
      the cohort distribution unless cohort design is balanced.
    - This metric is best interpreted as a batch-effect proxy, not as a pure
      stain-normalization truth metric.
    """
    cohort_names, channels, pooled_features = _pool_cohort_features(
        cohorts,
        feature_domain=feature_domain,
        use_tissue_mask=use_tissue_mask,
        luminosity_threshold=luminosity_threshold,
        use_connected_components=use_connected_components,
        max_pixels_per_image=max_pixels_per_image,
        random_state=random_state,
    )

    matrix = np.zeros((len(cohort_names), len(cohort_names)), dtype=np.float64)
    channel_distances: dict[tuple[str, str], np.ndarray] = {}

    for i, left_name in enumerate(cohort_names):
        left = pooled_features[left_name]
        for j in range(i + 1, len(cohort_names)):
            right_name = cohort_names[j]
            right = pooled_features[right_name]
            distances = np.array(
                [
                    wasserstein_distance_1d(left[:, channel_idx], right[:, channel_idx])
                    for channel_idx in range(left.shape[1])
                ],
                dtype=np.float64,
            )
            matrix[i, j] = matrix[j, i] = float(np.mean(distances))
            channel_distances[(left_name, right_name)] = distances
            channel_distances[(right_name, left_name)] = distances.copy()

    return CohortDistanceResult(
        cohort_names=cohort_names,
        distance_matrix=matrix,
        channel_distances=channel_distances,
        feature_domain=feature_domain,
        channels=channels,
    )


def plot_cohort_feature_distributions(
    cohorts: Mapping[str, Sequence[ImageInput]],
    *,
    output_path: str | Path,
    feature_domain: str = "od",
    use_tissue_mask: bool = True,
    luminosity_threshold: float = 0.8,
    use_connected_components: bool = True,
    max_pixels_per_image: int | None = 50_000,
    random_state: int | np.random.Generator | None = 0,
    plot_kind: Literal["hist", "cdf", "both"] = "both",
    bins: int = 128,
    density: bool = True,
) -> CohortDistributionPlotResult:
    """Save cohort-level feature distribution plots for interpretation.

    This helper uses the same pooled cohort features as
    ``cohort_wasserstein_matrix`` so the saved plots reflect the distributions
    that drive the reported Wasserstein distances.

    Recommended use:
    - save plots for the raw cohorts
    - save plots again for the separately normalized cohorts
    - compare the before/after overlays alongside the scalar distances

    Notes:
    - Cohort-level plots are usually more informative than per-image plots for
      this workflow because the Wasserstein statistic is also computed at the
      cohort level.
    - Histograms show shape and spread.
    - CDFs are especially useful for Wasserstein interpretation because the
      distance depends on differences between cumulative distributions.
    """
    cohort_names, channels, pooled_features = _pool_cohort_features(
        cohorts,
        feature_domain=feature_domain,
        use_tissue_mask=use_tissue_mask,
        luminosity_threshold=luminosity_threshold,
        use_connected_components=use_connected_components,
        max_pixels_per_image=max_pixels_per_image,
        random_state=random_state,
    )

    if plot_kind not in {"hist", "cdf", "both"}:
        raise ValueError("plot_kind must be one of: 'hist', 'cdf', 'both'.")
    if bins < 2:
        raise ValueError("bins must be at least 2.")

    n_channels = len(channels)
    n_cols = 2 if plot_kind == "both" else 1
    fig, axes = plt.subplots(
        n_channels,
        n_cols,
        figsize=(6.5 * n_cols, 3.8 * n_channels),
        squeeze=False,
    )

    for channel_idx, channel_name in enumerate(channels):
        channel_values = [
            np.asarray(pooled_features[cohort_name][:, channel_idx], dtype=np.float64)
            for cohort_name in cohort_names
        ]
        channel_min = min(float(np.min(values)) for values in channel_values)
        channel_max = max(float(np.max(values)) for values in channel_values)
        if channel_min == channel_max:
            channel_max = channel_min + 1e-6

        hist_ax = axes[channel_idx, 0]
        cdf_ax = axes[channel_idx, 1] if n_cols == 2 else axes[channel_idx, 0]

        if plot_kind in {"hist", "both"}:
            for cohort_name, values in zip(cohort_names, channel_values, strict=True):
                hist_ax.hist(
                    values,
                    bins=bins,
                    range=(channel_min, channel_max),
                    density=density,
                    alpha=0.4,
                    linewidth=1.0,
                    histtype="stepfilled",
                    label=f"{cohort_name} (n={values.size:,})",
                )
            hist_ax.set_title(f"{channel_name} histogram")
            hist_ax.set_xlabel(channel_name)
            hist_ax.set_ylabel("density" if density else "count")
            hist_ax.grid(True, alpha=0.2)
            hist_ax.legend(loc="best", fontsize=9)

        if plot_kind in {"cdf", "both"}:
            for cohort_name, values in zip(cohort_names, channel_values, strict=True):
                sorted_values = np.sort(values)
                cdf = np.linspace(0.0, 1.0, sorted_values.size, endpoint=True)
                cdf_ax.plot(
                    sorted_values,
                    cdf,
                    linewidth=2.0,
                    label=f"{cohort_name} (n={values.size:,})",
                )
            cdf_ax.set_title(f"{channel_name} empirical CDF")
            cdf_ax.set_xlabel(channel_name)
            cdf_ax.set_ylabel("cumulative probability")
            cdf_ax.grid(True, alpha=0.2)
            cdf_ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        f"Cohort feature distributions in {feature_domain.upper()} space",
        fontsize=14,
    )
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return CohortDistributionPlotResult(
        output_path=str(output),
        cohort_names=cohort_names,
        feature_domain=feature_domain,
        channels=channels,
        plot_kind=plot_kind,
    )


def summarize_reference_cohort_improvement(
    before: CohortDistanceResult,
    after: CohortDistanceResult,
    *,
    reference_cohort: str,
) -> ReferenceCohortImprovement:
    """Summarize change in reference distances between two evaluation runs.

    Typical usage is to pass a "before normalization" cohort distance result and
    an "after normalization" cohort distance result, but this helper only
    compares two precomputed matrices. It does not normalize images itself.
    """
    if before.cohort_names != after.cohort_names:
        raise ValueError("before and after results must use the same cohort ordering.")
    if reference_cohort not in before.cohort_names:
        raise ValueError(f"reference_cohort {reference_cohort!r} was not found.")

    ref_index = before.cohort_names.index(reference_cohort)
    before_distances: dict[str, float] = {}
    after_distances: dict[str, float] = {}
    deltas: dict[str, float] = {}

    for idx, cohort_name in enumerate(before.cohort_names):
        if cohort_name == reference_cohort:
            continue
        before_value = float(before.distance_matrix[ref_index, idx])
        after_value = float(after.distance_matrix[ref_index, idx])
        before_distances[cohort_name] = before_value
        after_distances[cohort_name] = after_value
        deltas[cohort_name] = after_value - before_value

    improved = tuple(name for name, delta in deltas.items() if delta < 0.0)
    return ReferenceCohortImprovement(
        reference_cohort=reference_cohort,
        before=before_distances,
        after=after_distances,
        deltas=deltas,
        improved_cohorts=improved,
    )


def structural_similarity_score(
    source: ImageInput,
    transformed: ImageInput,
    *,
    feature_domain: str = "lab_l",
    tissue_mask: np.ndarray | None = None,
    use_tissue_mask: bool = True,
    luminosity_threshold: float = 0.8,
    use_connected_components: bool = True,
) -> float:
    """Compute SSIM between source and transformed images in a validation feature space."""
    source_rgb = load_rgb_uint8(source)
    transformed_rgb = load_rgb_uint8(transformed)
    if source_rgb.shape != transformed_rgb.shape:
        raise ValueError(
            f"source and transformed must have the same shape, got {source_rgb.shape} and {transformed_rgb.shape}."
        )

    mask = _resolve_mask(
        source_rgb,
        use_tissue_mask=use_tissue_mask,
        tissue_mask=tissue_mask,
        luminosity_threshold=luminosity_threshold,
        use_connected_components=use_connected_components,
    )
    if not np.any(mask):
        raise ValueError("No foreground pixels available after masking.")

    source_feature = _extract_feature_image(source_rgb, feature_domain)
    transformed_feature = _extract_feature_image(transformed_rgb, feature_domain)
    if source_feature.shape[-1] != 1:
        raise ValueError(
            "structural_similarity_score currently supports single-channel feature domains such as 'lab_l'."
        )

    source_map = source_feature[..., 0].astype(np.float64)
    transformed_map = transformed_feature[..., 0].astype(np.float64)

    rows, cols = np.where(mask)
    row_slice = slice(int(rows.min()), int(rows.max()) + 1)
    col_slice = slice(int(cols.min()), int(cols.max()) + 1)
    source_crop = source_map[row_slice, col_slice].copy()
    transformed_crop = transformed_map[row_slice, col_slice].copy()
    mask_crop = mask[row_slice, col_slice]

    fill_value = float(np.mean(source_crop[mask_crop]))
    source_crop[~mask_crop] = fill_value
    transformed_crop[~mask_crop] = fill_value

    data_range = float(
        max(np.max(source_crop), np.max(transformed_crop))
        - min(np.min(source_crop), np.min(transformed_crop))
    )
    if data_range == 0.0:
        return 1.0
    return float(structural_similarity(source_crop, transformed_crop, data_range=data_range))


def paired_structural_similarity(
    sources: Sequence[ImageInput],
    transformed: Sequence[ImageInput],
    *,
    feature_domain: str = "lab_l",
    use_tissue_mask: bool = True,
    luminosity_threshold: float = 0.8,
    use_connected_components: bool = True,
) -> StructuralSimilarityResult:
    """Compute SSIM for aligned source/transformed image pairs."""
    if len(sources) != len(transformed):
        raise ValueError("sources and transformed must have the same length.")
    if len(sources) == 0:
        raise ValueError("paired_structural_similarity requires at least one image pair.")

    scores = np.array(
        [
            structural_similarity_score(
                source,
                transformed_image,
                feature_domain=feature_domain,
                use_tissue_mask=use_tissue_mask,
                luminosity_threshold=luminosity_threshold,
                use_connected_components=use_connected_components,
            )
            for source, transformed_image in zip(sources, transformed, strict=True)
        ],
        dtype=np.float64,
    )
    return StructuralSimilarityResult(
        scores=scores,
        mean_score=float(np.mean(scores)),
        std_score=float(np.std(scores)),
        feature_domain=feature_domain,
    )


__all__ = [
    "CohortDistributionPlotResult",
    "CohortDistanceResult",
    "ReferenceCohortImprovement",
    "StructuralSimilarityResult",
    "cohort_wasserstein_matrix",
    "load_rgb_uint8",
    "paired_structural_similarity",
    "plot_cohort_feature_distributions",
    "sample_image_features",
    "structural_similarity_score",
    "summarize_reference_cohort_improvement",
    "wasserstein_distance_1d",
]
