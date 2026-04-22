"""Histogram specification / histogram matching utilities."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Sequence

import numpy as np

from ..utils import get_tissue_mask


def _validate_uint8_image(image: np.ndarray) -> np.ndarray:
    """Return a validated uint8 image array."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image data, got {arr.dtype}.")
    if arr.ndim not in {2, 3}:
        raise ValueError(f"Expected 2D or 3D image array, got shape {arr.shape}.")
    return arr


def _compute_discrete_cdf(channel: np.ndarray, levels: int = 256) -> np.ndarray:
    """Compute the scaled and rounded discrete CDF for one uint8 channel."""
    flat = np.asarray(channel, dtype=np.uint8).ravel()
    if flat.size == 0:
        raise ValueError("Channel must contain at least one pixel.")

    hist = np.bincount(flat, minlength=levels)
    cdf = hist.cumsum()
    return np.round((cdf / cdf[-1]) * (levels - 1)).astype(np.int16)


def build_histogram_specification_lut(
    source_channel: np.ndarray,
    target_channel: np.ndarray,
    levels: int = 256,
) -> np.ndarray:
    """Build the textbook discrete histogram-specification lookup table."""
    source = np.asarray(source_channel, dtype=np.uint8)
    target = np.asarray(target_channel, dtype=np.uint8)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError(
            "Histogram specification expects 2D single-channel arrays, got "
            f"{source.shape} and {target.shape}."
        )

    source_cdf = _compute_discrete_cdf(source, levels=levels)
    target_cdf = _compute_discrete_cdf(target, levels=levels)

    lut = np.empty(levels, dtype=np.uint8)
    for intensity in range(levels):
        lut[intensity] = np.argmin(np.abs(target_cdf - source_cdf[intensity]))
    return lut


def _extract_masked_channel(
    image: np.ndarray,
    mask: np.ndarray | None,
    channel_index: int | None = None,
) -> np.ndarray:
    """Return a 2D channel image or its masked 1D values."""
    arr = np.asarray(image, dtype=np.uint8)
    if channel_index is not None:
        arr = arr[..., channel_index]

    if mask is None:
        return arr

    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape != arr.shape[:2]:
        raise ValueError(
            f"Mask shape {mask_bool.shape} does not match image spatial shape {arr.shape[:2]}."
        )
    masked = arr[mask_bool]
    if masked.size == 0:
        raise ValueError("Mask selected no pixels for histogram matching.")
    return masked


def match_channel_histogram(
    source_channel: np.ndarray,
    target_channel: np.ndarray,
    levels: int = 256,
) -> np.ndarray:
    """Match one uint8 channel to a target channel via histogram specification."""
    source = np.asarray(source_channel, dtype=np.uint8)
    lut = build_histogram_specification_lut(source, target_channel, levels=levels)
    return lut[source]


def apply_histogram_lut(
    image: np.ndarray,
    lut: np.ndarray,
) -> np.ndarray:
    """Apply a precomputed histogram-specification LUT to one image or channel."""
    arr = _validate_uint8_image(image)
    lookup = np.asarray(lut, dtype=np.uint8)

    if arr.ndim == 2:
        if lookup.ndim == 2:
            if lookup.shape[0] != 1:
                raise ValueError(
                    "Grayscale images require a 1D LUT or a 2D LUT with exactly one channel."
                )
            lookup = lookup[0]
        if lookup.ndim != 1 or lookup.shape[0] != 256:
            raise ValueError(f"Expected LUT shape (256,), got {lookup.shape}.")
        return lookup[arr]

    if lookup.ndim == 1:
        lookup = np.broadcast_to(lookup[None, :], (arr.shape[-1], lookup.shape[0]))
    if lookup.ndim != 2 or lookup.shape != (arr.shape[-1], 256):
        raise ValueError(
            f"Expected LUT shape ({arr.shape[-1]}, 256) for image with {arr.shape[-1]} channels, got {lookup.shape}."
        )

    matched = np.empty_like(arr)
    for channel_index in range(arr.shape[-1]):
        matched[..., channel_index] = lookup[channel_index][arr[..., channel_index]]
    return matched


def histogram_specification(
    source_img: np.ndarray,
    target_img: np.ndarray,
    *,
    source_mask: np.ndarray | None = None,
    target_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply histogram specification independently to each channel."""
    source = _validate_uint8_image(source_img)
    target = _validate_uint8_image(target_img)

    if source.ndim != target.ndim:
        raise ValueError(
            "Source and target must have the same rank, got "
            f"{source.ndim} and {target.ndim}."
        )

    if source.ndim == 2:
        source_samples = _extract_masked_channel(source, source_mask)
        target_samples = _extract_masked_channel(target, target_mask)
        lut = build_histogram_specification_lut(source_samples.reshape(1, -1), target_samples.reshape(1, -1))
        return lut[source]

    if source.shape[-1] != target.shape[-1]:
        raise ValueError(
            "Source and target must have the same number of channels, got "
            f"{source.shape[-1]} and {target.shape[-1]}."
        )

    matched = np.empty_like(source)
    for channel_index in range(source.shape[-1]):
        source_samples = _extract_masked_channel(source, source_mask, channel_index=channel_index)
        target_samples = _extract_masked_channel(target, target_mask, channel_index=channel_index)
        lut = build_histogram_specification_lut(
            source_samples.reshape(1, -1),
            target_samples.reshape(1, -1),
        )
        matched[..., channel_index] = lut[source[..., channel_index]]
    return matched


def _validate_image_collection(
    images: Sequence[np.ndarray],
    *,
    label: str,
) -> list[np.ndarray]:
    if len(images) == 0:
        raise ValueError(f"{label} must contain at least one image.")

    validated = [_validate_uint8_image(image) for image in images]
    reference = validated[0]
    for idx, image in enumerate(validated[1:], start=1):
        if image.ndim != reference.ndim:
            raise ValueError(
                f"All {label} images must have the same rank. "
                f"Image 0 has rank {reference.ndim}, image {idx} has rank {image.ndim}."
            )
        if image.ndim == 3 and image.shape[-1] != reference.shape[-1]:
            raise ValueError(
                f"All {label} images must have the same channel count. "
                f"Image 0 has {reference.shape[-1]} channels, image {idx} has {image.shape[-1]}."
            )
    return [image.copy() for image in validated]


def _validate_mask_collection(
    masks: Sequence[np.ndarray | None] | None,
    images: Sequence[np.ndarray],
    *,
    label: str,
) -> list[np.ndarray | None]:
    if masks is None:
        return [None] * len(images)
    if len(masks) != len(images):
        raise ValueError(f"{label} must have the same length as images.")

    validated: list[np.ndarray | None] = []
    for idx, (mask, image) in enumerate(zip(masks, images, strict=True)):
        if mask is None:
            validated.append(None)
            continue
        mask_bool = np.asarray(mask, dtype=bool)
        if mask_bool.shape != image.shape[:2]:
            raise ValueError(
                f"{label}[{idx}] shape {mask_bool.shape} does not match image shape {image.shape[:2]}."
            )
        validated.append(mask_bool)
    return validated


def build_cohort_histogram_specification_lut(
    source_images: Sequence[np.ndarray],
    target_images: Sequence[np.ndarray],
    *,
    source_masks: Sequence[np.ndarray | None] | None = None,
    target_masks: Sequence[np.ndarray | None] | None = None,
    levels: int = 256,
) -> np.ndarray:
    """Learn a fixed histogram-matching LUT from source and reference cohorts.

    Step by step:
    1. Validate every source and reference image as uint8 with consistent channel
       structure inside each cohort and across cohorts.
    2. Optionally apply one tissue mask per image so only selected pixels are
       used to estimate the cohort distributions.
    3. Pool masked pixel values across all source images into one source cohort
       distribution for each channel.
    4. Pool masked pixel values across all reference images into one reference cohort
       distribution for each channel.
    5. For each channel, build one shared LUT that answers this question:
       if a pixel value sits at percentile p in the source cohort, what pixel
       value sits at that same percentile p in the reference cohort?
    6. Reuse that same learned LUT when transforming individual source images.

    Why this can be more robust than single-source/single-target matching:
    - The mapping is not dominated by one particular source slide's staining,
      tissue mix, or artifact pattern.
    - The reference style is estimated from several reference images instead of one.
    - Small masking errors or unusual tissue regions in one image contribute
      less because the cohort pools average them out.

    Assumptions:
    - Source images come from a cohort that should share one common mapping to
      the reference cohort.
    - Reference images represent the desired appearance template as a cohort.
    - Pooling pixel distributions across images is a meaningful summary.

    Limitations:
    - This learns a fixed cohort-level LUT, so it is less adaptive to unusual
      individual source slides than classic per-image histogram matching.
    - Tissue composition differences can still influence the learned mapping.
    - Channel dependence is ignored because each channel is matched separately.
    """
    source_cohort = _validate_image_collection(source_images, label="source_images")
    target_cohort = _validate_image_collection(target_images, label="target_images")

    source_reference = source_cohort[0]
    target_reference = target_cohort[0]
    if source_reference.ndim != target_reference.ndim:
        raise ValueError(
            "Source and target cohorts must have the same image rank, got "
            f"{source_reference.ndim} and {target_reference.ndim}."
        )
    if source_reference.ndim == 3 and source_reference.shape[-1] != target_reference.shape[-1]:
        raise ValueError(
            "Source and target cohorts must have the same number of channels, got "
            f"{source_reference.shape[-1]} and {target_reference.shape[-1]}."
        )

    source_mask_list = _validate_mask_collection(source_masks, source_cohort, label="source_masks")
    target_mask_list = _validate_mask_collection(target_masks, target_cohort, label="target_masks")

    if source_reference.ndim == 2:
        source_values = np.concatenate(
            [
                _extract_masked_channel(image, mask).reshape(-1)
                for image, mask in zip(source_cohort, source_mask_list, strict=True)
            ]
        )
        target_values = np.concatenate(
            [
                _extract_masked_channel(image, mask).reshape(-1)
                for image, mask in zip(target_cohort, target_mask_list, strict=True)
            ]
        )
        return build_histogram_specification_lut(
            source_values.reshape(1, -1),
            target_values.reshape(1, -1),
            levels=levels,
        )

    channel_luts = np.empty((source_reference.shape[-1], levels), dtype=np.uint8)
    for channel_index in range(source_reference.shape[-1]):
        source_values = np.concatenate(
            [
                _extract_masked_channel(image, mask, channel_index=channel_index).reshape(-1)
                for image, mask in zip(source_cohort, source_mask_list, strict=True)
            ]
        )
        target_values = np.concatenate(
            [
                _extract_masked_channel(image, mask, channel_index=channel_index).reshape(-1)
                for image, mask in zip(target_cohort, target_mask_list, strict=True)
            ]
        )
        channel_luts[channel_index] = build_histogram_specification_lut(
            source_values.reshape(1, -1),
            target_values.reshape(1, -1),
            levels=levels,
        )
    return channel_luts


@dataclass
class HistogramMatchingNormalizer:
    """Histogram matching with single-target and cohort-level fitting modes.

    Supported modes:
    - ``fit(target_img)``:
      classic single-reference histogram matching. A fresh LUT is derived for
      each source image so that the transformed source follows the stored
      reference image histogram.
    - ``fit_multi_source_target(source_imgs, target_imgs)``:
      learn one fixed cohort-level LUT from pooled source and reference
      cohorts, then reuse that LUT for each transformed source image.
    """

    luminosity_threshold: float = 0.8
    use_connected_components: bool = True
    min_component_size_fraction: float = 5e-4
    min_component_size_relative_to_largest: float = 1e-2
    cumulative_foreground_coverage: float = 0.995
    connected_components_connectivity: int = 2
    connected_components_fail_safe: bool = True
    target_image: np.ndarray | None = None
    target_tissue_mask: np.ndarray | None = field(default=None, init=False, repr=False)
    source_tissue_mask: np.ndarray | None = field(default=None, init=False, repr=False)
    channel_luts: np.ndarray | None = field(default=None, init=False, repr=False)
    fit_mode: str | None = field(default=None, init=False)
    fitted_source_images: list[np.ndarray] | None = field(default=None, init=False, repr=False)
    fitted_target_images: list[np.ndarray] | None = field(default=None, init=False, repr=False)
    fitted_source_tissue_masks: list[np.ndarray] | None = field(default=None, init=False, repr=False)
    fitted_target_tissue_masks: list[np.ndarray] | None = field(default=None, init=False, repr=False)

    def _compute_tissue_mask(self, image: np.ndarray) -> np.ndarray:
        """Compute tissue mask using the shared luminosity + CC thresholder."""
        return get_tissue_mask(
            image,
            luminosity_threshold=self.luminosity_threshold,
            use_connected_components=self.use_connected_components,
            min_component_size_fraction=self.min_component_size_fraction,
            min_component_size_relative_to_largest=self.min_component_size_relative_to_largest,
            cumulative_foreground_coverage=self.cumulative_foreground_coverage,
            connected_components_connectivity=self.connected_components_connectivity,
            connected_components_fail_safe=self.connected_components_fail_safe,
        ).astype(bool, copy=False)

    def fit(self, target_img: np.ndarray) -> "HistogramMatchingNormalizer":
        """Store the target image used for subsequent matching."""
        self.target_image = _validate_uint8_image(target_img).copy()
        self.target_tissue_mask = self._compute_tissue_mask(self.target_image)
        self.channel_luts = None
        self.fit_mode = "single_target"
        self.fitted_source_images = None
        self.fitted_target_images = [self.target_image.copy()]
        self.fitted_source_tissue_masks = None
        self.fitted_target_tissue_masks = [self.target_tissue_mask.copy()]
        return self

    def fit_multi_source_target(
        self,
        source_imgs: Sequence[np.ndarray],
        target_imgs: Sequence[np.ndarray],
    ) -> "HistogramMatchingNormalizer":
        """Learn a fixed cohort-level mapping from many sources to many references.

        This mode is intended for the case where you do not want the mapping to
        be defined by one source slide and one reference slide alone. Instead,
        it pools tissue pixels across a source cohort and a reference cohort to
        learn one channel-wise LUT per cohort pair. After fitting,
        ``transform()`` applies that same fixed mapping to each source image
        individually so the transformed sources follow the reference cohort
        appearance.

        Detailed assumptions:
        - The provided source images are representative of the source cohort you
          want to map from.
        - The provided target images are representative of the desired
          reference/template cohort appearance.
        - A single global LUT per channel is appropriate for the entire source
          cohort, even though individual slides may still vary.

        Why it can be more robust:
        - One noisy or atypical slide has less influence on the learned mapping.
        - Reference appearance is estimated from several reference slides rather
          than from one potentially unrepresentative slide.
        - The same mapping can then be applied consistently across sources.

        Tradeoff versus single-image histogram matching:
        - More robust at the cohort level.
        - Less adaptive to per-slide idiosyncrasies, because the mapping is
          fixed after fit rather than recomputed per source image.
        """
        source_cohort = _validate_image_collection(source_imgs, label="source_imgs")
        target_cohort = _validate_image_collection(target_imgs, label="target_imgs")
        source_masks = [self._compute_tissue_mask(image) for image in source_cohort]
        target_masks = [self._compute_tissue_mask(image) for image in target_cohort]

        self.channel_luts = build_cohort_histogram_specification_lut(
            source_cohort,
            target_cohort,
            source_masks=source_masks,
            target_masks=target_masks,
        )
        self.fit_mode = "multi_source_target"
        self.target_image = None
        self.target_tissue_mask = None
        self.fitted_source_images = [image.copy() for image in source_cohort]
        self.fitted_target_images = [image.copy() for image in target_cohort]
        self.fitted_source_tissue_masks = [mask.copy() for mask in source_masks]
        self.fitted_target_tissue_masks = [mask.copy() for mask in target_masks]
        return self

    def transform(self, source_img: np.ndarray, apply_source_tissue_mask: bool = True) -> np.ndarray:
        """Match a source image to the fitted target histogram."""
        source = _validate_uint8_image(source_img)
        if self.fit_mode is None:
            raise RuntimeError("Target mapping is not available. Run fit() or fit_multi_source_target() first.")

        self.source_tissue_mask = self._compute_tissue_mask(source)
        if self.fit_mode == "multi_source_target":
            if self.channel_luts is None:
                raise RuntimeError("Channel LUTs are not available. Run fit_multi_source_target() again.")
            transformed = apply_histogram_lut(source, self.channel_luts)
        else:
            if self.target_image is None:
                raise RuntimeError("Target image is not available. Run fit() first.")
            transformed = histogram_specification(
                source,
                self.target_image,
                source_mask=self.source_tissue_mask,
                target_mask=self.target_tissue_mask,
            )
        if apply_source_tissue_mask:
            transformed = transformed.copy()
            transformed[~self.source_tissue_mask] = 255
        return transformed

    def fit_transform(
        self,
        source_img: np.ndarray,
        target_img: np.ndarray,
        apply_source_tissue_mask: bool = True,
    ) -> np.ndarray:
        """Fit on the target image and transform the source image."""
        self.fit(target_img)
        return self.transform(source_img, apply_source_tissue_mask=apply_source_tissue_mask)
