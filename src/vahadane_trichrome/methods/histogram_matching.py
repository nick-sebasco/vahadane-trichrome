"""Histogram specification / histogram matching utilities."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

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


@dataclass
class HistogramMatchingNormalizer:
    """Simple fit/transform wrapper around histogram specification."""

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
        return self

    def transform(self, source_img: np.ndarray, apply_source_tissue_mask: bool = True) -> np.ndarray:
        """Match a source image to the fitted target histogram."""
        if self.target_image is None:
            raise RuntimeError("Target image is not available. Run fit() first.")
        source = _validate_uint8_image(source_img)
        self.source_tissue_mask = self._compute_tissue_mask(source)

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
