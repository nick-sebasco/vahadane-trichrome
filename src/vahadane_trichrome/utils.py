"""Public utility helpers for stain processing workflows."""

import numpy as np
from skimage import measure


def rgb2od(img: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 image to optical density (OD) space.

    Uses the standard transform ``OD = -log(I / 255)`` with clipping to avoid
    log(0).

    Args:
        img (:class:`numpy.ndarray`):
            RGB image with shape ``(H, W, 3)``.

    Returns:
        :class:`numpy.ndarray`:
            OD image of shape ``(H, W, 3)`` as ``float32``.

    """
    image = np.asarray(img, dtype=np.uint8)
    image = np.clip(image, 1, 255).astype(np.float32, copy=False)
    return -np.log(image / 255.0)


def get_luminosity_tissue_mask(img: np.ndarray, threshold: float = 0.8) -> np.ndarray:
    """Compute a simple luminosity-based tissue mask.

    A pixel is considered tissue if its normalized grayscale luminosity is below
    ``threshold``.

    Args:
        img (:class:`numpy.ndarray`):
            RGB uint8 image with shape ``(H, W, 3)``.
        threshold (float):
            Normalized luminosity threshold in ``[0, 1]``.

    Returns:
        :class:`numpy.ndarray`:
            Boolean tissue mask of shape ``(H, W)``.

    """
    image = np.asarray(img, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image shape (H, W, 3), got {image.shape}.")
    lum = (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]) / 255.0
    return lum < float(threshold)


def refine_tissue_mask_connected_components(
    tissue_mask: np.ndarray,
    *,
    min_component_size_fraction: float = 5e-4,
    min_component_size_relative_to_largest: float = 1e-2,
    cumulative_foreground_coverage: float = 0.995,
    connectivity: int = 2,
    fail_safe_return_raw_mask: bool = True,
) -> np.ndarray:
    """Refine a binary tissue mask by removing small connected components.

    This function intentionally does not fill holes or run closing/opening, so
    internal whitespace (for example lumens) remains background by default.

    Args:
        tissue_mask (:class:`numpy.ndarray`):
            Boolean or binary mask of shape ``(H, W)``.
        min_component_size_fraction (float):
            Minimum kept component area as fraction of image area.
        min_component_size_relative_to_largest (float):
            Minimum kept component area as fraction of largest component area.
        cumulative_foreground_coverage (float):
            Keep largest components until this fraction of foreground area is
            reached. Must be in ``(0, 1]``.
        connectivity (int):
            Connected-components neighborhood. ``1`` is 4-connectivity,
            ``2`` is 8-connectivity for 2D masks.
        fail_safe_return_raw_mask (bool):
            If filtering removes all components, return input mask instead.

    Returns:
        :class:`numpy.ndarray`:
            Refined boolean tissue mask of shape ``(H, W)``.

    """
    mask = np.asarray(tissue_mask).astype(bool, copy=False)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D tissue mask, got shape {mask.shape}.")
    if not np.any(mask):
        return mask.copy()

    if not (0.0 <= min_component_size_fraction <= 1.0):
        raise ValueError("min_component_size_fraction must be in [0, 1].")
    if not (0.0 <= min_component_size_relative_to_largest <= 1.0):
        raise ValueError("min_component_size_relative_to_largest must be in [0, 1].")
    if not (0.0 < cumulative_foreground_coverage <= 1.0):
        raise ValueError("cumulative_foreground_coverage must be in (0, 1].")

    labels = measure.label(mask, connectivity=connectivity)
    component_labels, counts = np.unique(labels[labels > 0], return_counts=True)
    if component_labels.size == 0:
        return mask.copy()

    order = np.argsort(-counts)
    component_labels = component_labels[order]
    counts = counts[order]

    image_area = mask.size
    largest_area = int(counts[0])
    min_abs_area = int(np.ceil(min_component_size_fraction * image_area))
    min_rel_area = int(np.ceil(min_component_size_relative_to_largest * largest_area))
    min_area = max(1, min_abs_area, min_rel_area)

    eligible = counts >= min_area
    if not np.any(eligible):
        return mask.copy() if fail_safe_return_raw_mask else np.zeros_like(mask, dtype=bool)

    eligible_labels = component_labels[eligible]
    eligible_counts = counts[eligible]
    total_fg = int(np.sum(counts))
    target_fg = cumulative_foreground_coverage * float(total_fg)

    keep_labels: list[int] = []
    covered = 0.0
    for label_value, area in zip(eligible_labels, eligible_counts):
        keep_labels.append(int(label_value))
        covered += float(area)
        if covered >= target_fg:
            break

    refined = np.isin(labels, keep_labels)
    if not np.any(refined) and fail_safe_return_raw_mask:
        return mask.copy()
    return refined


def get_tissue_mask(
    img: np.ndarray,
    *,
    luminosity_threshold: float = 0.8,
    use_connected_components: bool = True,
    min_component_size_fraction: float = 5e-4,
    min_component_size_relative_to_largest: float = 1e-2,
    cumulative_foreground_coverage: float = 0.995,
    connected_components_connectivity: int = 2,
    connected_components_fail_safe: bool = True,
) -> np.ndarray:
    """Compute tissue mask from luminosity with optional CC refinement."""
    raw_mask = get_luminosity_tissue_mask(img, threshold=luminosity_threshold)
    if not use_connected_components:
        return raw_mask
    return refine_tissue_mask_connected_components(
        raw_mask,
        min_component_size_fraction=min_component_size_fraction,
        min_component_size_relative_to_largest=min_component_size_relative_to_largest,
        cumulative_foreground_coverage=cumulative_foreground_coverage,
        connectivity=connected_components_connectivity,
        fail_safe_return_raw_mask=connected_components_fail_safe,
    )


__all__ = [
    "rgb2od",
    "get_luminosity_tissue_mask",
    "refine_tissue_mask_connected_components",
    "get_tissue_mask",
]
