"""Unit tests for utility helpers in vahadane_trichrome.utils."""

import numpy as np

from vahadane_trichrome import get_luminosity_tissue_mask
from vahadane_trichrome import get_tissue_mask


def test_connected_components_refinement_keeps_multiple_large_regions_and_drops_noise() -> None:
    """Connected-components refinement should preserve multiple valid tissue regions.

    This verifies we do not enforce a "largest component only" behavior.
    """
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    img[5:25, 5:25] = np.array([70, 35, 35], dtype=np.uint8)
    img[50:85, 55:90] = np.array([75, 40, 40], dtype=np.uint8)
    img[10, 90] = np.array([50, 50, 50], dtype=np.uint8)
    img[90, 10] = np.array([50, 50, 50], dtype=np.uint8)

    refined = get_tissue_mask(
        img,
        luminosity_threshold=0.9,
        use_connected_components=True,
        min_component_size_fraction=0.005,
        min_component_size_relative_to_largest=0.02,
        cumulative_foreground_coverage=1.0,
    )

    assert refined[10, 10]
    assert refined[60, 60]
    assert not refined[10, 90]
    assert not refined[90, 10]


def test_get_tissue_mask_can_disable_connected_components() -> None:
    """Disabling CC refinement should return the raw luminosity mask exactly."""
    img = np.full((32, 32, 3), 255, dtype=np.uint8)
    img[6:12, 6:12] = np.array([80, 30, 30], dtype=np.uint8)
    img[20, 22] = np.array([40, 40, 40], dtype=np.uint8)

    raw = get_luminosity_tissue_mask(img, threshold=0.9)
    no_cc = get_tissue_mask(
        img,
        luminosity_threshold=0.9,
        use_connected_components=False,
    )

    assert np.array_equal(raw, no_cc)
