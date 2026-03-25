"""Real-image histogram matching examples.

Run:
python /home/sebasn/vahadane-trichrome/examples/histogram_matching_examples.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from vahadane_trichrome import HistogramMatchingNormalizer


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "examples", "example_data")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs", "examples", "histogram_matching")

IMAGE_LIBRARY = {
    "BU_A": os.path.join(TEST_DATA_DIR, "BU_2_-_2024-08-23_08.38.00.png"),
    "BU_B": os.path.join(TEST_DATA_DIR, "BU_6_-_2024-08-23_08.50.20.png"),
    "NW_LEFT": os.path.join(
        TEST_DATA_DIR,
        "NW_NW_Trichrome_Box_1_LF_RF_SSCMH27_BASE_LF_04-27-11_-_2022-11-10_16.02.34_left.png",
    ),
}


def _to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert loaded image arrays to RGB uint8."""
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
        scale = 255.0 if rgb.max() <= 1.0 else 1.0
        rgb = np.clip(rgb * scale, 0, 255).astype(np.uint8)
    return rgb


def _load_image(key: str) -> np.ndarray:
    path = IMAGE_LIBRARY[key]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing image for key '{key}': {path}")
    return _to_rgb_uint8(plt.imread(path))


def _save_triptych(
    source: np.ndarray,
    target: np.ndarray,
    matched: np.ndarray,
    output_path: str,
    *,
    title_prefix: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(source)
    axes[0].set_title(f"{title_prefix} source")
    axes[1].imshow(target)
    axes[1].set_title(f"{title_prefix} target")
    axes[2].imshow(matched)
    axes[2].set_title(f"{title_prefix} matched")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _mean_channel_values(image: np.ndarray) -> list[float]:
    return [float(image[..., idx].mean()) for idx in range(image.shape[-1])]


def run_histogram_matching_example(
    *,
    example_name: str,
    source_key: str,
    target_key: str,
) -> None:
    """Run one real-image histogram matching experiment and save outputs."""
    source = _load_image(source_key)
    target = _load_image(target_key)

    normalizer = HistogramMatchingNormalizer().fit(target)
    matched = normalizer.transform(source, apply_source_tissue_mask=True)

    out_dir = os.path.join(OUTPUT_ROOT, example_name)
    os.makedirs(out_dir, exist_ok=True)

    source_path = os.path.join(out_dir, "source.png")
    target_path = os.path.join(out_dir, "target.png")
    matched_path = os.path.join(out_dir, "matched.png")
    comparison_path = os.path.join(out_dir, "comparison.png")

    plt.imsave(source_path, source)
    plt.imsave(target_path, target)
    plt.imsave(matched_path, matched)
    _save_triptych(source, target, matched, comparison_path, title_prefix=example_name)

    print(f"Example: {example_name}")
    print(f"Source key: {source_key}")
    print(f"Target key: {target_key}")
    print(f"Source mean RGB: {_mean_channel_values(source)}")
    print(f"Target mean RGB: {_mean_channel_values(target)}")
    print(f"Matched mean RGB: {_mean_channel_values(matched)}")
    print(f"Saved source: {source_path}")
    print(f"Saved target: {target_path}")
    print(f"Saved matched: {matched_path}")
    print(f"Saved comparison: {comparison_path}")


def main() -> None:
    """Run a simple real-data histogram matching experiment."""
    run_histogram_matching_example(
        example_name="bu_to_nw_left",
        source_key="BU_A",
        target_key="NW_LEFT",
    )


if __name__ == "__main__":
    main()
