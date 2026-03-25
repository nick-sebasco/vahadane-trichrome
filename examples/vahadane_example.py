"""Trichrome normalization examples.

Run:
python /home/sebasn/vahadane-trichrome/examples/vahadane_example.py

Usage model:
- One example is executed at a time.
- Edit ``ACTIVE_EXAMPLE`` to choose the scenario you want to run.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from vahadane_trichrome import VahadaneTrichromeNormalizer


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "examples", "example_data")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs", "examples")
# Optional pre-resize for faster interactive runs.
# - None: use original full resolution.
# - Integer N: downsample each image by integer stride so longest side <= N.
#   Example: for a 5000px-wide image and N=500, stride~10.
#   This is for quick visual iteration only, not final scientific runs.
MAX_DIM_FOR_EXAMPLES = None # 512

# Optional cap on number of tissue pixels used to learn the stain basis during fit.
# - None: use all tissue pixels (highest fidelity, slowest).
# - Integer M: randomly sample up to M tissue pixels for faster fit.
#   This affects fit speed only; transform still runs on the full input image.
MAX_TISSUE_PIXELS_FOR_EXAMPLES = 8_000

# Dictionary-learning solver used in examples.
# "cd" (coordinate descent) is usually faster than "lars" for large images.
FAST_FIT_ALGORITHM = "cd"

# Sparse-coding solver paired with the fit algorithm above.
# "lasso_cd" is the coordinate-descent variant for transform coding.
FAST_TRANSFORM_ALGORITHM = "lasso_cd"

# Maximum iterations for dictionary-learning fit step in example mode.
# Lower values speed up examples but can reduce convergence quality.
FAST_DL_MAX_ITER = 35

# Maximum iterations for sparse coding inside dictionary learning.
# Higher values may improve convergence but increase runtime.
FAST_DL_TRANSFORM_MAX_ITER = 300


IMAGE_LIBRARY = {
    "BU_A": os.path.join(TEST_DATA_DIR, "BU_2_-_2024-08-23_08.38.00.png"),
    "BU_B": os.path.join(TEST_DATA_DIR, "BU_6_-_2024-08-23_08.50.20.png"),
    "NW_LEFT": os.path.join(
        TEST_DATA_DIR,
        "NW_NW_Trichrome_Box_1_LF_RF_SSCMH27_BASE_LF_04-27-11_-_2022-11-10_16.02.34_left.png",
    ),
    "NW_MULTI": os.path.join(TEST_DATA_DIR, "NW_2_CONCAT.png"),
    "NW_RIGHT": os.path.join(
        TEST_DATA_DIR,
        "NW_NW_Trichrome_Box_1_LF_RF_SSCMH27_BASE_LF_04-27-11_-_2022-11-10_16.02.34_right.png",
    ),
    "KD_LEFT": os.path.join(TEST_DATA_DIR, "KD_V_tri_center.png"),
}


def to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert loaded image arrays to RGB uint8 with shape (H, W, 3)."""
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        rgb = np.repeat(arr[..., None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
        rgb = arr[..., :3]
    elif arr.ndim == 3 and arr.shape[0] in (3, 4):
        rgb = np.moveaxis(arr[:3, ...], 0, -1)
    else:
        raise ValueError(f"Could not convert array with shape {arr.shape} to RGB.")

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255.0 if rgb.max() <= 1.0 else rgb, 0, 255).astype(np.uint8)
    return rgb


def load_image(key: str) -> np.ndarray:
    """Load an image from IMAGE_LIBRARY and return RGB uint8."""
    path = IMAGE_LIBRARY[key]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing image for key '{key}': {path}")
    return to_rgb_uint8(plt.imread(path))


def maybe_downsample_for_example(img: np.ndarray, max_dim: int | None) -> np.ndarray:
    """Optionally downsample image by integer stride for faster example runs."""
    if max_dim is None:
        return img
    h, w = img.shape[:2]
    stride = max(1, int(np.ceil(max(h, w) / max_dim)))
    if stride == 1:
        return img
    return img[::stride, ::stride]


def run_normalization_example(
    *,
    example_name: str,
    input_key: str,
    reference_key: str,
    luminosity_threshold: float = 0.65,
    regularizer: float = 0.1,
    apply_source_tissue_mask: bool = True,
) -> None:
    """Run one normalization example and save all primary artifacts.

    Saves:
    - raw source image
    - raw target image
    - normalized image
    - side-by-side comparison figure
    - source/target ROI mask artifacts
    - stain swatches (human-interpretable RGB mode)
    """
    input_rgb = load_image(input_key)
    reference_rgb = load_image(reference_key)
    input_rgb = maybe_downsample_for_example(input_rgb, MAX_DIM_FOR_EXAMPLES)
    reference_rgb = maybe_downsample_for_example(reference_rgb, MAX_DIM_FOR_EXAMPLES)

    normalizer = VahadaneTrichromeNormalizer(
        n_components=3,
        luminosity_threshold=luminosity_threshold,
        regularizer=regularizer,
        sort_mode="none",
        max_tissue_pixels=MAX_TISSUE_PIXELS_FOR_EXAMPLES,
        random_state=0,
        fit_algorithm=FAST_FIT_ALGORITHM,
        transform_algorithm=FAST_TRANSFORM_ALGORITHM,
        dl_max_iter=FAST_DL_MAX_ITER,
        dl_transform_max_iter=FAST_DL_TRANSFORM_MAX_ITER,
    )
    # IMPORTANT: model is fitted on the REFERENCE image style.
    normalizer.fit(reference_rgb)
    normalized_rgb = normalizer.transform(
        input_rgb,
        apply_source_tissue_mask=apply_source_tissue_mask,
    )

    out_dir = os.path.join(OUTPUT_ROOT, example_name)
    os.makedirs(out_dir, exist_ok=True)

    source_path = os.path.join(out_dir, "source.png")
    target_path = os.path.join(out_dir, "target.png")
    normalized_path = os.path.join(out_dir, "normalized.png")
    plt.imsave(source_path, input_rgb)
    plt.imsave(target_path, reference_rgb)
    plt.imsave(normalized_path, normalized_rgb)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(input_rgb)
    axes[0].set_title(f"Input ({input_key})")
    axes[1].imshow(reference_rgb)
    axes[1].set_title(f"Reference ({reference_key})")
    axes[2].imshow(normalized_rgb)
    axes[2].set_title("Normalized")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    comparison_path = os.path.join(out_dir, "comparison_source_target_normalized.png")
    fig.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    roi_outputs = normalizer.save_roi_images(
        source_img=input_rgb,
        target_img=reference_rgb,
        output_dir=out_dir,
        prefix="run",
    )
    swatch_outputs = normalizer.save_stain_vector_swatches(
        output_dir=out_dir,
        prefix="run",
        rgb=True,
    )

    tissue_intersection = None
    tissue_intersection_note = None
    if normalizer.source_tissue_mask is not None and normalizer.target_tissue_mask is not None:
        if input_rgb.shape == reference_rgb.shape:
            overlap = normalizer.source_tissue_mask & normalizer.target_tissue_mask
            if np.any(overlap):
                src_to_tgt_mae = float(np.mean(np.abs(input_rgb[overlap].astype(np.float32) - reference_rgb[overlap].astype(np.float32))))
                norm_to_tgt_mae = float(np.mean(np.abs(normalized_rgb[overlap].astype(np.float32) - reference_rgb[overlap].astype(np.float32))))
                tissue_intersection = (src_to_tgt_mae, norm_to_tgt_mae)
        else:
            tissue_intersection_note = "Skipped tissue-overlap MAE (input/reference shapes differ)."

    print(f"Example: {example_name}")
    print(f"Input key: {input_key}")
    print(f"Reference key (fit style): {reference_key}")
    print(f"Input shape used: {input_rgb.shape}")
    print(f"Reference shape used: {reference_rgb.shape}")
    print(
        "DL settings used: "
        f"fit={FAST_FIT_ALGORITHM}, transform={FAST_TRANSFORM_ALGORITHM}, "
        f"max_iter={FAST_DL_MAX_ITER}, transform_max_iter={FAST_DL_TRANSFORM_MAX_ITER}, "
        f"max_tissue_pixels={MAX_TISSUE_PIXELS_FOR_EXAMPLES}"
    )
    print(f"Saved source: {source_path}")
    print(f"Saved target: {target_path}")
    print(f"Saved normalized: {normalized_path}")
    print(f"Saved comparison: {comparison_path}")
    print("Saved ROI artifacts:")
    for name, path in roi_outputs.items():
        print(f"  - {name}: {path}")
    print("Saved RGB swatches:")
    for name, path in swatch_outputs.items():
        print(f"  - {name}: {path}")
    if tissue_intersection is not None:
        src_to_tgt_mae, norm_to_tgt_mae = tissue_intersection
        print(f"Tissue-overlap MAE source->target: {src_to_tgt_mae:.2f}")
        print(f"Tissue-overlap MAE normalized->target: {norm_to_tgt_mae:.2f}")
    if tissue_intersection_note is not None:
        print(tissue_intersection_note)
    print("Example run complete.")


def example_bu_to_bu_cross_slide() -> None:
    """Within-cohort, different slide example: input BU_A, reference BU_B.

    Useful for checking normalization across two BU slides from the same cohort.
    """
    run_normalization_example(
        example_name="bu_to_bu_cross_slide",
        input_key="BU_A",
        reference_key="BU_B",
    )


def example_nw_left_to_right_same_slide() -> None:
    """Within-cohort, same-slide split example: input NW_LEFT, reference NW_RIGHT.

    Uses left/right halves from a split NW image as source and target.
    """
    run_normalization_example(
        example_name="nw_left_to_right_same_slide",
        input_key="NW_LEFT",
        reference_key="NW_RIGHT",
    )


def example_kd_to_nw_external() -> None:
    """External cohort example: input KD_LEFT, reference NW_LEFT.

    Applies NW style to KD tissue.
    """
    run_normalization_example(
        example_name="kd_to_nw_external",
        input_key="KD_LEFT",
        reference_key="NW_LEFT",
    )


def example_bu_to_nw_external() -> None:
    """External cohort example: input BU_A, reference NW_LEFT.

    Applies NW style to BU tissue.
    """
    run_normalization_example(
        example_name="bu_to_nw_external",
        input_key="BU_A",
        reference_key="NW_LEFT",
    )


def example_bu_to_multi_target_nw_external() -> None:
    """External cohort example: input BU_A, reference NW_LEFT.

    Applies 2 NW references via handcrafted concat
    to BU tissue.
    """
    run_normalization_example(
        example_name="bu_to_multi_target_nw_external",
        input_key="BU_A",
        reference_key="NW_MULTI",
    )

def main() -> None:
    """Run exactly one example.

    Edit this function directly and call the scenario you want.
    """
    # example_bu_to_bu_cross_slide()  # default
    # example_nw_left_to_right_same_slide()
    # example_kd_to_nw_external()
    #example_bu_to_nw_external()
    example_bu_to_multi_target_nw_external()


if __name__ == "__main__":
    main()
