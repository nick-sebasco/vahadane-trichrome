"""Image-driven sanity checks for Vahadane trichrome normalization."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from vahadane_trichrome import VahadaneTrichromeNormalizer


def _make_toy_tissue_image(
    height: int = 192,
    width: int = 192,
    *,
    square_rgb: tuple[int, int, int] = (210, 40, 40),
    line_rgb: tuple[int, int, int] = (30, 80, 220),
    dot_rgb: tuple[int, int, int] = (20, 20, 20),
) -> tuple[np.ndarray, np.ndarray]:
    """Create a paint-style toy image with one square tissue region."""
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
    """Choose persistent test output directory when present, else use tmp_path."""
    persistent_dir = Path(__file__).resolve().parent / "test_outputs"
    if persistent_dir.is_dir():
        return persistent_dir
    return tmp_path


def _simple_edge_map(gray: np.ndarray, threshold: float = 18.0) -> np.ndarray:
    """Compute a lightweight edge map from finite differences."""
    gray = gray.astype(np.float32)
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    grad_mag = np.sqrt(gx * gx + gy * gy)
    return grad_mag >= threshold


def _mean_tissue_gray(img: np.ndarray, tissue_mask: np.ndarray) -> float:
    """Return mean grayscale intensity inside the provided tissue mask."""
    return float(np.mean(img[tissue_mask].astype(np.float32), axis=1).mean())


def _mean_swatch_gray(stain_matrix: np.ndarray) -> float:
    """Return mean grayscale intensity of 1x1 RGB swatches for a stain matrix."""
    swatch_colors = VahadaneTrichromeNormalizer._stain_matrix_to_swatch_image(
        stain_matrix,
        swatch_height=1,
        swatch_width=1,
        rgb=True,
    )[0, :, :].astype(np.float32)
    return float(np.mean(swatch_colors))


def test_toy_paint_image_sanity_white_background_is_wiped_out_with_source_mask(tmp_path: Path) -> None:
    """Masking should preserve white background outside the known toy tissue square."""
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
    assert "target_swatches" in swatch_outputs
    assert "source_aligned_swatches" in swatch_outputs
    assert "source_raw_swatches" not in swatch_outputs

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
    """Changing the toy square from red to green should change the saved swatches."""
    output_dir = _resolve_test_output_dir(tmp_path)

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


def test_toy_normalized_tissue_moves_toward_darker_target_when_source_square_is_lighter(
    tmp_path: Path,
) -> None:
    """A darker target should pull a lighter toy source toward lower tissue intensity.

    This is an image-level sanity check motivated by real runs where saved RGB
    swatches and final normalized tissue can appear to tell different stories.
    The toy setup isolates a simple case:
    - Source square is light red.
    - Target square is dark red.

    We verify two things separately:
    1) The learned target swatches are darker than the aligned source swatches.
    2) The normalized tissue moves darker than the source and closer to the
       target's tissue brightness, rather than drifting in the wrong direction.
    """
    output_dir = _resolve_test_output_dir(tmp_path)

    source_img, expected_tissue_mask = _make_toy_tissue_image(
        height=128,
        width=128,
        square_rgb=(230, 170, 170),
        line_rgb=(30, 80, 220),
        dot_rgb=(20, 20, 20),
    )
    target_img, _ = _make_toy_tissue_image(
        height=128,
        width=128,
        square_rgb=(125, 40, 40),
        line_rgb=(30, 80, 220),
        dot_rgb=(20, 20, 20),
    )

    normalizer = VahadaneTrichromeNormalizer(
        n_components=3,
        luminosity_threshold=0.85,
        regularizer=0.1,
        sort_mode="none",
    )
    normalizer.fit(target_img)
    transformed = normalizer.transform(source_img, apply_source_tissue_mask=True)

    source_path = output_dir / "toy_light_source.png"
    target_path = output_dir / "toy_dark_target.png"
    transformed_path = output_dir / "toy_light_to_dark_transformed.png"
    plt.imsave(source_path, source_img)
    plt.imsave(target_path, target_img)
    plt.imsave(transformed_path, transformed)

    swatch_outputs = normalizer.save_stain_vector_swatches(
        output_dir=str(output_dir),
        prefix="toy_light_to_dark",
        rgb=True,
    )

    assert normalizer.stain_matrix_target is not None
    assert normalizer.stain_matrix_source_aligned is not None
    assert normalizer.source_tissue_mask is not None

    target_swatch_gray = _mean_swatch_gray(normalizer.stain_matrix_target)
    source_aligned_swatch_gray = _mean_swatch_gray(normalizer.stain_matrix_source_aligned)
    source_tissue_gray = _mean_tissue_gray(source_img, expected_tissue_mask)
    target_tissue_gray = _mean_tissue_gray(target_img, expected_tissue_mask)
    transformed_tissue_gray = _mean_tissue_gray(transformed, expected_tissue_mask)

    print(f"Toy light source saved: {source_path}")
    print(f"Toy dark target saved: {target_path}")
    print(f"Toy transformed saved: {transformed_path}")
    for name, path in swatch_outputs.items():
        print(f"Swatch artifact [{name}]: {path}")
    print(
        "Mean gray summary: "
        f"source_swatch={source_aligned_swatch_gray:.2f}, "
        f"target_swatch={target_swatch_gray:.2f}, "
        f"source_tissue={source_tissue_gray:.2f}, "
        f"target_tissue={target_tissue_gray:.2f}, "
        f"transformed_tissue={transformed_tissue_gray:.2f}"
    )

    assert target_swatch_gray < source_aligned_swatch_gray
    assert transformed_tissue_gray < source_tissue_gray - 5.0
    assert abs(transformed_tissue_gray - target_tissue_gray) < abs(source_tissue_gray - target_tissue_gray)
    assert np.all(transformed[~expected_tissue_mask] == 255)


def test_end_to_end_identity_same_source_target_preserves_toy_structure(tmp_path: Path) -> None:
    """Using the same toy image for fit and transform should preserve structure."""
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

    inside_mae = float(
        np.mean(np.abs(transformed[tissue].astype(np.float32) - source_img[tissue].astype(np.float32)))
    )
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


def test_real_trichrome_integration_runs_and_logs_outputs(tmp_path: Path) -> None:
    """Real-image integration should run and save artifacts for visual inspection."""
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
    assert "source_aligned_swatches" in swatch_outputs
    assert "source_raw_swatches" not in swatch_outputs

    assert normalizer.source_tissue_mask is not None


def test_save_stain_vector_swatches_can_include_raw_source_for_debugging(tmp_path: Path) -> None:
    """Raw source swatches remain available as an explicit debugging artifact."""
    target_img, _ = _make_toy_tissue_image(
        height=96,
        width=96,
        square_rgb=(170, 70, 70),
        line_rgb=(80, 120, 200),
        dot_rgb=(35, 35, 35),
    )
    source_img, _ = _make_toy_tissue_image(
        height=96,
        width=96,
        square_rgb=(70, 170, 70),
        line_rgb=(80, 120, 200),
        dot_rgb=(35, 35, 35),
    )

    normalizer = VahadaneTrichromeNormalizer(sort_mode="none")
    normalizer.fit(target_img)
    normalizer.transform(source_img)

    swatch_outputs = normalizer.save_stain_vector_swatches(
        output_dir=str(tmp_path),
        prefix="debug",
        rgb=True,
        include_source_raw=True,
    )

    assert "source_aligned_swatches" in swatch_outputs
    assert "source_raw_swatches" in swatch_outputs


def test_save_load_fit_state_round_trip_preserves_transform_behavior(tmp_path: Path) -> None:
    """Saved target fit state should reproduce the same toy-image transform."""
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