"""Command-line interface for Vahadane trichrome normalization."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .core import VahadaneTrichromeNormalizer


def _str_to_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert loaded image arrays to RGB uint8 with shape (H, W, 3)."""
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


def _load_rgb_uint8(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return _to_rgb_uint8(plt.imread(path))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vahadane-trichrome",
        description="Run single- or multi-reference Vahadane-style trichrome normalization.",
    )
    parser.add_argument("--source", type=Path, required=True, help="Path to source RGB image.")
    parser.add_argument(
        "--reference",
        type=Path,
        nargs="+",
        required=True,
        help="One or more reference RGB images. One image uses single-reference fit; multiple images use multi-target fit.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Path to normalized output image.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Optional directory for swatches, masks, and run metadata.",
    )
    parser.add_argument(
        "--apply-source-tissue-mask",
        action="store_true",
        help="Apply normalized values only on source tissue and set non-tissue to white.",
    )
    parser.add_argument(
        "--multi-target-aggregation",
        choices=("median", "mean"),
        default="median",
        help="Aggregation used when multiple references are supplied.",
    )
    parser.add_argument(
        "--multi-target-max-workers",
        type=int,
        default=None,
        help="Worker processes for multi-target extraction. Default uses all visible CPUs.",
    )
    parser.add_argument(
        "--multi-target-anchor-index",
        type=int,
        default=None,
        help="Optional explicit anchor index for multi-target stain-matrix alignment.",
    )

    parser.add_argument("--luminosity-threshold", type=float, default=0.8)
    parser.add_argument("--use-connected-components", type=_str_to_bool, default=True)
    parser.add_argument("--min-component-size-fraction", type=float, default=5e-4)
    parser.add_argument("--min-component-size-relative-to-largest", type=float, default=1e-2)
    parser.add_argument("--cumulative-foreground-coverage", type=float, default=0.995)
    parser.add_argument("--connected-components-connectivity", type=int, default=2)
    parser.add_argument("--connected-components-fail-safe", type=_str_to_bool, default=True)

    parser.add_argument("--regularizer", type=float, default=0.1)
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--sort-mode", choices=("none", "dominant_channel"), default="none")
    parser.add_argument("--max-tissue-pixels", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--fit-algorithm", choices=("lars", "cd"), default="lars")
    parser.add_argument(
        "--transform-algorithm",
        choices=("lasso_lars", "lasso_cd", "omp", "threshold"),
        default="lasso_lars",
    )
    parser.add_argument("--dl-max-iter", type=int, default=100)
    parser.add_argument("--dl-transform-max-iter", type=int, default=1000)
    parser.add_argument(
        "--dl-n-jobs",
        type=int,
        default=-1,
        help="Thread count passed to sklearn DictionaryLearning. Use -1 for all visible CPUs.",
    )
    parser.add_argument(
        "--max-concentration-scale-factor",
        type=float,
        default=4.0,
        help="Upper bound on per-channel percentile scaling during transform. Use 0 or a negative value only if you intentionally disable clipping in code.",
    )

    parser.add_argument(
        "--save-swatches",
        action="store_true",
        help="Save stain vector swatches into artifact directory.",
    )
    parser.add_argument(
        "--save-roi-images",
        action="store_true",
        help="Save source/target ROI masks and crops into artifact directory.",
    )
    parser.add_argument(
        "--save-fit-state",
        type=Path,
        default=None,
        help="Optional .npz path for saving fitted target state after fit.",
    )
    return parser


def _make_normalizer(args: argparse.Namespace) -> VahadaneTrichromeNormalizer:
    max_concentration_scale_factor = (
        args.max_concentration_scale_factor
        if args.max_concentration_scale_factor > 0
        else None
    )
    return VahadaneTrichromeNormalizer(
        luminosity_threshold=args.luminosity_threshold,
        use_connected_components=args.use_connected_components,
        min_component_size_fraction=args.min_component_size_fraction,
        min_component_size_relative_to_largest=args.min_component_size_relative_to_largest,
        cumulative_foreground_coverage=args.cumulative_foreground_coverage,
        connected_components_connectivity=args.connected_components_connectivity,
        connected_components_fail_safe=args.connected_components_fail_safe,
        regularizer=args.regularizer,
        n_components=args.n_components,
        sort_mode=args.sort_mode,
        max_tissue_pixels=args.max_tissue_pixels,
        random_state=args.random_state,
        fit_algorithm=args.fit_algorithm,
        transform_algorithm=args.transform_algorithm,
        dl_max_iter=args.dl_max_iter,
        dl_transform_max_iter=args.dl_transform_max_iter,
        dl_n_jobs=args.dl_n_jobs,
        max_concentration_scale_factor=max_concentration_scale_factor,
    )


def run_cli(args: argparse.Namespace) -> int:
    source_rgb = _load_rgb_uint8(args.source)
    reference_rgbs = [_load_rgb_uint8(path) for path in args.reference]

    if (args.save_swatches or args.save_roi_images) and args.artifact_dir is None:
        raise ValueError("--save-swatches and --save-roi-images require --artifact-dir.")
    if args.save_roi_images and len(reference_rgbs) != 1:
        raise ValueError("--save-roi-images currently requires exactly one reference image.")

    normalizer = _make_normalizer(args)
    if len(reference_rgbs) == 1:
        normalizer.fit(reference_rgbs[0])
        fit_mode = "single_target"
    else:
        normalizer.fit_multi_target(
            reference_rgbs,
            aggregation=args.multi_target_aggregation,
            max_workers=args.multi_target_max_workers,
            anchor_index=args.multi_target_anchor_index,
        )
        fit_mode = "multi_target"

    normalized_rgb = normalizer.transform(
        source_rgb,
        apply_source_tissue_mask=args.apply_source_tissue_mask,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(args.output, normalized_rgb)

    artifact_dir = args.artifact_dir
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "source": str(args.source),
            "references": [str(path) for path in args.reference],
            "output": str(args.output),
            "fit_mode": fit_mode,
            "multi_target_aggregation": args.multi_target_aggregation if len(reference_rgbs) > 1 else None,
            "multi_target_max_workers": args.multi_target_max_workers,
            "multi_target_anchor_index": args.multi_target_anchor_index,
            "luminosity_threshold": args.luminosity_threshold,
            "use_connected_components": args.use_connected_components,
            "min_component_size_fraction": args.min_component_size_fraction,
            "min_component_size_relative_to_largest": args.min_component_size_relative_to_largest,
            "cumulative_foreground_coverage": args.cumulative_foreground_coverage,
            "connected_components_connectivity": args.connected_components_connectivity,
            "connected_components_fail_safe": args.connected_components_fail_safe,
            "regularizer": args.regularizer,
            "n_components": args.n_components,
            "sort_mode": args.sort_mode,
            "max_tissue_pixels": args.max_tissue_pixels,
            "random_state": args.random_state,
            "fit_algorithm": args.fit_algorithm,
            "transform_algorithm": args.transform_algorithm,
            "dl_max_iter": args.dl_max_iter,
            "dl_transform_max_iter": args.dl_transform_max_iter,
            "dl_n_jobs": args.dl_n_jobs,
            "max_concentration_scale_factor": args.max_concentration_scale_factor,
        }
        (artifact_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        if args.save_swatches:
            normalizer.save_stain_vector_swatches(str(artifact_dir), prefix="run", rgb=True)
        if args.save_roi_images and len(reference_rgbs) == 1:
            normalizer.save_roi_images(
                source_img=source_rgb,
                target_img=reference_rgbs[0],
                output_dir=str(artifact_dir),
                prefix="run",
            )

    if args.save_fit_state is not None:
        args.save_fit_state.parent.mkdir(parents=True, exist_ok=True)
        normalizer.save_fit_state(str(args.save_fit_state), metadata={"fit_mode": fit_mode})

    print(f"Saved normalized image: {args.output}")
    print(f"Fit mode: {fit_mode}")
    if len(reference_rgbs) > 1:
        print(f"Multi-target aggregation: {args.multi_target_aggregation}")
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())