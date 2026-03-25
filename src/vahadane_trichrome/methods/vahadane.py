"""Core Vahadane-style extractor and normalizer implementation."""

import concurrent.futures
import itertools
import json
import logging
import os
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import NMF

from ..utils import get_tissue_mask
from ..utils import rgb2od

logger = logging.getLogger(__name__)


def _normalize_extractor_backend_name(backend: str) -> str:
    """Return a canonical extractor backend name."""
    normalized = backend.strip().lower().replace("-", "_")
    if normalized not in {"dictionary_learning", "nmf"}:
        raise ValueError(
            "backend must be one of {'dictionary_learning', 'nmf'}, "
            f"got {backend!r}."
        )
    return normalized


def _unit_row_normalize(matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized matrix with epsilon guard for zero rows."""
    arr = np.asarray(matrix, dtype=np.float64)
    return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-8)


def _validate_nmf_configuration(solver: str, beta_loss: str) -> None:
    """Reject unsupported sklearn NMF solver/loss combinations early."""
    if solver == "cd" and beta_loss != "frobenius":
        raise ValueError(
            "NMF solver='cd' only supports beta_loss='frobenius'. "
            f"Got beta_loss={beta_loss!r}."
        )


def _resolve_backend_regularizer(backend: str, regularizer: float | None) -> float:
    """Resolve backend-specific default regularization strength."""
    if regularizer is not None:
        return float(regularizer)
    if backend == "nmf":
        return 1e-4
    return 0.1


def _dl_output_for_h_and_e(dictionary: np.ndarray) -> np.ndarray:
    """Sort a 2-stain dictionary by blue-channel intensity (descending)."""
    matrix = np.asarray(dictionary)
    if matrix.shape != (2, 3):
        return matrix
    order = np.argsort(-matrix[:, 2])
    return matrix[order]


def _sort_dictionary_by_dominant_channel(dictionary: np.ndarray) -> np.ndarray:
    """Return a deterministic, channel-based ordering of stain vectors."""
    dominant_channel = np.argmax(dictionary, axis=1)
    dominant_value = np.max(dictionary, axis=1)
    order = np.lexsort((-dominant_value, dominant_channel))
    return dictionary[order]


def _match_source_rows_to_target(
    source_dictionary: np.ndarray,
    target_dictionary: np.ndarray,
) -> np.ndarray:
    """Align source stain rows to target stain rows by best global similarity."""
    best_perm = _get_best_alignment_permutation(source_dictionary, target_dictionary)
    return source_dictionary[list(best_perm), :]


def _get_best_alignment_permutation(
    source_dictionary: np.ndarray,
    target_dictionary: np.ndarray,
) -> tuple[int, ...]:
    """Return the best row permutation aligning source to target."""
    if source_dictionary.shape != target_dictionary.shape:
        msg = (
            "Source and target stain matrices must have the same shape, got "
            f"{source_dictionary.shape} and {target_dictionary.shape}."
        )
        raise ValueError(msg)

    source_norm = _unit_row_normalize(source_dictionary)
    target_norm = _unit_row_normalize(target_dictionary)
    similarity = source_norm @ target_norm.T

    n_components = source_dictionary.shape[0]
    best_perm = tuple(range(n_components))
    best_score = -np.inf
    for perm in itertools.permutations(range(n_components)):
        score = similarity[list(perm), range(n_components)].sum()
        if score > best_score:
            best_score = score
            best_perm = perm

    return tuple(best_perm)


def _stain_matrix_alignment_score(
    source_dictionary: np.ndarray,
    target_dictionary: np.ndarray,
) -> float:
    """Return best global cosine-similarity score between two stain matrices."""
    if source_dictionary.shape != target_dictionary.shape:
        msg = (
            "Source and target stain matrices must have the same shape, got "
            f"{source_dictionary.shape} and {target_dictionary.shape}."
        )
        raise ValueError(msg)

    source_norm = _unit_row_normalize(source_dictionary)
    target_norm = _unit_row_normalize(target_dictionary)
    similarity = source_norm @ target_norm.T
    best_perm = _get_best_alignment_permutation(source_dictionary, target_dictionary)
    return float(similarity[list(best_perm), range(source_dictionary.shape[0])].sum())


def _select_alignment_anchor_index(stain_matrices: Sequence[np.ndarray]) -> int:
    """Choose the medoid-like anchor index for a list of stain matrices."""
    if len(stain_matrices) == 0:
        raise ValueError("stain_matrices must contain at least one matrix.")
    if len(stain_matrices) == 1:
        return 0

    scores = np.zeros(len(stain_matrices), dtype=np.float64)
    for i, anchor in enumerate(stain_matrices):
        for matrix in stain_matrices:
            scores[i] += _stain_matrix_alignment_score(matrix, anchor)
    return int(np.argmax(scores))


def _align_stain_matrices_to_anchor(
    stain_matrices: Sequence[np.ndarray],
    anchor_index: int | None = None,
) -> tuple[list[np.ndarray], list[tuple[int, ...]], int]:
    """Align all stain matrices to a shared anchor basis."""
    if len(stain_matrices) == 0:
        raise ValueError("stain_matrices must contain at least one matrix.")

    resolved_anchor_index = (
        _select_alignment_anchor_index(stain_matrices)
        if anchor_index is None
        else int(anchor_index)
    )
    anchor = stain_matrices[resolved_anchor_index]
    permutations = [
        _get_best_alignment_permutation(matrix, anchor)
        for matrix in stain_matrices
    ]
    aligned = [
        matrix[list(perm), :]
        for matrix, perm in zip(stain_matrices, permutations)
    ]
    return aligned, permutations, resolved_anchor_index


def _aggregate_stain_matrices(
    stain_matrices: Sequence[np.ndarray],
    method: str = "median",
) -> np.ndarray:
    """Aggregate aligned stain matrices across references."""
    if len(stain_matrices) == 0:
        raise ValueError("stain_matrices must contain at least one matrix.")

    stack = np.stack(stain_matrices, axis=0)
    if method == "median":
        aggregated = np.median(stack, axis=0)
    elif method == "mean":
        aggregated = np.mean(stack, axis=0)
    else:
        raise ValueError(f"Unsupported aggregation method: {method}")

    return aggregated / np.maximum(np.linalg.norm(aggregated, axis=1, keepdims=True), 1e-8)


def _extract_single_reference_state(
    target_img: np.ndarray,
    extractor_params: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-reference stain matrix and channel scaling statistics."""
    extractor = VahadaneTrichromeExtractor(**extractor_params)
    stain_matrix = extractor.get_stain_matrix(target_img)
    target_tissue_mask = extractor.last_tissue_mask

    target_concentrations = VahadaneTrichromeNormalizer.get_concentrations(target_img, stain_matrix)
    target_mask_flat = target_tissue_mask.reshape(-1) if target_tissue_mask is not None else None
    if target_mask_flat is not None and np.any(target_mask_flat):
        target_concentrations_for_scale = target_concentrations[target_mask_flat]
        target_od_for_cap = rgb2od(target_img).reshape((-1, 3))[target_mask_flat]
    else:
        target_concentrations_for_scale = target_concentrations
        target_od_for_cap = rgb2od(target_img).reshape((-1, 3))
    max_c_target = np.percentile(target_concentrations_for_scale, 99, axis=0, keepdims=True)
    max_od_target = np.percentile(target_od_for_cap, 99, axis=0, keepdims=True)
    return stain_matrix, max_c_target, max_od_target


class VahadaneTrichromeExtractor:
    """Vahadane-style stain extractor generalized to arbitrary stain count."""

    def __init__(
        self,
        backend: str = "dictionary_learning",
        luminosity_threshold: float = 0.8,
        use_connected_components: bool = True,
        min_component_size_fraction: float = 5e-4,
        min_component_size_relative_to_largest: float = 1e-2,
        cumulative_foreground_coverage: float = 0.995,
        connected_components_connectivity: int = 2,
        connected_components_fail_safe: bool = True,
        regularizer: float | None = None,
        n_components: int = 3,
        sort_mode: str = "none",
        max_tissue_pixels: int | None = None,
        random_state: int = 0,
        fit_algorithm: str = "lars",
        transform_algorithm: str = "lasso_lars",
        dl_max_iter: int = 100,
        dl_transform_max_iter: int = 1000,
        dl_n_jobs: int | None = -1,
        nmf_init: str = "nndsvdar",
        nmf_solver: str = "cd",
        nmf_beta_loss: str = "frobenius",
        nmf_tol: float = 1e-4,
        nmf_max_iter: int = 3000,
        nmf_shuffle: bool = False,
    ) -> None:
        """Initialize :class:`VahadaneTrichromeExtractor`."""
        resolved_backend = _normalize_extractor_backend_name(backend)
        if resolved_backend == "dictionary_learning":
            logger.warning(
                "Vahadane stain extraction/normalization algorithms are unstable "
                "after the update to `dictionary learning` algorithm in "
                "scikit-learn > v0.23.0 (see issue #382). Please be advised and "
                "consider using other stain extraction (normalization) algorithms.",
                stacklevel=2,
            )
        else:
            _validate_nmf_configuration(nmf_solver, nmf_beta_loss)
        self.__backend = resolved_backend
        self.__regularizer = _resolve_backend_regularizer(resolved_backend, regularizer)
        self.__luminosity_threshold = luminosity_threshold
        self.__use_connected_components = use_connected_components
        self.__min_component_size_fraction = min_component_size_fraction
        self.__min_component_size_relative_to_largest = min_component_size_relative_to_largest
        self.__cumulative_foreground_coverage = cumulative_foreground_coverage
        self.__connected_components_connectivity = connected_components_connectivity
        self.__connected_components_fail_safe = connected_components_fail_safe
        self.__n_components = n_components
        self.__sort_mode = sort_mode
        self.__max_tissue_pixels = max_tissue_pixels
        self.__random_state = random_state
        self.__fit_algorithm = fit_algorithm
        self.__transform_algorithm = transform_algorithm
        self.__dl_max_iter = dl_max_iter
        self.__dl_transform_max_iter = dl_transform_max_iter
        self.__dl_n_jobs = dl_n_jobs
        self.__nmf_init = nmf_init
        self.__nmf_solver = nmf_solver
        self.__nmf_beta_loss = nmf_beta_loss
        self.__nmf_tol = nmf_tol
        self.__nmf_max_iter = nmf_max_iter
        self.__nmf_shuffle = nmf_shuffle
        self._last_tissue_mask: np.ndarray | None = None

    def get_params(self) -> dict:
        """Return extractor initialization parameters."""
        return {
            "backend": self.__backend,
            "luminosity_threshold": self.__luminosity_threshold,
            "use_connected_components": self.__use_connected_components,
            "min_component_size_fraction": self.__min_component_size_fraction,
            "min_component_size_relative_to_largest": self.__min_component_size_relative_to_largest,
            "cumulative_foreground_coverage": self.__cumulative_foreground_coverage,
            "connected_components_connectivity": self.__connected_components_connectivity,
            "connected_components_fail_safe": self.__connected_components_fail_safe,
            "regularizer": self.__regularizer,
            "n_components": self.__n_components,
            "sort_mode": self.__sort_mode,
            "max_tissue_pixels": self.__max_tissue_pixels,
            "random_state": self.__random_state,
            "fit_algorithm": self.__fit_algorithm,
            "transform_algorithm": self.__transform_algorithm,
            "dl_max_iter": self.__dl_max_iter,
            "dl_transform_max_iter": self.__dl_transform_max_iter,
            "dl_n_jobs": self.__dl_n_jobs,
            "nmf_init": self.__nmf_init,
            "nmf_solver": self.__nmf_solver,
            "nmf_beta_loss": self.__nmf_beta_loss,
            "nmf_tol": self.__nmf_tol,
            "nmf_max_iter": self.__nmf_max_iter,
            "nmf_shuffle": self.__nmf_shuffle,
        }

    @property
    def backend(self) -> str:
        """Return the configured stain-extraction backend."""
        return self.__backend

    @property
    def last_tissue_mask(self) -> np.ndarray | None:
        """Return the most recent 2D tissue mask used during extraction."""
        if self._last_tissue_mask is None:
            return None
        return self._last_tissue_mask.copy()

    def _prepare_tissue_od_matrix(self, img: np.ndarray) -> np.ndarray:
        """Return masked OD pixels used for stain estimation."""
        img = img.astype("uint8")

        tissue_mask_2d = get_tissue_mask(
            img,
            luminosity_threshold=self.__luminosity_threshold,
            use_connected_components=self.__use_connected_components,
            min_component_size_fraction=self.__min_component_size_fraction,
            min_component_size_relative_to_largest=self.__min_component_size_relative_to_largest,
            cumulative_foreground_coverage=self.__cumulative_foreground_coverage,
            connected_components_connectivity=self.__connected_components_connectivity,
            connected_components_fail_safe=self.__connected_components_fail_safe,
        )
        self._last_tissue_mask = tissue_mask_2d.astype(bool, copy=False)
        tissue_mask = tissue_mask_2d.reshape((-1,))
        img_od = rgb2od(img).reshape((-1, 3))
        img_od = img_od[tissue_mask]

        max_tissue_pixels = self.__max_tissue_pixels
        if max_tissue_pixels is not None and img_od.shape[0] > max_tissue_pixels:
            rng = np.random.default_rng(self.__random_state)
            sample_idx = rng.choice(img_od.shape[0], size=max_tissue_pixels, replace=False)
            img_od = img_od[sample_idx]

        return img_od

    def _estimate_stain_matrix_dictionary_learning(self, img_od: np.ndarray) -> np.ndarray:
        """Estimate stain matrix with sklearn DictionaryLearning."""
        dl = DictionaryLearning(
            n_components=self.__n_components,
            alpha=self.__regularizer,
            transform_alpha=self.__regularizer,
            fit_algorithm=self.__fit_algorithm,
            transform_algorithm=self.__transform_algorithm,
            positive_dict=True,
            verbose=False,
            max_iter=self.__dl_max_iter,
            transform_max_iter=self.__dl_transform_max_iter,
            random_state=self.__random_state,
            n_jobs=self.__dl_n_jobs,
        )
        dl.fit(X=img_od)
        return dl.components_

    def _estimate_stain_matrix_nmf(self, img_od: np.ndarray) -> np.ndarray:
        """Estimate stain matrix with sparse nonnegative matrix factorization."""
        nmf = NMF(
            n_components=self.__n_components,
            init=self.__nmf_init,
            solver=self.__nmf_solver,
            beta_loss=self.__nmf_beta_loss,
            tol=self.__nmf_tol,
            max_iter=self.__nmf_max_iter,
            random_state=self.__random_state,
            alpha_W=0.0,
            alpha_H=self.__regularizer,
            l1_ratio=1.0,
            shuffle=self.__nmf_shuffle,
        )
        nmf.fit_transform(img_od)
        return nmf.components_

    def _postprocess_stain_matrix(self, stain_matrix: np.ndarray) -> np.ndarray:
        """Apply deterministic ordering and unit-row normalization."""
        matrix = np.asarray(stain_matrix, dtype=np.float64)
        if self.__n_components == 2:
            matrix = _dl_output_for_h_and_e(matrix)
        elif self.__sort_mode == "dominant_channel":
            matrix = _sort_dictionary_by_dominant_channel(matrix)

        return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)

    def _validate_estimated_stain_matrix(self, stain_matrix: np.ndarray) -> None:
        """Raise if the estimated stain basis is degenerate."""
        if self.__backend != "nmf":
            return

        nonzero_rows = int(np.sum(np.linalg.norm(stain_matrix, axis=1) > 1e-6))
        if nonzero_rows < self.__n_components:
            raise RuntimeError(
                "NMF backend produced a degenerate stain matrix with "
                f"{nonzero_rows}/{self.__n_components} nonzero components. "
                "Lower the NMF regularizer or adjust the NMF settings."
            )

    def get_stain_matrix(self, img: np.ndarray) -> np.ndarray:
        """Stain matrix estimation."""
        img_od = self._prepare_tissue_od_matrix(img)

        if self.__backend == "dictionary_learning":
            stain_matrix = self._estimate_stain_matrix_dictionary_learning(img_od)
        else:
            stain_matrix = self._estimate_stain_matrix_nmf(img_od)

        stain_matrix = self._postprocess_stain_matrix(stain_matrix)
        self._validate_estimated_stain_matrix(stain_matrix)
        return stain_matrix


class VahadaneTrichromeNormalizer:
    """Generic Vahadane-style normalizer supporting arbitrary number of stains."""

    def __init__(
        self,
        extractor: VahadaneTrichromeExtractor | None = None,
        *,
        backend: str = "dictionary_learning",
        luminosity_threshold: float = 0.8,
        use_connected_components: bool = True,
        min_component_size_fraction: float = 5e-4,
        min_component_size_relative_to_largest: float = 1e-2,
        cumulative_foreground_coverage: float = 0.995,
        connected_components_connectivity: int = 2,
        connected_components_fail_safe: bool = True,
        regularizer: float | None = None,
        n_components: int = 3,
        sort_mode: str = "none",
        max_tissue_pixels: int | None = None,
        random_state: int = 0,
        fit_algorithm: str = "lars",
        transform_algorithm: str = "lasso_lars",
        dl_max_iter: int = 100,
        dl_transform_max_iter: int = 1000,
        dl_n_jobs: int | None = -1,
        nmf_init: str = "nndsvdar",
        nmf_solver: str = "cd",
        nmf_beta_loss: str = "frobenius",
        nmf_tol: float = 1e-4,
        nmf_max_iter: int = 3000,
        nmf_shuffle: bool = False,
        max_concentration_scale_factor: float | None = 4.0,
    ) -> None:
        self.extractor = extractor or VahadaneTrichromeExtractor(
            backend=backend,
            luminosity_threshold=luminosity_threshold,
            use_connected_components=use_connected_components,
            min_component_size_fraction=min_component_size_fraction,
            min_component_size_relative_to_largest=min_component_size_relative_to_largest,
            cumulative_foreground_coverage=cumulative_foreground_coverage,
            connected_components_connectivity=connected_components_connectivity,
            connected_components_fail_safe=connected_components_fail_safe,
            regularizer=regularizer,
            n_components=n_components,
            sort_mode=sort_mode,
            max_tissue_pixels=max_tissue_pixels,
            random_state=random_state,
            fit_algorithm=fit_algorithm,
            transform_algorithm=transform_algorithm,
            dl_max_iter=dl_max_iter,
            dl_transform_max_iter=dl_transform_max_iter,
            dl_n_jobs=dl_n_jobs,
            nmf_init=nmf_init,
            nmf_solver=nmf_solver,
            nmf_beta_loss=nmf_beta_loss,
            nmf_tol=nmf_tol,
            nmf_max_iter=nmf_max_iter,
            nmf_shuffle=nmf_shuffle,
        )
        self.max_concentration_scale_factor = max_concentration_scale_factor
        self.stain_matrix_target: np.ndarray | None = None
        self.stain_matrix_source_raw: np.ndarray | None = None
        self.stain_matrix_source_aligned: np.ndarray | None = None
        self.max_c_target: np.ndarray | None = None
        self.max_od_target: np.ndarray | None = None
        self.target_tissue_mask: np.ndarray | None = None
        self.source_tissue_mask: np.ndarray | None = None
        self.fit_metadata: dict | None = None

    @staticmethod
    def _stain_matrix_to_swatch_image(
        stain_matrix: np.ndarray,
        swatch_height: int = 80,
        swatch_width: int = 140,
        rgb: bool = False,
        rgb_strength: float = 2.5,
    ) -> np.ndarray:
        """Convert a stain matrix into an RGB swatch strip image."""
        matrix = np.asarray(stain_matrix, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != 3:
            raise ValueError(f"Expected stain_matrix shape (n_components, 3), got {matrix.shape}.")

        if rgb:
            row_max = np.maximum(matrix.max(axis=1, keepdims=True), 1e-8)
            relative_absorbance = np.clip(matrix / row_max, 0.0, 1.0)
            colors = np.clip((1.0 - relative_absorbance) * 255.0, 0, 255).astype(np.uint8)
        else:
            denom = np.maximum(matrix.max(axis=1, keepdims=True), 1e-8)
            colors = np.clip((matrix / denom) * 255.0, 0, 255).astype(np.uint8)

        canvas = np.full((swatch_height, swatch_width * matrix.shape[0], 3), 255, dtype=np.uint8)
        for idx in range(matrix.shape[0]):
            canvas[:, idx * swatch_width:(idx + 1) * swatch_width] = colors[idx]
        return canvas

    def save_stain_vector_swatches(
        self,
        output_dir: str,
        prefix: str = "vahadane_stains",
        rgb: bool = False,
        include_source_raw: bool = False,
    ) -> dict[str, str]:
        """Save stain vector swatches, preferring post-alignment source ordering.

        By default, source swatches are exported only from the aligned source
        stain matrix so the saved strip matches the fitted target ordering used
        during normalization. Pre-alignment source swatches remain available as
        an explicit debug artifact via ``include_source_raw=True``.
        """
        if self.stain_matrix_target is None:
            raise RuntimeError("Target stain matrix is not available. Run fit() first.")

        os.makedirs(output_dir, exist_ok=True)
        outputs: dict[str, str] = {}
        mode_tag = "rgb" if rgb else "od"

        target_swatch = self._stain_matrix_to_swatch_image(self.stain_matrix_target, rgb=rgb)
        target_path = os.path.join(output_dir, f"{prefix}_{mode_tag}_target_swatches.png")
        plt.imsave(target_path, target_swatch)
        outputs["target_swatches"] = target_path

        if self.stain_matrix_source_aligned is not None:
            source_aligned_swatch = self._stain_matrix_to_swatch_image(self.stain_matrix_source_aligned, rgb=rgb)
            source_aligned_path = os.path.join(output_dir, f"{prefix}_{mode_tag}_source_aligned_swatches.png")
            plt.imsave(source_aligned_path, source_aligned_swatch)
            outputs["source_aligned_swatches"] = source_aligned_path

        if include_source_raw and self.stain_matrix_source_raw is not None:
            source_raw_swatch = self._stain_matrix_to_swatch_image(self.stain_matrix_source_raw, rgb=rgb)
            source_raw_path = os.path.join(output_dir, f"{prefix}_{mode_tag}_source_raw_swatches.png")
            plt.imsave(source_raw_path, source_raw_swatch)
            outputs["source_raw_swatches"] = source_raw_path

        return outputs

    @staticmethod
    def _apply_tissue_mask(img: np.ndarray, tissue_mask: np.ndarray) -> np.ndarray:
        """Return RGB image where non-tissue pixels are set to white."""
        roi = np.full_like(img, 255)
        roi[tissue_mask] = img[tissue_mask]
        return roi

    @staticmethod
    def _crop_to_tissue_bbox(img: np.ndarray, tissue_mask: np.ndarray) -> np.ndarray:
        """Return tight bounding-box crop around mask; fallback to original image if empty."""
        ys, xs = np.where(tissue_mask)
        if ys.size == 0 or xs.size == 0:
            return img
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        return img[y0:y1, x0:x1]

    def save_roi_images(
        self,
        source_img: np.ndarray,
        target_img: np.ndarray,
        output_dir: str,
        prefix: str = "vahadane_roi",
        cmap: str = "inferno",
    ) -> dict[str, str]:
        """Save source/target tissue masks, masked full images, and cropped ROIs."""
        if self.target_tissue_mask is None or self.source_tissue_mask is None:
            msg = "Target/source masks are not available. Run fit() and transform() first."
            raise RuntimeError(msg)

        os.makedirs(output_dir, exist_ok=True)

        source_roi = self._apply_tissue_mask(source_img, self.source_tissue_mask)
        target_roi = self._apply_tissue_mask(target_img, self.target_tissue_mask)
        source_roi_crop = self._crop_to_tissue_bbox(source_roi, self.source_tissue_mask)
        target_roi_crop = self._crop_to_tissue_bbox(target_roi, self.target_tissue_mask)

        source_mask_path = os.path.join(output_dir, f"{prefix}_source_mask.png")
        target_mask_path = os.path.join(output_dir, f"{prefix}_target_mask.png")
        source_masked_path = os.path.join(output_dir, f"{prefix}_source_masked_full.png")
        target_masked_path = os.path.join(output_dir, f"{prefix}_target_masked_full.png")
        source_roi_crop_path = os.path.join(output_dir, f"{prefix}_source_roi_crop.png")
        target_roi_crop_path = os.path.join(output_dir, f"{prefix}_target_roi_crop.png")

        plt.imsave(source_mask_path, self.source_tissue_mask.astype(np.uint8) * 255, cmap=cmap)
        plt.imsave(target_mask_path, self.target_tissue_mask.astype(np.uint8) * 255, cmap=cmap)
        plt.imsave(source_masked_path, source_roi)
        plt.imsave(target_masked_path, target_roi)
        plt.imsave(source_roi_crop_path, source_roi_crop)
        plt.imsave(target_roi_crop_path, target_roi_crop)

        return {
            "source_mask": source_mask_path,
            "target_mask": target_mask_path,
            "source_masked_full": source_masked_path,
            "target_masked_full": target_masked_path,
            "source_roi_crop": source_roi_crop_path,
            "target_roi_crop": target_roi_crop_path,
        }

    @staticmethod
    def get_concentrations(
        img: np.ndarray,
        stain_matrix: np.ndarray,
        clip_non_negative: bool = True,
    ) -> np.ndarray:
        """Estimate stain concentrations in OD space given a stain basis."""
        od = rgb2od(img).reshape((-1, 3))
        concentrations, _, _, _ = np.linalg.lstsq(stain_matrix.T, od.T, rcond=-1)
        concentrations = concentrations.T
        if clip_non_negative:
            concentrations = np.clip(concentrations, 0, None)
        return concentrations

    @staticmethod
    def _compute_channel_scale_factors(
        max_c_target: np.ndarray,
        max_c_source: np.ndarray,
        max_scale_factor: float | None,
    ) -> np.ndarray:
        """Compute robust per-channel concentration scaling factors.

        Unbounded percentile ratios can explode when a stain channel is weak or
        nearly absent in the source but strong in the target. That drives the OD
        reconstruction to saturation and yields near-black tissue.
        """
        scale_factors = max_c_target / np.maximum(max_c_source, 1e-8)
        if max_scale_factor is not None:
            if max_scale_factor <= 0:
                raise ValueError("max_scale_factor must be > 0 when provided.")
            scale_factors = np.minimum(scale_factors, max_scale_factor)
        return scale_factors

    def fit(self, target: np.ndarray) -> None:
        """Fit target stain basis and target concentration scaling statistics."""
        self.stain_matrix_target = self.extractor.get_stain_matrix(target)
        self.target_tissue_mask = self.extractor.last_tissue_mask
        target_concentrations = self.get_concentrations(target, self.stain_matrix_target)
        target_od = rgb2od(target).reshape((-1, 3))
        target_mask_flat = self.target_tissue_mask.reshape(-1) if self.target_tissue_mask is not None else None
        if target_mask_flat is not None and np.any(target_mask_flat):
            target_concentrations_for_scale = target_concentrations[target_mask_flat]
            target_od_for_cap = target_od[target_mask_flat]
        else:
            target_concentrations_for_scale = target_concentrations
            target_od_for_cap = target_od
        self.max_c_target = np.percentile(target_concentrations_for_scale, 99, axis=0, keepdims=True)
        self.max_od_target = np.percentile(target_od_for_cap, 99, axis=0, keepdims=True)

    def fit_multi_target(
        self,
        targets: Sequence[np.ndarray],
        *,
        aggregation: str = "median",
        max_workers: int | None = None,
        anchor_index: int | None = None,
    ) -> None:
        """Fit a multi-target reference state from several target images.

        This implements an Avg-post style workflow, but uses robust aggregation
        (``median`` by default) after explicit stain-row alignment.

        Args:
            targets (Sequence[numpy.ndarray]):
                Reference RGB images.
            aggregation (str):
                Aggregation method across aligned reference matrices. Supported:
                ``"median"`` and ``"mean"``.
            max_workers (int | None):
                Number of worker processes used to extract per-target stain
                matrices. ``None`` uses all available cores. When processing
                targets in parallel, extractor-side ``dl_n_jobs`` is forced to 1
                per worker to avoid nested oversubscription.
            anchor_index (int | None):
                Optional explicit anchor matrix index for alignment. ``None``
                selects a medoid-like anchor automatically.

        """
        if len(targets) == 0:
            raise ValueError("targets must contain at least one image.")
        if len(targets) == 1:
            self.fit(targets[0])
            return

        target_list = [np.asarray(target, dtype=np.uint8) for target in targets]

        if isinstance(self.extractor, VahadaneTrichromeExtractor):
            extractor_params = self.extractor.get_params()
        else:
            extractor_params = None

        use_parallel = extractor_params is not None and (max_workers is None or max_workers != 1)
        if use_parallel:
            worker_params = dict(extractor_params)
            if worker_params.get("backend") == "dictionary_learning":
                worker_params["dl_n_jobs"] = 1
            resolved_max_workers = max_workers or os.cpu_count() or 1
            with concurrent.futures.ProcessPoolExecutor(max_workers=resolved_max_workers) as executor:
                reference_states = list(
                    executor.map(
                        _extract_single_reference_state,
                        target_list,
                        [worker_params] * len(target_list),
                    )
                )
        else:
            reference_states = []
            for target in target_list:
                stain_matrix = self.extractor.get_stain_matrix(target)
                target_tissue_mask = self.extractor.last_tissue_mask
                target_concentrations = self.get_concentrations(target, stain_matrix)
                target_mask_flat = target_tissue_mask.reshape(-1) if target_tissue_mask is not None else None
                if target_mask_flat is not None and np.any(target_mask_flat):
                    target_concentrations_for_scale = target_concentrations[target_mask_flat]
                else:
                    target_concentrations_for_scale = target_concentrations
                max_c_target = np.percentile(target_concentrations_for_scale, 99, axis=0, keepdims=True)
                target_od = rgb2od(target).reshape((-1, 3))
                if target_mask_flat is not None and np.any(target_mask_flat):
                    target_od_for_cap = target_od[target_mask_flat]
                else:
                    target_od_for_cap = target_od
                max_od_target = np.percentile(target_od_for_cap, 99, axis=0, keepdims=True)
                reference_states.append((stain_matrix, max_c_target, max_od_target))

        stain_matrices = [state[0] for state in reference_states]
        scale_vectors = [state[1] for state in reference_states]
        od_caps = [state[2] for state in reference_states]

        aligned_stain_matrices, permutations, resolved_anchor_index = _align_stain_matrices_to_anchor(
            stain_matrices,
            anchor_index=anchor_index,
        )
        aligned_scale_vectors = [
            scale_vector[:, list(perm)]
            for scale_vector, perm in zip(scale_vectors, permutations)
        ]

        self.stain_matrix_target = _aggregate_stain_matrices(
            aligned_stain_matrices,
            method=aggregation,
        )
        scale_stack = np.stack(aligned_scale_vectors, axis=0)
        if aggregation == "median":
            self.max_c_target = np.median(scale_stack, axis=0)
            self.max_od_target = np.median(np.stack(od_caps, axis=0), axis=0)
        elif aggregation == "mean":
            self.max_c_target = np.mean(scale_stack, axis=0)
            self.max_od_target = np.mean(np.stack(od_caps, axis=0), axis=0)
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")

        self.target_tissue_mask = None
        self.fit_metadata = {
            "fit_mode": "multi_target",
            "n_targets": len(target_list),
            "aggregation": aggregation,
            "anchor_index": resolved_anchor_index,
        }

    def save_fit_state(self, file_path: str, metadata: dict | None = None) -> str:
        """Persist fitted target state for reuse across future transforms."""
        if self.stain_matrix_target is None or self.max_c_target is None:
            raise RuntimeError("Call fit(target) before save_fit_state(...).")
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError(f"metadata must be dict or None, got {type(metadata)}")

        metadata_dict = metadata or {}
        try:
            metadata_json = json.dumps(metadata_dict)
        except TypeError as exc:
            raise ValueError("metadata must be JSON-serializable.") from exc

        target_mask = self.target_tissue_mask
        has_target_mask = bool(target_mask is not None)
        target_mask_uint8 = (
            target_mask.astype(np.uint8)
            if target_mask is not None
            else np.zeros((0, 0), dtype=np.uint8)
        )

        np.savez_compressed(
            file_path,
            stain_matrix_target=self.stain_matrix_target.astype(np.float32),
            max_c_target=self.max_c_target.astype(np.float32),
            max_od_target=(
                self.max_od_target.astype(np.float32)
                if self.max_od_target is not None
                else np.zeros((1, 3), dtype=np.float32)
            ),
            target_tissue_mask=target_mask_uint8,
            has_target_tissue_mask=np.array([has_target_mask], dtype=np.uint8),
            metadata_json=np.array(metadata_json),
            has_max_od_target=np.array([self.max_od_target is not None], dtype=np.uint8),
            state_version=np.array([2], dtype=np.int32),
        )
        return file_path

    def load_fit_state(self, file_path: str) -> dict:
        """Load a previously saved fit state into this normalizer instance."""
        with np.load(file_path, allow_pickle=False) as payload:
            self.stain_matrix_target = payload["stain_matrix_target"].astype(np.float64)
            self.max_c_target = payload["max_c_target"].astype(np.float64)
            has_max_od_target = bool(payload["has_max_od_target"][0]) if "has_max_od_target" in payload else False
            if has_max_od_target:
                self.max_od_target = payload["max_od_target"].astype(np.float64)
            else:
                self.max_od_target = None

            has_target_mask = bool(payload["has_target_tissue_mask"][0])
            if has_target_mask:
                self.target_tissue_mask = payload["target_tissue_mask"].astype(bool)
            else:
                self.target_tissue_mask = None

            metadata_json = str(payload["metadata_json"]) if "metadata_json" in payload else "{}"
            metadata = json.loads(metadata_json)

        self.fit_metadata = metadata
        self.stain_matrix_source_raw = None
        self.stain_matrix_source_aligned = None
        self.source_tissue_mask = None
        return metadata

    def transform(self, img: np.ndarray, apply_source_tissue_mask: bool = False) -> np.ndarray:
        """Normalize source image to fitted target stain appearance."""
        if self.stain_matrix_target is None or self.max_c_target is None:
            msg = "Call fit(target) before transform(img)."
            raise RuntimeError(msg)

        stain_matrix_source = self.extractor.get_stain_matrix(img)
        self.stain_matrix_source_raw = stain_matrix_source.copy()
        self.source_tissue_mask = self.extractor.last_tissue_mask
        stain_matrix_source = _match_source_rows_to_target(
            stain_matrix_source,
            self.stain_matrix_target,
        )
        self.stain_matrix_source_aligned = stain_matrix_source.copy()

        source_concentrations = self.get_concentrations(img, stain_matrix_source)
        source_mask_flat = self.source_tissue_mask.reshape(-1) if self.source_tissue_mask is not None else None
        if source_mask_flat is not None and np.any(source_mask_flat):
            source_concentrations_for_scale = source_concentrations[source_mask_flat]
        else:
            source_concentrations_for_scale = source_concentrations
        max_c_source = np.percentile(source_concentrations_for_scale, 99, axis=0, keepdims=True)
        scale_factors = self._compute_channel_scale_factors(
            self.max_c_target,
            max_c_source,
            self.max_concentration_scale_factor,
        )
        source_concentrations *= scale_factors

        reconstructed_od = np.dot(source_concentrations, self.stain_matrix_target)
        if self.max_od_target is not None:
            reconstructed_od = np.minimum(reconstructed_od, self.max_od_target)

        transformed = 255 * np.exp(-1 * reconstructed_od)
        transformed = np.clip(transformed, 0, 255).reshape(img.shape).astype(np.uint8)

        if apply_source_tissue_mask:
            if self.source_tissue_mask is None:
                msg = "Source tissue mask is not available after transform extraction."
                raise RuntimeError(msg)
            return self._apply_tissue_mask(transformed, self.source_tissue_mask)

        return transformed


__all__ = [
    "VahadaneTrichromeExtractor",
    "VahadaneTrichromeNormalizer",
]
