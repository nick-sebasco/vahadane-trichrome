"""
This module implements the Vahadane stain normalization method for Mason's Trichrome stained images.  
The Vahadane method is a stain normalization technique that uses non-negative matrix factorization (NMF)
to decompose the image into stain concentration and stain color matrices. This allows for the normalization
of staining variations across different images, making it easier to compare and analyze histopathological images.

The default implmentations for almost every stain normalization method I have looked at including Vahdane
is that the algorithm only supports 2 stains like H&E. However, Mason's Trichrome has 3 stains. So I have 
implemented a custom version of the Vahadane method.

Looking at the TiaToolbox implementation we can see that they have a hard_coded n_components = 2. 

Commands:
Run:
singularity exec /projects/korstanje-lab/singularity-containers/vit_tiatoolbox.sif python

Usage examples:

1) Trichrome normalization (recommended default path)

        >>> extractor = VahadaneTrichromeExtractor(
        ...     luminosity_threshold=0.8,
        ...     regularizer=0.1,
        ...     n_components=3,
        ...     sort_mode="none",
        ... )
        >>> normalizer = VahadaneTrichromeNormalizer(extractor=extractor)
        >>> normalizer.fit(target_rgb_uint8)
        >>> normalized = normalizer.transform(source_rgb_uint8)

2) Short form with bubbled extractor parameters

        >>> normalizer = VahadaneTrichromeNormalizer(
        ...     n_components=3,
        ...     luminosity_threshold=0.8,
        ...     regularizer=0.1,
        ...     sort_mode="none",
        ... )
        >>> normalizer.fit(target_rgb_uint8)
        >>> normalized = normalizer.transform(source_rgb_uint8)

Notes:
- ``sort_mode='none'`` is the default because semantic alignment is handled by
    target-aware source/target row matching during ``transform``.
- ``sort_mode='dominant_channel'`` can improve reproducibility of extracted row
    order for debugging/inspection, but is not used as semantic truth.
"""

import os
import itertools
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import DictionaryLearning

logger = logging.getLogger(__name__)


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


def dl_output_for_h_and_e(dictionary: np.ndarray) -> np.ndarray:
    """Sort a 2-stain dictionary by blue-channel intensity (descending).

    This mimics the common H&E convention where hematoxylin tends to have higher
    blue-channel OD response than eosin.

    Args:
        dictionary (:class:`numpy.ndarray`):
            Stain matrix of shape ``(2, 3)``.

    Returns:
        :class:`numpy.ndarray`:
            Reordered matrix of shape ``(2, 3)``.

    """
    matrix = np.asarray(dictionary)
    if matrix.shape != (2, 3):
        return matrix
    order = np.argsort(-matrix[:, 2])
    return matrix[order]


def _sort_dictionary_by_dominant_channel(dictionary: np.ndarray) -> np.ndarray:
    """Return a deterministic, channel-based ordering of stain vectors.

    This helper is a *stability/convenience* ordering only. It does **not** claim
    semantic stain identity (for example, collagen vs cytoplasm vs nuclei).

    Approach:
    1) For each stain vector row, find its dominant channel via ``argmax(R, G, B)``.
    2) Sort rows by dominant channel index (R=0, G=1, B=2).
    3) Break ties by dominant magnitude (stronger first).

    Why this exists:
    - Dictionary learning is permutation-invariant, so row order can change between
      runs even when vectors are otherwise similar.
    - A deterministic order helps debugging, logging, and visual comparisons.

    Important limitation:
    - For trichrome (or any >2-stain setting), channel dominance is not a reliable
      universal mapping to biological stain names. Therefore this function should be
      considered optional and is not sufficient by itself for source/target channel
      alignment during normalization.

    Args:
        dictionary (:class:`numpy.ndarray`):
            Stain matrix of shape ``(n_components, 3)`` in OD space.

    Returns:
        :class:`numpy.ndarray`:
            Reordered stain matrix with deterministic row order.

    """
    dominant_channel = np.argmax(dictionary, axis=1)
    dominant_value = np.max(dictionary, axis=1)
    order = np.lexsort((-dominant_value, dominant_channel))
    return dictionary[order]


def _match_source_rows_to_target(
    source_dictionary: np.ndarray,
    target_dictionary: np.ndarray,
) -> np.ndarray:
    """Align source stain rows to target stain rows by best global similarity.

    Why this is needed:
    - Dictionary learning returns stain vectors up to permutation.
    - In normalization, concentration channel ``i`` from source must correspond to
      channel ``i`` in the target basis before percentile scaling and reconstruction.
    - If rows are not aligned, channels are mixed and colors become inconsistent.

    Why this is more valid than H&E blue-channel sorting for trichrome:
    - In the Vahadane H&E setup, ordering by blue intensity is used because
      hematoxylin/eosin have a known two-stain spectral relationship.
    - Trichrome has three stains and no single monotonic blue-channel rule that
      robustly separates all components across scanners/tissues.
    - Therefore we use data-driven alignment to the *fitted target basis* instead of
      imposing an H&E-specific heuristic.

    Method:
    1) Normalize rows of source and target matrices to unit length.
    2) Compute cosine-similarity matrix ``S = source_norm @ target_norm.T``.
    3) Evaluate all row permutations and select the one maximizing
       ``sum(S[perm[i], i])``.

    For ``n_components=3`` this is only 6 permutations, so exhaustive search is
    exact and inexpensive.

    Args:
        source_dictionary (:class:`numpy.ndarray`):
            Source stain matrix of shape ``(n_components, 3)``.
        target_dictionary (:class:`numpy.ndarray`):
            Target stain matrix of shape ``(n_components, 3)``.

    Returns:
        :class:`numpy.ndarray`:
            Source stain matrix with rows permuted to best align with target rows.

    Raises:
        ValueError:
            If source and target shapes differ.

    """
    if source_dictionary.shape != target_dictionary.shape:
        msg = (
            "Source and target stain matrices must have the same shape, got "
            f"{source_dictionary.shape} and {target_dictionary.shape}."
        )
        raise ValueError(msg)

    source_norm = source_dictionary / np.linalg.norm(source_dictionary, axis=1, keepdims=True)
    target_norm = target_dictionary / np.linalg.norm(target_dictionary, axis=1, keepdims=True)
    similarity = source_norm @ target_norm.T

    n_components = source_dictionary.shape[0]
    best_perm = tuple(range(n_components))
    best_score = -np.inf
    for perm in itertools.permutations(range(n_components)):
        score = similarity[list(perm), range(n_components)].sum()
        if score > best_score:
            best_score = score
            best_perm = perm

    return source_dictionary[list(best_perm), :]



class VahadaneTrichromeExtractor:
    """Vahadane-style stain extractor generalized to arbitrary stain count.

    Get the stain matrix as defined in:

    Vahadane, Abhishek, et al. "Structure-preserving color normalization
    and sparse stain separation for histological images."
    IEEE transactions on medical imaging 35.8 (2016): 1962-1971.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.

    .. warning::
        Vahadane stain extraction/normalization algorithms are unstable
        after the update to `dictionary learning` algorithm in
        scikit-learn > v0.23.0 (see issue #382). Please be advised and
        consider using other stain extraction (normalization) algorithms
        or toolboxes, such as https://github.com/CielAl/torch-staintools

        Design notes:
        - TIAToolbox's original :class:`VahadaneExtractor` is hard-coded with
            ``n_components=2`` and applies an H&E-specific ordering helper
            (:func:`dl_output_for_h_and_e`).
        - In the Vahadane paper's H&E context, sorting by blue-channel intensity is
            sensible because hematoxylin/eosin have known spectral behavior.
        - For trichrome, that assumption is not generally valid; there are three stains
            and no universal blue-ranking rule that cleanly identifies all channels.
        - This extractor therefore exposes ``n_components`` and keeps semantic channel
            alignment to a separate target-aware matching step in
            :func:`_match_source_rows_to_target`.

        Args:
                luminosity_threshold (float):
                        Threshold used for tissue area selection.
                regularizer (float):
                        Regularizer used in dictionary learning.
                n_components (int):
                        Number of stain vectors to learn from dictionary learning.
                        Use ``2`` for H&E-like behavior or ``3`` for trichrome.
                sort_mode (str):
                        Optional extractor-side ordering mode.
                        - ``"none"``: keep learned order.
                        - ``"dominant_channel"``: deterministic channel-based sort
                            (useful for reproducibility, not semantic labeling).

    Examples:
        >>> extractor = VahadaneTrichromeExtractor()
        >>> img = imread('path/to/image')
        >>> stain_matrix = extractor.get_stain_matrix(img)

    """

    def __init__(
        self,
        luminosity_threshold: float = 0.8,
        regularizer: float = 0.1,
        n_components: int = 3,
        sort_mode: str = "none",
        max_tissue_pixels: int | None = None,
        random_state: int = 0,
        fit_algorithm: str = "lars",
        transform_algorithm: str = "lasso_lars",
        dl_max_iter: int = 100,
        dl_transform_max_iter: int = 1000,
    ) -> None:
        """Initialize :class:`VahadaneTrichromeExtractor`."""
        # Issue a warning about the algorithm's stability
        logger.warning(
            "Vahadane stain extraction/normalization algorithms are unstable "
            "after the update to `dictionary learning` algorithm in "
            "scikit-learn > v0.23.0 (see issue #382). Please be advised and "
            "consider using other stain extraction (normalization) algorithms.",
            stacklevel=2,
        )
        self.__luminosity_threshold = luminosity_threshold
        self.__regularizer = regularizer
        self.__n_components = n_components
        self.__sort_mode = sort_mode
        self.__max_tissue_pixels = max_tissue_pixels
        self.__random_state = random_state
        self.__fit_algorithm = fit_algorithm
        self.__transform_algorithm = transform_algorithm
        self.__dl_max_iter = dl_max_iter
        self.__dl_transform_max_iter = dl_transform_max_iter
        self._last_tissue_mask: np.ndarray | None = None

    @property
    def last_tissue_mask(self) -> np.ndarray | None:
        """Return the most recent 2D tissue mask used during extraction."""
        if self._last_tissue_mask is None:
            return None
        return self._last_tissue_mask.copy()

    def get_stain_matrix(self, img: np.ndarray) -> np.ndarray:
        """Stain matrix estimation.

        Args:
            img (:class:`numpy.ndarray`):
                Input image used for stain matrix estimation.

        Returns:
            :class:`numpy.ndarray`:
                Estimated stain matrix.

        """
        img = img.astype("uint8")  # ensure input image is uint8
        luminosity_threshold = self.__luminosity_threshold
        regularizer = self.__regularizer
        n_components = self.__n_components
        # convert to OD and ignore background
        tissue_mask_2d = get_luminosity_tissue_mask(
            img,
            threshold=luminosity_threshold,
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

        # do the dictionary learning
        dl = DictionaryLearning(
            n_components=n_components,
            alpha=regularizer,
            transform_alpha=regularizer,
            fit_algorithm=self.__fit_algorithm,
            transform_algorithm=self.__transform_algorithm,
            positive_dict=True,
            verbose=False,
            max_iter=self.__dl_max_iter,
            transform_max_iter=self.__dl_transform_max_iter,
            random_state=self.__random_state,
        )
        dl.fit(X=img_od)
        dictionary = dl.components_

        # Keep TIAToolbox's H&E ordering behavior only for two-component extraction.
        if n_components == 2:
            dictionary = dl_output_for_h_and_e(dictionary)
        elif self.__sort_mode == "dominant_channel":
            dictionary = _sort_dictionary_by_dominant_channel(dictionary)

        return dictionary / np.maximum(np.linalg.norm(dictionary, axis=1, keepdims=True), 1e-8)


class VahadaneTrichromeNormalizer:
    """Generic Vahadane-style normalizer supporting arbitrary number of stains.

    Notes:
    - This implementation follows the widely used TIAToolbox-style pipeline:
      stain basis estimation with dictionary learning, then concentration recovery
      with least-squares (plus non-negativity clipping in this module).
    - The original Vahadane paper formulates a joint sparse non-negative matrix
      factorization (SNMF) problem where non-negativity is enforced as an explicit
      optimization constraint. Therefore this implementation should be interpreted
      as a practical approximation of that ideal constrained formulation.
    """

    def __init__(
        self,
        extractor: VahadaneTrichromeExtractor | None = None,
        *,
        luminosity_threshold: float = 0.8,
        regularizer: float = 0.1,
        n_components: int = 3,
        sort_mode: str = "none",
        max_tissue_pixels: int | None = None,
        random_state: int = 0,
        fit_algorithm: str = "lars",
        transform_algorithm: str = "lasso_lars",
        dl_max_iter: int = 100,
        dl_transform_max_iter: int = 1000,
    ) -> None:
        """Initialize normalizer and optionally bubble extractor parameters.

        Args:
            extractor (:class:`VahadaneTrichromeExtractor` | None):
                Optional preconfigured extractor. If provided, keyword parameters
                below are ignored.
            luminosity_threshold (float):
                Tissue threshold used when creating the default extractor.
            regularizer (float):
                Dictionary-learning regularization used by default extractor.
            n_components (int):
                Number of stains used by default extractor (typically 3 for
                trichrome in RGB).
            sort_mode (str):
                Extractor sort mode used by default extractor.
            max_tissue_pixels (int | None):
                Optional cap on tissue pixels used for dictionary learning.
                Reduces runtime/memory for large images. ``None`` uses all pixels.
            random_state (int):
                Random seed used for tissue pixel subsampling.
            fit_algorithm (str):
                Dictionary-learning fitting algorithm (e.g. ``"lars"`` or ``"cd"``).
            transform_algorithm (str):
                Sparse coding algorithm for transform step (e.g.
                ``"lasso_lars"`` or ``"lasso_cd"``).
            dl_max_iter (int):
                Maximum dictionary-learning iterations.
            dl_transform_max_iter (int):
                Maximum sparse-coding iterations.

        """
        self.extractor = extractor or VahadaneTrichromeExtractor(
            luminosity_threshold=luminosity_threshold,
            regularizer=regularizer,
            n_components=n_components,
            sort_mode=sort_mode,
            max_tissue_pixels=max_tissue_pixels,
            random_state=random_state,
            fit_algorithm=fit_algorithm,
            transform_algorithm=transform_algorithm,
            dl_max_iter=dl_max_iter,
            dl_transform_max_iter=dl_transform_max_iter,
        )
        self.stain_matrix_target: np.ndarray | None = None
        self.stain_matrix_source_raw: np.ndarray | None = None
        self.stain_matrix_source_aligned: np.ndarray | None = None
        self.max_c_target: np.ndarray | None = None
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
        """Convert a stain matrix into an RGB swatch strip image.

        Args:
            stain_matrix (:class:`numpy.ndarray`):
                Stain matrix of shape ``(n_components, 3)`` in OD space.
            swatch_height (int):
                Height of each swatch tile.
            swatch_width (int):
                Width of each swatch tile.
            rgb (bool):
                If ``False`` (default), use absorbance-channel visualization
                (debug-oriented OD display). If ``True``, render human-
                interpretable perceived color using Beer-Lambert forward model.
            rgb_strength (float):
                Optical-density multiplier used only when ``rgb=True`` to
                control swatch saturation/brightness.

        Returns:
            :class:`numpy.ndarray`:
                RGB image of shape ``(swatch_height, swatch_width * n_components, 3)``.

        """
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
    ) -> dict[str, str]:
        """Save stain vector swatch images for target and source (if available).

        Saves:
        - target swatches after ``fit``
        - source raw swatches after ``transform``
        - source aligned swatches after ``transform``

        Args:
            output_dir (str):
                Directory where swatch images are saved.
            prefix (str):
                Filename prefix.
            rgb (bool):
                If ``True``, save human-interpretable RGB swatches via
                Beer-Lambert rendering. If ``False`` (default), save OD-channel
                debug swatches.

        Returns:
            dict[str, str]:
                Mapping from swatch label to output file path.

        Raises:
            RuntimeError:
                If target stain matrix is not available (fit not called).

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

        if self.stain_matrix_source_raw is not None:
            source_raw_swatch = self._stain_matrix_to_swatch_image(self.stain_matrix_source_raw, rgb=rgb)
            source_raw_path = os.path.join(output_dir, f"{prefix}_{mode_tag}_source_raw_swatches.png")
            plt.imsave(source_raw_path, source_raw_swatch)
            outputs["source_raw_swatches"] = source_raw_path

        if self.stain_matrix_source_aligned is not None:
            source_aligned_swatch = self._stain_matrix_to_swatch_image(self.stain_matrix_source_aligned, rgb=rgb)
            source_aligned_path = os.path.join(output_dir, f"{prefix}_{mode_tag}_source_aligned_swatches.png")
            plt.imsave(source_aligned_path, source_aligned_swatch)
            outputs["source_aligned_swatches"] = source_aligned_path

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
        """Save source/target tissue masks, masked full images, and cropped ROIs.

        Args:
            source_img (:class:`numpy.ndarray`):
                Source RGB image used in ``transform``.
            target_img (:class:`numpy.ndarray`):
                Target RGB image used in ``fit``.
            output_dir (str):
                Directory where mask and ROI images are written.
            prefix (str):
                Prefix for saved filenames.

        Returns:
            dict[str, str]:
                Mapping of output labels to file paths.

        Raises:
            RuntimeError:
                If masks are not available (fit/transform not executed yet).

        """
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
        """Estimate stain concentrations in OD space given a stain basis.

        This solves a linear least-squares system per pixel:
        ``OD ~= C @ stain_matrix``.

        Args:
            img (:class:`numpy.ndarray`):
                RGB ``uint8`` image of shape ``(H, W, 3)``.
            stain_matrix (:class:`numpy.ndarray`):
                Stain basis matrix of shape ``(n_components, 3)`` in OD space.
            clip_non_negative (bool):
                If ``True``, clip concentrations to ``>= 0`` to enforce physical
                non-negativity of stain amounts and reduce whitening artifacts.

        Returns:
            :class:`numpy.ndarray`:
                Concentration matrix of shape ``(H*W, n_components)``.

        """
        od = rgb2od(img).reshape((-1, 3))
        concentrations, _, _, _ = np.linalg.lstsq(stain_matrix.T, od.T, rcond=-1)
        concentrations = concentrations.T
        if clip_non_negative:
            concentrations = np.clip(concentrations, 0, None)
        return concentrations

    def fit(self, target: np.ndarray) -> None:
        """Fit target stain basis and target concentration scaling statistics.

        Stores:
        - ``stain_matrix_target`` from extractor.
        - ``max_c_target`` as 99th percentile per target concentration channel,
          used later to scale source concentrations in ``transform``.

        Args:
            target (:class:`numpy.ndarray`):
                Target/reference RGB ``uint8`` image.

        """
        self.stain_matrix_target = self.extractor.get_stain_matrix(target)
        self.target_tissue_mask = self.extractor.last_tissue_mask
        target_concentrations = self.get_concentrations(target, self.stain_matrix_target)
        target_mask_flat = self.target_tissue_mask.reshape(-1) if self.target_tissue_mask is not None else None
        if target_mask_flat is not None and np.any(target_mask_flat):
            target_concentrations_for_scale = target_concentrations[target_mask_flat]
        else:
            target_concentrations_for_scale = target_concentrations
        self.max_c_target = np.percentile(target_concentrations_for_scale, 99, axis=0, keepdims=True)

    def save_fit_state(self, file_path: str, metadata: dict | None = None) -> str:
        """Persist fitted target state for reuse across future transforms.

        This saves the minimum required state to apply ``transform`` without
        re-running ``fit`` on the target image:
        - ``stain_matrix_target``
        - ``max_c_target``
        - ``target_tissue_mask`` (if available)

        Optional user metadata can be included (for example fit date, source
        image identifiers, cohort, operator notes, experiment tags).

        Args:
            file_path (str):
                Output path for the saved ``.npz`` file.
            metadata (dict | None):
                Optional JSON-serializable metadata.

        Returns:
            str:
                The saved file path.

        Raises:
            RuntimeError:
                If called before ``fit``.
            TypeError:
                If metadata is not a dict.
            ValueError:
                If metadata is not JSON-serializable.

        """
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
            target_tissue_mask=target_mask_uint8,
            has_target_tissue_mask=np.array([has_target_mask], dtype=np.uint8),
            metadata_json=np.array(metadata_json),
            state_version=np.array([1], dtype=np.int32),
        )
        return file_path

    def load_fit_state(self, file_path: str) -> dict:
        """Load a previously saved fit state into this normalizer instance.

        After loading, this normalizer can run ``transform`` directly.

        Args:
            file_path (str):
                Path to a state file created by ``save_fit_state``.

        Returns:
            dict:
                Metadata dictionary stored in the state file.

        """
        with np.load(file_path, allow_pickle=False) as payload:
            self.stain_matrix_target = payload["stain_matrix_target"].astype(np.float64)
            self.max_c_target = payload["max_c_target"].astype(np.float64)

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
        """Normalize source image to fitted target stain appearance.

        Pipeline:
        1) Extract source stain matrix.
        2) Align source rows to target rows using global best permutation.
        3) Solve source concentrations.
        4) Scale source concentration channels by ``max_c_target / max_c_source``.
        5) Reconstruct RGB image from scaled concentrations and target basis.
        6) Optionally keep non-tissue source pixels unchanged/white via mask.

        Args:
            img (:class:`numpy.ndarray`):
                Source RGB ``uint8`` image.
            apply_source_tissue_mask (bool):
                If ``True``, apply normalized values only on source tissue mask and
                set non-tissue pixels to white. If ``False`` (default), return
                normalized result for the full image.

        Returns:
            :class:`numpy.ndarray`:
                Normalized RGB ``uint8`` image with same shape as input.

        Raises:
            RuntimeError:
                If called before ``fit``.

        """
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
        source_concentrations *= self.max_c_target / np.maximum(max_c_source, 1e-8)

        transformed = 255 * np.exp(-1 * np.dot(source_concentrations, self.stain_matrix_target))
        transformed = np.clip(transformed, 0, 255).reshape(img.shape).astype(np.uint8)

        if apply_source_tissue_mask:
            if self.source_tissue_mask is None:
                msg = "Source tissue mask is not available after transform extraction."
                raise RuntimeError(msg)
            return self._apply_tissue_mask(transformed, self.source_tissue_mask)

        return transformed

