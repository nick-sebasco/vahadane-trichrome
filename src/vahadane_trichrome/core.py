"""Compatibility layer for the Vahadane implementation."""

from .methods.vahadane import NMF
from .methods.vahadane import VahadaneTrichromeExtractor
from .methods.vahadane import VahadaneTrichromeNormalizer
from .methods.vahadane import _align_stain_matrices_to_anchor
from .methods.vahadane import _aggregate_stain_matrices
from .methods.vahadane import _extract_single_reference_state
from .methods.vahadane import _get_best_alignment_permutation
from .methods.vahadane import _match_source_rows_to_target
from .methods.vahadane import _normalize_extractor_backend_name
from .methods.vahadane import _resolve_backend_regularizer
from .methods.vahadane import _select_alignment_anchor_index
from .methods.vahadane import _sort_dictionary_by_dominant_channel
from .methods.vahadane import _stain_matrix_alignment_score
from .methods.vahadane import _unit_row_normalize
from .methods.vahadane import _validate_nmf_configuration

__all__ = [
    "VahadaneTrichromeExtractor",
    "VahadaneTrichromeNormalizer",
    "_align_stain_matrices_to_anchor",
    "_aggregate_stain_matrices",
    "_extract_single_reference_state",
    "_get_best_alignment_permutation",
    "_match_source_rows_to_target",
    "NMF",
    "_normalize_extractor_backend_name",
    "_resolve_backend_regularizer",
    "_select_alignment_anchor_index",
    "_sort_dictionary_by_dominant_channel",
    "_stain_matrix_alignment_score",
    "_unit_row_normalize",
    "_validate_nmf_configuration",
]
