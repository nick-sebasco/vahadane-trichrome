"""Public histogram-matching API."""

from .methods.histogram_matching import apply_histogram_lut
from .methods.histogram_matching import build_cohort_histogram_specification_lut
from .methods.histogram_matching import HistogramMatchingNormalizer
from .methods.histogram_matching import build_histogram_specification_lut
from .methods.histogram_matching import histogram_specification
from .methods.histogram_matching import match_channel_histogram

__all__ = [
    "apply_histogram_lut",
    "build_cohort_histogram_specification_lut",
    "HistogramMatchingNormalizer",
    "build_histogram_specification_lut",
    "match_channel_histogram",
    "histogram_specification",
]
