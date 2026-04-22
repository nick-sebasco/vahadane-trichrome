"""Public package API for vahadane_trichrome."""

from .evaluation import CohortDistributionPlotResult
from .evaluation import CohortDistanceResult
from .evaluation import ReferenceCohortImprovement
from .evaluation import StructuralSimilarityResult
from .evaluation import cohort_wasserstein_matrix
from .evaluation import load_rgb_uint8
from .evaluation import paired_structural_similarity
from .evaluation import plot_cohort_feature_distributions
from .evaluation import sample_image_features
from .evaluation import structural_similarity_score
from .evaluation import summarize_reference_cohort_improvement
from .evaluation import wasserstein_distance_1d
from .histogram_matching import HistogramMatchingNormalizer
from .histogram_matching import apply_histogram_lut
from .histogram_matching import build_cohort_histogram_specification_lut
from .histogram_matching import build_histogram_specification_lut
from .histogram_matching import histogram_specification
from .histogram_matching import match_channel_histogram
from .methods.vahadane import VahadaneTrichromeExtractor
from .methods.vahadane import VahadaneTrichromeNormalizer
from .utils import get_tissue_mask
from .utils import get_luminosity_tissue_mask
from .utils import rgb2od

__all__ = [
    "VahadaneTrichromeExtractor",
    "VahadaneTrichromeNormalizer",
    "HistogramMatchingNormalizer",
    "CohortDistributionPlotResult",
    "CohortDistanceResult",
    "StructuralSimilarityResult",
    "ReferenceCohortImprovement",
    "apply_histogram_lut",
    "build_cohort_histogram_specification_lut",
    "build_histogram_specification_lut",
    "match_channel_histogram",
    "histogram_specification",
    "load_rgb_uint8",
    "sample_image_features",
    "wasserstein_distance_1d",
    "cohort_wasserstein_matrix",
    "plot_cohort_feature_distributions",
    "structural_similarity_score",
    "paired_structural_similarity",
    "summarize_reference_cohort_improvement",
    "rgb2od",
    "get_luminosity_tissue_mask",
    "get_tissue_mask",
]
