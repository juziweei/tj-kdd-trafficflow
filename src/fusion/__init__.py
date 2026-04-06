"""Fusion utilities for combining multiple forecasting trunks."""

from .adaptive_weight import AdaptiveFusionWeights, fit_adaptive_fusion_weights

__all__ = ["AdaptiveFusionWeights", "fit_adaptive_fusion_weights"]
"""Fusion algorithms for combining model branches."""
