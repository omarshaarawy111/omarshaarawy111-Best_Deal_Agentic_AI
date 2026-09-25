"""
Price ensemble modules for Best Deal.

Description:
    This package separates ensemble mathematics from individual model agents.

Responsibilities:
    - Combine specialist model estimates.
    - Compute business-facing deal metrics.
"""

from best_deal.ml.ensemble.ensemble import WeightedPriceEnsemble
from best_deal.ml.ensemble.business_metrics import DealEvaluator, calculate_business_metrics

__all__ = ["WeightedPriceEnsemble", "DealEvaluator", "calculate_business_metrics"]
