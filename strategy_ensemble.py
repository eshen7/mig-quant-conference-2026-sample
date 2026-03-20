"""
Ensemble Averaging Strategy
============================

Reduces overfitting by running the full etf_arb_tuned strategy with multiple
train_frac values and averaging the resulting actions across all models.

Train fractions used: [0.60, 0.70, 0.80, 0.85, 0.90]
"""

import numpy as np
from etf_arb_tuned import get_actions as _etf_get_actions


TRAIN_FRACS = [0.60, 0.70, 0.80, 0.85, 0.90]


def get_actions(prices: np.ndarray) -> np.ndarray:
    """
    Ensemble averaging over multiple train_frac values.

    For each train_frac, runs the full etf_arb_tuned strategy, then averages
    all action matrices element-wise. The result is rounded and clipped.
    """
    num_stocks, num_days = prices.shape
    accumulated = np.zeros((num_stocks, num_days), dtype=np.float64)

    for tf in TRAIN_FRACS:
        actions = _etf_get_actions(prices, train_frac=tf)
        accumulated += actions

    # Average across all models
    averaged = accumulated / len(TRAIN_FRACS)

    # Round to integers and clip
    averaged = np.round(averaged).astype(np.float64)
    averaged = np.clip(averaged, -100, 100)

    return averaged
