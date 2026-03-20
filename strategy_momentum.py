"""
MIG Quant Competition — Cross-Sectional Momentum Strategy
==========================================================

Ranks stocks by past returns over a formation period, goes long the top
quintile and short the bottom quintile.  Rebalances every `REBAL_FREQ`
trading days to reduce turnover.  A skip period avoids short-term reversal
contamination of the momentum signal.

Position sizes are kept small (max ~6 shares) so this strategy can be
combined with other alpha sources.
"""

import numpy as np

# --- Parameters (tuned on dev_data_30) ---
TRAIN_FRAC = 0.85          # first 85% used to calibrate (not directly needed here,
                           # but we skip those days to let lookback fill)
LOOKBACK = 60              # formation period in trading days
SKIP = 5                   # skip most recent N days to dodge reversal
REBAL_FREQ = 5             # rebalance every N trading days
LONG_K = 6                 # number of stocks to go long
SHORT_K = 6                # number of stocks to short
SHARES_PER_STOCK = 6       # position size per stock
POS_LIMIT = 100            # hard cap per stock


def get_actions(prices: np.ndarray) -> np.ndarray:
    num_stocks, num_days = prices.shape
    actions = np.zeros((num_stocks, num_days), dtype=np.float64)

    # We need at least LOOKBACK + SKIP days of history before we can trade
    start_day = LOOKBACK + SKIP

    # Current positions tracker (so we only emit *changes*)
    positions = np.zeros(num_stocks, dtype=np.float64)

    for day in range(start_day, num_days):
        # Only rebalance on schedule
        if (day - start_day) % REBAL_FREQ != 0:
            continue

        # --- Compute momentum signal: return from (day-LOOKBACK-SKIP) to (day-SKIP) ---
        past_price = prices[:, day - LOOKBACK - SKIP]
        recent_price = prices[:, day - SKIP]

        # Guard against zero / NaN prices
        valid = past_price > 0
        momentum = np.zeros(num_stocks)
        momentum[valid] = (recent_price[valid] - past_price[valid]) / past_price[valid]
        momentum[~valid] = -np.inf  # push invalids to bottom

        # --- Cross-sectional rank ---
        ranked = np.argsort(momentum)  # ascending: worst first
        short_stocks = ranked[:SHORT_K]
        long_stocks = ranked[-LONG_K:]

        # --- Build target portfolio ---
        target = np.zeros(num_stocks)
        target[long_stocks] = SHARES_PER_STOCK
        target[short_stocks] = -SHARES_PER_STOCK

        # --- Emit delta (target - current) ---
        delta = target - positions
        actions[:, day] = delta
        positions = target.copy()

    return actions
