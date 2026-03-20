"""
MIG Quant Competition — Time-Series Momentum / Trend Following Strategy
========================================================================

Uses dual moving average crossovers with volatility-adjusted position sizing.
  - Fast MA crosses above slow MA → long signal
  - Fast MA crosses below slow MA → short signal
  - Positions sized inversely to rolling volatility (ATR proxy via rolling std)
  - Whipsaw filter: only trade when MA spread exceeds a volatility-scaled threshold
"""

import numpy as np

# --- Parameters ---
FAST_PERIOD = 8          # fast moving average lookback
SLOW_PERIOD = 40         # slow moving average lookback
VOL_LOOKBACK = 20        # rolling std lookback for vol adjustment
THRESHOLD_MULT = 0.15    # MA spread must exceed this * vol to trigger signal
BASE_SHARES = 6          # base position size (shares) before vol scaling
MAX_SHARES = 8           # max shares per stock per signal
MIN_SHARES = 1           # min shares when signal is active
TARGET_VOL = 0.02        # target daily vol for position sizing


def _rolling_mean(arr, window):
    """Compute rolling mean along axis=1 for a 2D array."""
    num_stocks, num_days = arr.shape
    out = np.full_like(arr, np.nan)
    cumsum = np.cumsum(arr, axis=1)
    out[:, window - 1:] = cumsum[:, window - 1:] / window
    out[:, window:] -= cumsum[:, :-window] / window
    return out


def _rolling_std(arr, window):
    """Compute rolling std along axis=1 for a 2D array."""
    num_stocks, num_days = arr.shape
    out = np.full_like(arr, np.nan)
    for d in range(window - 1, num_days):
        out[:, d] = np.std(arr[:, d - window + 1:d + 1], axis=1)
    return out


def get_actions(prices: np.ndarray) -> np.ndarray:
    num_stocks, num_days = prices.shape
    actions = np.zeros((num_stocks, num_days))

    # Compute indicators
    fast_ma = _rolling_mean(prices, FAST_PERIOD)
    slow_ma = _rolling_mean(prices, SLOW_PERIOD)

    # Daily returns for vol estimation
    returns = np.zeros_like(prices)
    returns[:, 1:] = (prices[:, 1:] - prices[:, :-1]) / (prices[:, :-1] + 1e-10)
    vol = _rolling_std(returns, VOL_LOOKBACK)

    # MA spread
    spread = fast_ma - slow_ma

    # Track current position for each stock
    positions = np.zeros(num_stocks)

    start_day = max(SLOW_PERIOD, VOL_LOOKBACK) + 1

    for day in range(start_day, num_days):
        for s in range(num_stocks):
            if np.isnan(fast_ma[s, day]) or np.isnan(slow_ma[s, day]) or np.isnan(vol[s, day]):
                continue

            current_vol = vol[s, day]
            if current_vol < 1e-8:
                continue

            # Volatility-scaled threshold
            threshold = THRESHOLD_MULT * current_vol * prices[s, day]

            # Vol-adjusted position size: scale inversely with vol
            vol_scale = TARGET_VOL / (current_vol + 1e-10)
            target_shares = int(np.clip(BASE_SHARES * vol_scale, MIN_SHARES, MAX_SHARES))

            current_spread = spread[s, day]

            # Determine desired position
            if current_spread > threshold:
                desired = target_shares   # long
            elif current_spread < -threshold:
                desired = -target_shares  # short
            else:
                # In the dead zone: hold current position unless we want to flatten
                # Flatten only if spread crossed zero (trend reversal)
                prev_spread = spread[s, day - 1] if day > start_day else 0
                if positions[s] > 0 and current_spread < 0:
                    desired = 0  # exit long
                elif positions[s] < 0 and current_spread > 0:
                    desired = 0  # exit short
                else:
                    desired = positions[s]  # hold

            # Compute action needed to reach desired position
            action = desired - positions[s]
            if action != 0:
                actions[s, day] = action
                positions[s] = desired

    return actions
