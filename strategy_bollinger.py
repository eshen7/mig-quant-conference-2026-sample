"""
MIG Quant Competition — Bollinger Band Mean Reversion Strategy
===============================================================

For each stock, compute a rolling mean and standard deviation over a lookback
window.  When the z-score (deviation from the mean in std units) exceeds the
entry threshold the strategy opens a position betting on reversion to the mean.
Positions are closed when the z-score crosses back through the exit threshold.
A stop-loss fires when the z-score diverges even further (trend continuation).

Position sizes are volatility-scaled so that higher-vol stocks get fewer shares.
"""

import numpy as np

# --- Tuned parameters ---
LOOKBACK = 20          # rolling window for mean / std
ENTRY_Z = 2.0          # open position when |z| > this
EXIT_Z = 0.3           # close position when |z| < this
STOP_Z = 3.5           # stop-loss when |z| > this (trend continuation)
BASE_SHARES = 6        # base position size (shares)
POS_LIMIT = 100        # hard cap per stock
MIN_STD_PCT = 0.005    # skip stocks whose rolling std / price < this


def get_actions(prices: np.ndarray) -> np.ndarray:
    num_stocks, num_days = prices.shape
    actions = np.zeros((num_stocks, num_days))

    # Track current position per stock: +1 long, -1 short, 0 flat
    position = np.zeros(num_stocks, dtype=int)   # direction
    pos_size = np.zeros(num_stocks, dtype=int)    # absolute shares held

    for day in range(LOOKBACK, num_days):
        window = prices[:, day - LOOKBACK : day]  # (num_stocks, LOOKBACK)
        rolling_mean = window.mean(axis=1)
        rolling_std = window.std(axis=1)

        current_price = prices[:, day]

        # Avoid division by zero and skip illiquid / flat stocks
        valid = (rolling_std / (current_price + 1e-10)) > MIN_STD_PCT
        z = np.where(valid, (current_price - rolling_mean) / (rolling_std + 1e-10), 0.0)

        # Volatility-scaled position sizing: fewer shares for high-vol stocks
        # Normalise std across stocks; inverse-vol weighting
        median_std = np.median(rolling_std[valid]) if valid.any() else 1.0
        vol_scale = np.where(
            rolling_std > 0,
            np.clip(median_std / rolling_std, 0.3, 2.0),
            1.0,
        )
        sized = np.round(BASE_SHARES * vol_scale).astype(int)
        sized = np.clip(sized, 1, POS_LIMIT)

        for i in range(num_stocks):
            if not valid[i]:
                # Close any open position on invalid stock
                if position[i] != 0:
                    actions[i, day] = -position[i] * pos_size[i]
                    position[i] = 0
                    pos_size[i] = 0
                continue

            zi = z[i]

            # --- Already in a position ---
            if position[i] != 0:
                # Stop-loss: z moved further against us
                if (position[i] == 1 and zi < -STOP_Z) or \
                   (position[i] == -1 and zi > STOP_Z):
                    actions[i, day] = -position[i] * pos_size[i]
                    position[i] = 0
                    pos_size[i] = 0
                # Mean reversion complete — exit
                elif abs(zi) < EXIT_Z:
                    actions[i, day] = -position[i] * pos_size[i]
                    position[i] = 0
                    pos_size[i] = 0
                # else: hold
                continue

            # --- Flat: check for new entry ---
            if zi > ENTRY_Z:
                # Overbought → short
                shares = sized[i]
                actions[i, day] = -shares
                position[i] = -1
                pos_size[i] = shares
            elif zi < -ENTRY_Z:
                # Oversold → long
                shares = sized[i]
                actions[i, day] = shares
                position[i] = 1
                pos_size[i] = shares

    return actions
