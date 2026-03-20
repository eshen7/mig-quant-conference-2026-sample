"""
MIG Quant Competition -- Short-Term Cross-Sectional Mean Reversion
==================================================================

Each day, compute short-term returns for every stock, rank them
cross-sectionally, go long the biggest losers and short the biggest
winners. Positions are held for a fixed number of days and then closed.
Dollar-neutral construction ensures roughly equal long and short
exposure. A dispersion filter avoids trading in low-volatility regimes
where mean-reversion signals are unreliable.
"""

import numpy as np

# --- Tunable parameters ---
LOOKBACK = 3          # days of return to measure (1, 2, 3, or 5)
HOLD_PERIOD = 2       # days to hold before closing (1, 2, or 3)
NUM_LONG = 5          # number of stocks to go long (biggest losers)
NUM_SHORT = 5         # number of stocks to short (biggest winners)
MAX_SHARES = 6        # max shares per stock per signal
DISPERSION_THRESH = 0.005  # min cross-sectional std of returns to trade
WARMUP = 20           # days before trading starts (need stable stats)


def get_actions(prices: np.ndarray) -> np.ndarray:
    num_stocks, num_days = prices.shape
    actions = np.zeros((num_stocks, num_days), dtype=np.float64)

    # Track open positions: list of (day_opened, stock_idx, shares) tuples
    open_positions = []

    for day in range(WARMUP, num_days):
        # --- Close positions that have been held long enough ---
        still_open = []
        for (open_day, stock_idx, shares) in open_positions:
            if day - open_day >= HOLD_PERIOD:
                # Close: reverse the position
                actions[stock_idx, day] -= shares
            else:
                still_open.append((open_day, stock_idx, shares))
        open_positions = still_open

        # --- Compute short-term returns ---
        if day < LOOKBACK:
            continue

        returns = np.zeros(num_stocks)
        for s in range(num_stocks):
            p_now = prices[s, day]
            p_prev = prices[s, day - LOOKBACK]
            if p_prev > 0:
                returns[s] = (p_now - p_prev) / p_prev

        # --- Dispersion filter ---
        cross_std = np.std(returns)
        if cross_std < DISPERSION_THRESH:
            continue

        # --- Cross-sectional ranking ---
        # Rank stocks by return: lowest returns = biggest losers (buy),
        # highest returns = biggest winners (short)
        ranked = np.argsort(returns)  # ascending: losers first
        long_stocks = ranked[:NUM_LONG]
        short_stocks = ranked[-NUM_SHORT:]

        # --- Dollar-neutral position sizing ---
        # Compute shares for each leg so dollar exposure is roughly equal
        long_prices = prices[long_stocks, day]
        short_prices = prices[short_stocks, day]

        # Target equal dollar per stock within each leg
        # Use a fixed small number of shares, capped by MAX_SHARES
        long_shares = np.zeros(NUM_LONG, dtype=int)
        short_shares = np.zeros(NUM_SHORT, dtype=int)

        # Compute target dollar amount: use the median price * MAX_SHARES / 2
        # to keep positions small
        for i in range(NUM_LONG):
            if long_prices[i] > 0:
                long_shares[i] = min(MAX_SHARES, max(1, int(MAX_SHARES)))

        for i in range(NUM_SHORT):
            if short_prices[i] > 0:
                short_shares[i] = min(MAX_SHARES, max(1, int(MAX_SHARES)))

        # Adjust for dollar neutrality: scale the leg with more dollar exposure down
        long_dollar = np.sum(long_shares * long_prices)
        short_dollar = np.sum(short_shares * short_prices)

        if long_dollar > 0 and short_dollar > 0:
            ratio = short_dollar / long_dollar
            if ratio > 1.2:
                # Short leg is bigger, scale short shares down
                scale = long_dollar / short_dollar
                short_shares = np.maximum(1, (short_shares * scale).astype(int))
            elif ratio < 0.8:
                # Long leg is bigger, scale long shares down
                scale = short_dollar / long_dollar
                long_shares = np.maximum(1, (long_shares * scale).astype(int))

        # --- Apply actions ---
        for i, s in enumerate(long_stocks):
            sh = int(long_shares[i])
            if sh > 0:
                actions[s, day] += sh
                open_positions.append((day, s, sh))

        for i, s in enumerate(short_stocks):
            sh = int(short_shares[i])
            if sh > 0:
                actions[s, day] -= sh
                open_positions.append((day, s, -sh))

    # --- Close all remaining positions on the last day ---
    # (Already handled by the loop if hold period is reached,
    #  but force-close anything still open)
    # We don't add closing actions beyond the last day since
    # that would be out of bounds. The backtester handles end-of-data.

    return actions
