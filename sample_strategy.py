"""
MIG Quant Competition — Stat Arb Strategy
==========================================

Cointegration-based statistical arbitrage. Uses the first 70% of days as a
training window to fit OLS hedge ratios (each stock regressed on all others).
Stocks whose spread passes the ADF stationarity test (p < 0.05) are traded:
  - z-score > entry threshold  → short target, long top-3 basket constituents
  - z-score < −entry threshold → long target, short top-3 basket constituents
  - |z-score| < exit threshold → close the signal (both legs)

Parameters tuned on dev_data_30 with 0.1% fees and 100-share position limit.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller

# --- Tuned parameters ---
TRAIN_FRAC = 0.70       # fraction of days used for fitting (~pre-2020 on dev data)
ENTRY_Z = 1.75          # open signal when |z| exceeds this
EXIT_Z = 0.4            # close signal when |z| drops below this
DOLLAR_PER_SIGNAL = 450  # dollar exposure per signal on the target stock
MAX_SIGNALS = 10         # max concurrent open signals
POS_LIMIT = 100          # max shares per stock (long or short)
TOP_K_BASKET = 3         # number of basket constituents to trade per signal


def get_actions(prices: np.ndarray) -> np.ndarray:
    num_stocks, num_days = prices.shape
    split = int(num_days * TRAIN_FRAC)
    train = prices[:, :split]

    # --- Fit cointegration models on training window ---
    models = {}
    qualifying = []
    for i in range(num_stocks):
        y = train[i]
        X = np.delete(train, i, axis=0).T  # (split, num_stocks-1)
        reg = LinearRegression().fit(X, y)
        spread = y - reg.predict(X)
        adf_pval = adfuller(spread, maxlag=20, autolag='AIC')[1]
        if adf_pval >= 0.05:
            continue

        other_idxs = [j for j in range(num_stocks) if j != i]
        weight_pairs = sorted(
            zip(other_idxs, reg.coef_),
            key=lambda x: abs(x[1]), reverse=True,
        )
        models[i] = {
            'coef': reg.coef_,
            'intercept': reg.intercept_,
            'top_k': weight_pairs[:TOP_K_BASKET],
            'mean': spread.mean(),
            'std': spread.std(),
        }
        qualifying.append(i)

    # --- Generate actions ---
    actions = np.zeros((num_stocks, num_days))
    active_signals = {}  # target_idx -> {stock_idx: shares_held}

    for day in range(split, num_days):
        # Compute z-scores for qualifying stocks
        z_scores = {}
        for i in qualifying:
            info = models[i]
            other_prices = np.delete(prices[:, day], i)
            basket_val = info['coef'] @ other_prices + info['intercept']
            spread_val = prices[i, day] - basket_val
            z_scores[i] = (spread_val - info['mean']) / info['std']

        # Close signals that have reverted
        to_close = [t for t in active_signals if abs(z_scores.get(t, 0)) < EXIT_Z]
        for target_i in to_close:
            held = active_signals.pop(target_i)
            for s, shares in held.items():
                actions[s, day] -= shares  # reverse the position

        # Open new signals (strongest z-score first)
        if len(active_signals) < MAX_SIGNALS:
            candidates = sorted(
                [(abs(z_scores[i]), i, z_scores[i])
                 for i in qualifying if i not in active_signals],
                reverse=True,
            )
            for abs_z, i, z in candidates:
                if len(active_signals) >= MAX_SIGNALS or abs_z < ENTRY_Z:
                    break

                price_i = prices[i, day]
                target_shares = min(int(DOLLAR_PER_SIGNAL / price_i), POS_LIMIT)
                if target_shares == 0:
                    continue

                # Build the signal: target leg + basket legs
                held = {}
                direction = -1 if z > ENTRY_Z else 1  # short if overvalued
                held[i] = direction * target_shares

                for bidx, w in models[i]['top_k']:
                    bs = min(
                        max(int(target_shares * abs(w) * price_i / prices[bidx, day]), 1),
                        POS_LIMIT,
                    )
                    basket_dir = -direction if w > 0 else direction
                    held[bidx] = held.get(bidx, 0) + basket_dir * bs

                # Verify no stock exceeds position limit
                ok = all(
                    abs(sum(sig.get(s, 0) for sig in active_signals.values())
                        + sh + actions[s, day]) <= POS_LIMIT
                    for s, sh in held.items()
                )
                if ok:
                    for s, sh in held.items():
                        actions[s, day] += sh
                    active_signals[i] = held

    return actions
