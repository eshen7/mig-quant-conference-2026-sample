"""
Combined ETF Stat Arb: Tight Exit Variant
==========================================

Same as strategy_combined.py but with tighter exit thresholds to capture
profits before full mean reversion.

Tuned params (selected via grid search over dev_data_30):
  Returns-based: exit_threshold tuned (was 0.0)
  Adaptive:      k_exit tuned (was 0.1)
"""

import numpy as np
from itertools import combinations


# ---------------------------------------------------------------------------
# Strategy 1: Returns-Based ETF Stat Arb
# ---------------------------------------------------------------------------
def _returns_based(prices, entry_threshold=0.02, exit_threshold=0.008,
                   train_start=0, train_end=None, max_basket_size=4, hedge_ratio=0.5,
                   max_position=20, r2_cutoff=0.80):
    num_stocks, num_days = prices.shape
    actions = np.zeros((num_stocks, num_days), dtype=np.float64)

    if train_end is None:
        train_end = int(num_days * 0.85)

    log_returns = np.log(prices / prices[:, 0:1])
    train_log_returns = log_returns[:, train_start:train_end]

    daily_log_returns = np.diff(np.log(prices + 1e-10), axis=1)[:, train_start:train_end - 1]
    corr_matrix = np.corrcoef(daily_log_returns)

    pairs = []
    for target in range(num_stocks):
        corrs = corr_matrix[target].copy()
        corrs[target] = -1

        top_indices = np.argsort(corrs)[::-1][:max_basket_size * 2]

        best_r2 = 0
        best_basket = None
        best_weights = None

        for bsize in range(2, max_basket_size + 1):
            for basket_indices in combinations(top_indices[:max_basket_size * 2], bsize):
                basket_indices = list(basket_indices)

                X = train_log_returns[basket_indices, :].T
                y = train_log_returns[target, :].T
                X_with_const = np.column_stack([X, np.ones(len(X))])

                try:
                    weights, _, _, _ = np.linalg.lstsq(X_with_const, y, rcond=None)
                    y_pred = X_with_const @ weights
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                    if r2 > best_r2:
                        best_r2 = r2
                        best_basket = basket_indices
                        best_weights = weights
                except:
                    continue

        if best_r2 > r2_cutoff and best_basket is not None:
            pairs.append({
                'target': target, 'basket': best_basket,
                'weights': best_weights, 'r2': best_r2,
            })

    pairs.sort(key=lambda x: x['r2'], reverse=True)
    used_stocks = set()
    active_pairs = []
    for p in pairs:
        if p['target'] not in used_stocks:
            active_pairs.append(p)
            used_stocks.add(p['target'])
        if len(active_pairs) >= 10:
            break

    for pair in active_pairs:
        target = pair['target']
        basket = pair['basket']
        weights = pair['weights']

        spreads = np.zeros(num_days)
        for day in range(num_days):
            basket_val = sum(weights[i] * log_returns[basket[i], day] for i in range(len(basket)))
            basket_val += weights[-1]
            spreads[day] = log_returns[target, day] - basket_val

        train_spread = spreads[train_start:train_end]
        spread_mean = np.mean(train_spread)
        centered_spreads = spreads - spread_mean

        position = 0
        for day in range(1, num_days):
            spread = centered_spreads[day]

            if position == 0:
                if spread > entry_threshold:
                    position = -1
                    actions[target, day] -= max_position
                    for i, stock_idx in enumerate(basket):
                        basket_shares = int(max_position * abs(weights[i]) * hedge_ratio)
                        basket_shares = max(1, min(basket_shares, max_position))
                        if weights[i] > 0:
                            actions[stock_idx, day] += basket_shares
                        else:
                            actions[stock_idx, day] -= basket_shares
                elif spread < -entry_threshold:
                    position = 1
                    actions[target, day] += max_position
                    for i, stock_idx in enumerate(basket):
                        basket_shares = int(max_position * abs(weights[i]) * hedge_ratio)
                        basket_shares = max(1, min(basket_shares, max_position))
                        if weights[i] > 0:
                            actions[stock_idx, day] -= basket_shares
                        else:
                            actions[stock_idx, day] += basket_shares

            elif position == 1:
                if spread > exit_threshold:
                    position = 0
                    actions[target, day] -= max_position
                    for i, stock_idx in enumerate(basket):
                        basket_shares = int(max_position * abs(weights[i]) * hedge_ratio)
                        basket_shares = max(1, min(basket_shares, max_position))
                        if weights[i] > 0:
                            actions[stock_idx, day] += basket_shares
                        else:
                            actions[stock_idx, day] -= basket_shares

            elif position == -1:
                if spread < exit_threshold:
                    position = 0
                    actions[target, day] += max_position
                    for i, stock_idx in enumerate(basket):
                        basket_shares = int(max_position * abs(weights[i]) * hedge_ratio)
                        basket_shares = max(1, min(basket_shares, max_position))
                        if weights[i] > 0:
                            actions[stock_idx, day] -= basket_shares
                        else:
                            actions[stock_idx, day] += basket_shares

    actions = np.clip(actions, -100, 100)
    return actions


# ---------------------------------------------------------------------------
# Strategy 2: Adaptive Threshold ETF Stat Arb
# ---------------------------------------------------------------------------
def _adaptive_threshold(prices, k_entry=1.0, k_exit=0.2, vol_window=40,
                        train_start=0, train_end=None, max_basket_size=4, hedge_ratio=0.5,
                        max_position=20):
    num_stocks, num_days = prices.shape
    actions = np.zeros((num_stocks, num_days), dtype=np.float64)

    if train_end is None:
        train_end = int(num_days * 0.85)
    train_prices = prices[:, train_start:train_end]

    log_returns = np.diff(np.log(train_prices + 1e-10), axis=1)
    corr_matrix = np.corrcoef(log_returns)

    pairs = []
    for target in range(num_stocks):
        corrs = corr_matrix[target].copy()
        corrs[target] = -1

        top_indices = np.argsort(corrs)[::-1][:max_basket_size * 2]

        best_r2 = 0
        best_basket = None
        best_weights = None

        for bsize in range(2, max_basket_size + 1):
            for basket_indices in combinations(top_indices[:max_basket_size * 2], bsize):
                basket_indices = list(basket_indices)

                X = train_prices[basket_indices, :].T
                y = train_prices[target, :].T
                X_with_const = np.column_stack([X, np.ones(len(X))])

                try:
                    weights, _, _, _ = np.linalg.lstsq(X_with_const, y, rcond=None)
                    y_pred = X_with_const @ weights
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                    if r2 > best_r2:
                        best_r2 = r2
                        best_basket = basket_indices
                        best_weights = weights
                except:
                    continue

        if best_r2 > 0.85 and best_basket is not None:
            pairs.append({
                'target': target, 'basket': best_basket,
                'weights': best_weights, 'r2': best_r2,
            })

    pairs.sort(key=lambda x: x['r2'], reverse=True)
    used_stocks = set()
    active_pairs = []
    for p in pairs:
        if p['target'] not in used_stocks:
            active_pairs.append(p)
            used_stocks.add(p['target'])
        if len(active_pairs) >= 10:
            break

    for pair in active_pairs:
        target = pair['target']
        basket = pair['basket']
        weights = pair['weights']

        spreads = np.zeros(num_days)
        for day in range(num_days):
            basket_val = sum(weights[i] * prices[basket[i], day] for i in range(len(basket)))
            basket_val += weights[-1]
            spreads[day] = prices[target, day] - basket_val

        train_spread = spreads[train_start:train_end]
        spread_mean = np.mean(train_spread)
        centered_spreads = spreads - spread_mean

        rolling_std = np.zeros(num_days)
        for day in range(num_days):
            start = max(0, day - vol_window + 1)
            window_data = centered_spreads[start:day + 1]
            if len(window_data) >= 2:
                rolling_std[day] = np.std(window_data, ddof=1)
            else:
                rolling_std[day] = np.std(train_spread, ddof=1)

        position = 0
        for day in range(1, num_days):
            spread = centered_spreads[day]
            vol = rolling_std[day]

            entry_threshold = k_entry * vol
            exit_threshold = k_exit * vol

            if position == 0:
                if spread > entry_threshold:
                    position = -1
                    actions[target, day] -= max_position
                    for i, stock_idx in enumerate(basket):
                        basket_shares = int(max_position * abs(weights[i]) * hedge_ratio)
                        basket_shares = max(1, min(basket_shares, max_position))
                        if weights[i] > 0:
                            actions[stock_idx, day] += basket_shares
                        else:
                            actions[stock_idx, day] -= basket_shares
                elif spread < -entry_threshold:
                    position = 1
                    actions[target, day] += max_position
                    for i, stock_idx in enumerate(basket):
                        basket_shares = int(max_position * abs(weights[i]) * hedge_ratio)
                        basket_shares = max(1, min(basket_shares, max_position))
                        if weights[i] > 0:
                            actions[stock_idx, day] -= basket_shares
                        else:
                            actions[stock_idx, day] += basket_shares

            elif position == 1:
                if spread > exit_threshold:
                    position = 0
                    actions[target, day] -= max_position
                    for i, stock_idx in enumerate(basket):
                        basket_shares = int(max_position * abs(weights[i]) * hedge_ratio)
                        basket_shares = max(1, min(basket_shares, max_position))
                        if weights[i] > 0:
                            actions[stock_idx, day] += basket_shares
                        else:
                            actions[stock_idx, day] -= basket_shares

            elif position == -1:
                if spread < -exit_threshold:
                    position = 0
                    actions[target, day] += max_position
                    for i, stock_idx in enumerate(basket):
                        basket_shares = int(max_position * abs(weights[i]) * hedge_ratio)
                        basket_shares = max(1, min(basket_shares, max_position))
                        if weights[i] > 0:
                            actions[stock_idx, day] -= basket_shares
                        else:
                            actions[stock_idx, day] += basket_shares

    actions = np.clip(actions, -100, 100)
    return actions


# ---------------------------------------------------------------------------
# Combined Ensemble
# ---------------------------------------------------------------------------
def get_actions(prices,
                ret_entry=0.02, ret_exit=0.008, ret_max_pos=20, ret_r2=0.80,
                adp_k_entry=1.0, adp_k_exit=0.2, adp_vol_window=40, adp_max_pos=20,
                train_frac=0.85, train_start=None, train_end=None):
    num_days = prices.shape[1]
    if train_start is None:
        train_start = 0
    if train_end is None:
        train_end = int(num_days * train_frac)
    acts_ret = _returns_based(
        prices, entry_threshold=ret_entry, exit_threshold=ret_exit,
        max_position=ret_max_pos,
        r2_cutoff=ret_r2, train_start=train_start, train_end=train_end,
    )
    acts_adp = _adaptive_threshold(
        prices, k_entry=adp_k_entry, k_exit=adp_k_exit,
        vol_window=adp_vol_window, max_position=adp_max_pos,
        train_start=train_start, train_end=train_end,
    )
    return np.clip(acts_ret + acts_adp, -100, 100)
