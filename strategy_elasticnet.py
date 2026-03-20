"""
Combined ETF Stat Arb with Elastic Net Regression
===================================================

Replaces ridge regression (_ridge_fit) with elastic net (L1+L2 regularization).
The L1 penalty zeros out weak basket constituents, reducing model complexity.
Multiple alpha and l1_ratio combinations are tried; best is picked via in-sample R².

Otherwise identical to etf_arb_tuned.py:
1. Volatility-scaled position sizing
2. Tighter exit thresholds
3. Circuit breaker
"""

import numpy as np
from itertools import combinations
from sklearn.linear_model import ElasticNet


def _elasticnet_fit(X, y, alphas=None, l1_ratios=None):
    """Elastic net regression with intercept. Tries multiple (alpha, l1_ratio)
    combinations and returns the one with the best in-sample R².

    X: (n_samples, n_features), y: (n_samples,)
    Returns weights array of shape (n_features + 1,) where last element is intercept,
    and the best R².
    """
    if alphas is None:
        alphas = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    if l1_ratios is None:
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]

    best_r2 = -np.inf
    best_coefs = None
    best_intercept = None

    y_mean = y.mean()
    ss_tot = np.sum((y - y_mean) ** 2)
    if ss_tot == 0:
        weights = np.zeros(X.shape[1] + 1)
        weights[-1] = y_mean
        return weights, 0.0

    for alpha in alphas:
        for l1_ratio in l1_ratios:
            model = ElasticNet(
                alpha=alpha, l1_ratio=l1_ratio,
                fit_intercept=True, max_iter=5000, tol=1e-5,
            )
            model.fit(X, y)
            y_pred = model.predict(X)
            ss_res = np.sum((y - y_pred) ** 2)
            r2 = 1.0 - ss_res / ss_tot

            if r2 > best_r2:
                best_r2 = r2
                best_coefs = model.coef_.copy()
                best_intercept = model.intercept_

    weights = np.append(best_coefs, best_intercept)
    return weights, best_r2


# ---------------------------------------------------------------------------
# Strategy 1: Returns-Based ETF Stat Arb
# ---------------------------------------------------------------------------
def _returns_based(prices, entry_threshold=0.02, exit_threshold=0.008,
                   train_start=0, train_end=None, max_basket_size=4, hedge_ratio=0.5,
                   max_position=20, r2_cutoff=0.80, vol_lookback=20):
    num_stocks, num_days = prices.shape
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
                try:
                    weights, r2 = _elasticnet_fit(X, y)
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

    # Build per-pair action arrays (for circuit breaker)
    pair_actions_list = []
    for pair in active_pairs:
        target = pair['target']
        basket = pair['basket']
        weights = pair['weights']

        pair_actions = np.zeros((num_stocks, num_days), dtype=np.float64)

        spreads = np.zeros(num_days)
        for day in range(num_days):
            basket_val = sum(weights[i] * log_returns[basket[i], day] for i in range(len(basket)))
            basket_val += weights[-1]
            spreads[day] = log_returns[target, day] - basket_val

        train_spread = spreads[train_start:train_end]
        spread_mean = np.mean(train_spread)
        centered_spreads = spreads - spread_mean

        # Volatility-scaled position sizing
        baseline_vol = np.std(train_spread, ddof=1)
        target_risk = baseline_vol * max_position

        rolling_vol = np.zeros(num_days)
        for day in range(num_days):
            start = max(0, day - vol_lookback + 1)
            window_data = centered_spreads[start:day + 1]
            if len(window_data) >= 2:
                rolling_vol[day] = np.std(window_data, ddof=1)
            else:
                rolling_vol[day] = baseline_vol

        position = 0
        for day in range(1, num_days):
            spread = centered_spreads[day]

            # Vol-scaled position size
            rv = rolling_vol[day] if rolling_vol[day] > 1e-10 else baseline_vol
            pos_size = min(max_position, max(1, int(target_risk / rv)))

            if position == 0:
                if spread > entry_threshold:
                    position = -1
                    pair_actions[target, day] -= pos_size
                    for i, stock_idx in enumerate(basket):
                        basket_shares = int(pos_size * abs(weights[i]) * hedge_ratio)
                        basket_shares = max(1, min(basket_shares, pos_size))
                        if weights[i] > 0:
                            pair_actions[stock_idx, day] += basket_shares
                        else:
                            pair_actions[stock_idx, day] -= basket_shares
                elif spread < -entry_threshold:
                    position = 1
                    pair_actions[target, day] += pos_size
                    for i, stock_idx in enumerate(basket):
                        basket_shares = int(pos_size * abs(weights[i]) * hedge_ratio)
                        basket_shares = max(1, min(basket_shares, pos_size))
                        if weights[i] > 0:
                            pair_actions[stock_idx, day] -= basket_shares
                        else:
                            pair_actions[stock_idx, day] += basket_shares
            elif position == 1:
                if spread > exit_threshold:
                    position = 0
                    pair_actions[target, day] -= pos_size
                    for i, stock_idx in enumerate(basket):
                        basket_shares = int(pos_size * abs(weights[i]) * hedge_ratio)
                        basket_shares = max(1, min(basket_shares, pos_size))
                        if weights[i] > 0:
                            pair_actions[stock_idx, day] += basket_shares
                        else:
                            pair_actions[stock_idx, day] -= basket_shares
            elif position == -1:
                if spread < exit_threshold:
                    position = 0
                    pair_actions[target, day] += pos_size
                    for i, stock_idx in enumerate(basket):
                        basket_shares = int(pos_size * abs(weights[i]) * hedge_ratio)
                        basket_shares = max(1, min(basket_shares, pos_size))
                        if weights[i] > 0:
                            pair_actions[stock_idx, day] -= basket_shares
                        else:
                            pair_actions[stock_idx, day] += basket_shares

        pair_actions = np.clip(pair_actions, -100, 100)
        pair_actions_list.append(pair_actions)

    return pair_actions_list


# ---------------------------------------------------------------------------
# Strategy 2: Adaptive Threshold ETF Stat Arb
# ---------------------------------------------------------------------------
def _adaptive_threshold(prices, k_entry=1.0, k_exit=0.2, vol_window=40,
                        train_start=0, train_end=None, max_basket_size=4, hedge_ratio=0.5,
                        max_position=20, vol_lookback=20):
    num_stocks, num_days = prices.shape
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
                try:
                    weights, r2 = _elasticnet_fit(X, y)
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

    # Build per-pair action arrays (for circuit breaker)
    pair_actions_list = []
    for pair in active_pairs:
        target = pair['target']
        basket = pair['basket']
        weights = pair['weights']

        pair_actions = np.zeros((num_stocks, num_days), dtype=np.float64)

        spreads = np.zeros(num_days)
        for day in range(num_days):
            basket_val = sum(weights[i] * prices[basket[i], day] for i in range(len(basket)))
            basket_val += weights[-1]
            spreads[day] = prices[target, day] - basket_val

        train_spread = spreads[train_start:train_end]
        spread_mean = np.mean(train_spread)
        centered_spreads = spreads - spread_mean

        # Volatility-scaled position sizing
        baseline_vol = np.std(train_spread, ddof=1)
        target_risk = baseline_vol * max_position

        # Rolling std for adaptive thresholds
        rolling_std = np.zeros(num_days)
        for day in range(num_days):
            start = max(0, day - vol_window + 1)
            window_data = centered_spreads[start:day + 1]
            if len(window_data) >= 2:
                rolling_std[day] = np.std(window_data, ddof=1)
            else:
                rolling_std[day] = np.std(train_spread, ddof=1)

        # Rolling vol for position sizing (separate lookback)
        rolling_vol = np.zeros(num_days)
        for day in range(num_days):
            start = max(0, day - vol_lookback + 1)
            window_data = centered_spreads[start:day + 1]
            if len(window_data) >= 2:
                rolling_vol[day] = np.std(window_data, ddof=1)
            else:
                rolling_vol[day] = baseline_vol

        position = 0
        for day in range(1, num_days):
            spread = centered_spreads[day]
            vol = rolling_std[day]

            entry_threshold = k_entry * vol
            exit_threshold_val = k_exit * vol

            # Vol-scaled position size
            rv = rolling_vol[day] if rolling_vol[day] > 1e-10 else baseline_vol
            pos_size = min(max_position, max(1, int(target_risk / rv)))

            if position == 0:
                if spread > entry_threshold:
                    position = -1
                    pair_actions[target, day] -= pos_size
                    for i, stock_idx in enumerate(basket):
                        basket_shares = int(pos_size * abs(weights[i]) * hedge_ratio)
                        basket_shares = max(1, min(basket_shares, pos_size))
                        if weights[i] > 0:
                            pair_actions[stock_idx, day] += basket_shares
                        else:
                            pair_actions[stock_idx, day] -= basket_shares
                elif spread < -entry_threshold:
                    position = 1
                    pair_actions[target, day] += pos_size
                    for i, stock_idx in enumerate(basket):
                        basket_shares = int(pos_size * abs(weights[i]) * hedge_ratio)
                        basket_shares = max(1, min(basket_shares, pos_size))
                        if weights[i] > 0:
                            pair_actions[stock_idx, day] -= basket_shares
                        else:
                            pair_actions[stock_idx, day] += basket_shares
            elif position == 1:
                if spread > exit_threshold_val:
                    position = 0
                    pair_actions[target, day] -= pos_size
                    for i, stock_idx in enumerate(basket):
                        basket_shares = int(pos_size * abs(weights[i]) * hedge_ratio)
                        basket_shares = max(1, min(basket_shares, pos_size))
                        if weights[i] > 0:
                            pair_actions[stock_idx, day] += basket_shares
                        else:
                            pair_actions[stock_idx, day] -= basket_shares
            elif position == -1:
                if spread < -exit_threshold_val:
                    position = 0
                    pair_actions[target, day] += pos_size
                    for i, stock_idx in enumerate(basket):
                        basket_shares = int(pos_size * abs(weights[i]) * hedge_ratio)
                        basket_shares = max(1, min(basket_shares, pos_size))
                        if weights[i] > 0:
                            pair_actions[stock_idx, day] -= basket_shares
                        else:
                            pair_actions[stock_idx, day] += basket_shares

        pair_actions = np.clip(pair_actions, -100, 100)
        pair_actions_list.append(pair_actions)

    return pair_actions_list


# ---------------------------------------------------------------------------
# Circuit Breaker Logic (from strategy_circuit.py)
# ---------------------------------------------------------------------------
def _apply_circuit_breakers(prices, all_pair_actions,
                            pair_pnl_limit=-500.0,
                            portfolio_drawdown_limit=2000.0,
                            initial_cash=25000.0):
    """
    Forward-simulate approximate PnL and apply circuit breakers.

    1. Per-pair: track each pair's cumulative PnL. If it drops below
       pair_pnl_limit, zero out that pair's future actions (kill switch).
    2. Portfolio-level: track total portfolio value estimate. If drawdown
       from peak exceeds portfolio_drawdown_limit, only allow exits
       (actions that reduce position magnitude) until recovery.
    """
    num_stocks, num_days = prices.shape
    num_pairs = len(all_pair_actions)

    if num_pairs == 0:
        return np.zeros((num_stocks, num_days), dtype=np.float64)

    # --- Per-pair circuit breaker ---
    pair_killed = [False] * num_pairs

    for pidx in range(num_pairs):
        pa = all_pair_actions[pidx]
        positions = np.zeros(num_stocks)
        cumulative_cash = 0.0
        peak_pnl = 0.0

        for day in range(num_days):
            if pair_killed[pidx]:
                pa[:, day:] = 0
                break

            for s in range(num_stocks):
                action = pa[s, day]
                if action != 0:
                    cumulative_cash -= action * prices[s, day]
                    positions[s] += action

            mtm = 0.0
            for s in range(num_stocks):
                mtm += positions[s] * prices[s, day]
            pair_pnl = cumulative_cash + mtm

            if pair_pnl > peak_pnl:
                peak_pnl = pair_pnl

            if pair_pnl < pair_pnl_limit:
                pair_killed[pidx] = True
                pa[:, day + 1:] = 0

    # --- Portfolio-level circuit breaker ---
    combined = np.zeros((num_stocks, num_days), dtype=np.float64)
    for pa in all_pair_actions:
        combined += pa
    combined = np.clip(combined, -100, 100)

    positions = np.zeros(num_stocks)
    cash = initial_cash
    peak_value = initial_cash
    in_drawdown = False

    final_actions = np.zeros((num_stocks, num_days), dtype=np.float64)

    for day in range(num_days):
        day_actions = combined[:, day].copy()

        if in_drawdown:
            for s in range(num_stocks):
                action = day_actions[s]
                if action == 0:
                    continue
                current_pos = positions[s]
                new_pos = current_pos + action
                if abs(new_pos) < abs(current_pos):
                    pass  # closing/reducing - allow
                else:
                    day_actions[s] = 0  # opening/increasing - block

        final_actions[:, day] = day_actions

        for s in range(num_stocks):
            action = day_actions[s]
            if action != 0:
                cash -= action * prices[s, day]
                positions[s] += action

        port_value = cash
        for s in range(num_stocks):
            port_value += positions[s] * prices[s, day]

        if port_value > peak_value:
            peak_value = port_value

        drawdown = peak_value - port_value
        if drawdown > portfolio_drawdown_limit:
            in_drawdown = True
        else:
            in_drawdown = False

    return final_actions


# ---------------------------------------------------------------------------
# Combined Ensemble with All Three Improvements
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

    # Get per-pair action arrays from each sub-strategy
    ret_pairs = _returns_based(
        prices, entry_threshold=ret_entry, exit_threshold=ret_exit,
        max_position=ret_max_pos,
        r2_cutoff=ret_r2, train_start=train_start, train_end=train_end,
    )
    adp_pairs = _adaptive_threshold(
        prices, k_entry=adp_k_entry, k_exit=adp_k_exit,
        vol_window=adp_vol_window, max_position=adp_max_pos,
        train_start=train_start, train_end=train_end,
    )

    # Combine all pair actions and apply circuit breakers
    all_pair_actions = ret_pairs + adp_pairs
    final_actions = _apply_circuit_breakers(prices, all_pair_actions)

    return final_actions
