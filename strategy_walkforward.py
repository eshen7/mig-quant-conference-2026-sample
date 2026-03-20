"""
Walk-Forward Retraining ETF Stat Arb
=====================================

Uses expanding-window walk-forward optimization:
1. Divide the data into n_chunks (default 4 chunks of ~25% each)
2. Train on chunk 0, trade on chunk 1
3. Train on chunks 0+1, trade on chunk 2
4. Train on chunks 0+1+2, trade on chunk 3
5. Expanding window so each successive model has more training data
6. Concatenate trading-period actions (zeros for first training chunk)

The final fold's model (most training data) also generates in-sample
signals for the initial training period, ensuring full timeline coverage.

Applies the same circuit breaker, vol-scaled sizing, and tighter exits
as etf_arb_tuned.py.
"""

import numpy as np
from itertools import combinations


def _ridge_fit(X, y, alpha=1.0):
    """Ridge regression with intercept (intercept not penalized)."""
    n, p = X.shape
    X_mean = X.mean(axis=0)
    y_mean = y.mean()
    Xc = X - X_mean
    yc = y - y_mean
    A = Xc.T @ Xc + alpha * np.eye(p)
    b = Xc.T @ yc
    coefs = np.linalg.solve(A, b)
    intercept = y_mean - X_mean @ coefs
    weights = np.append(coefs, intercept)
    y_pred = X @ coefs + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return weights, r2


def _find_returns_pairs(prices, train_start, train_end,
                        max_basket_size=4, r2_cutoff=0.80, ridge_alpha=0.1,
                        max_pairs=10):
    """Find cointegrated pairs using log-returns regression."""
    num_stocks = prices.shape[0]
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
                    weights, r2 = _ridge_fit(X, y, alpha=ridge_alpha)
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
    active = []
    for p in pairs:
        if p['target'] not in used_stocks:
            active.append(p)
            used_stocks.add(p['target'])
        if len(active) >= max_pairs:
            break
    return active


def _find_adaptive_pairs(prices, train_start, train_end,
                         max_basket_size=4, r2_cutoff=0.85, ridge_alpha=0.1,
                         max_pairs=10):
    """Find cointegrated pairs using price-level regression."""
    num_stocks = prices.shape[0]
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
                    weights, r2 = _ridge_fit(X, y, alpha=ridge_alpha)
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
    active = []
    for p in pairs:
        if p['target'] not in used_stocks:
            active.append(p)
            used_stocks.add(p['target'])
        if len(active) >= max_pairs:
            break
    return active


def _trade_returns_pair(pair, prices, train_start, train_end, trade_start, trade_end,
                        entry_threshold=0.02, exit_threshold=0.008,
                        max_position=20, hedge_ratio=0.5, vol_lookback=20):
    """Generate actions for a returns-based pair."""
    num_stocks, num_days = prices.shape
    target = pair['target']
    basket = pair['basket']
    weights = pair['weights']

    pair_actions = np.zeros((num_stocks, num_days), dtype=np.float64)

    log_returns = np.log(prices / prices[:, 0:1])
    spreads = np.zeros(num_days)
    for day in range(num_days):
        bv = sum(weights[i] * log_returns[basket[i], day] for i in range(len(basket)))
        bv += weights[-1]
        spreads[day] = log_returns[target, day] - bv

    train_spread = spreads[train_start:train_end]
    spread_mean = np.mean(train_spread)
    cs = spreads - spread_mean

    baseline_vol = np.std(train_spread, ddof=1)
    if baseline_vol < 1e-10:
        return pair_actions
    target_risk = baseline_vol * max_position

    rvol = np.zeros(num_days)
    for day in range(num_days):
        s = max(0, day - vol_lookback + 1)
        w = cs[s:day + 1]
        rvol[day] = np.std(w, ddof=1) if len(w) >= 2 else baseline_vol

    position = 0
    for day in range(max(1, trade_start), trade_end):
        spread = cs[day]
        rv = rvol[day] if rvol[day] > 1e-10 else baseline_vol
        ps = min(max_position, max(1, int(target_risk / rv)))

        if position == 0:
            if spread > entry_threshold:
                position = -1
                pair_actions[target, day] -= ps
                for i, si in enumerate(basket):
                    bsh = max(1, min(int(ps * abs(weights[i]) * hedge_ratio), ps))
                    if weights[i] > 0:
                        pair_actions[si, day] += bsh
                    else:
                        pair_actions[si, day] -= bsh
            elif spread < -entry_threshold:
                position = 1
                pair_actions[target, day] += ps
                for i, si in enumerate(basket):
                    bsh = max(1, min(int(ps * abs(weights[i]) * hedge_ratio), ps))
                    if weights[i] > 0:
                        pair_actions[si, day] -= bsh
                    else:
                        pair_actions[si, day] += bsh
        elif position == 1:
            if spread > exit_threshold:
                position = 0
                pair_actions[target, day] -= ps
                for i, si in enumerate(basket):
                    bsh = max(1, min(int(ps * abs(weights[i]) * hedge_ratio), ps))
                    if weights[i] > 0:
                        pair_actions[si, day] += bsh
                    else:
                        pair_actions[si, day] -= bsh
        elif position == -1:
            if spread < exit_threshold:
                position = 0
                pair_actions[target, day] += ps
                for i, si in enumerate(basket):
                    bsh = max(1, min(int(ps * abs(weights[i]) * hedge_ratio), ps))
                    if weights[i] > 0:
                        pair_actions[si, day] -= bsh
                    else:
                        pair_actions[si, day] += bsh

    return np.clip(pair_actions, -100, 100)


def _trade_adaptive_pair(pair, prices, train_start, train_end, trade_start, trade_end,
                         k_entry=1.0, k_exit=0.2, vol_window=40,
                         max_position=20, hedge_ratio=0.5, vol_lookback=20):
    """Generate actions for an adaptive-threshold pair."""
    num_stocks, num_days = prices.shape
    target = pair['target']
    basket = pair['basket']
    weights = pair['weights']

    pair_actions = np.zeros((num_stocks, num_days), dtype=np.float64)

    spreads = np.zeros(num_days)
    for day in range(num_days):
        bv = sum(weights[i] * prices[basket[i], day] for i in range(len(basket)))
        bv += weights[-1]
        spreads[day] = prices[target, day] - bv

    train_spread = spreads[train_start:train_end]
    spread_mean = np.mean(train_spread)
    cs = spreads - spread_mean

    baseline_vol = np.std(train_spread, ddof=1)
    if baseline_vol < 1e-10:
        return pair_actions
    target_risk = baseline_vol * max_position

    rstd = np.zeros(num_days)
    for day in range(num_days):
        s = max(0, day - vol_window + 1)
        w = cs[s:day + 1]
        rstd[day] = np.std(w, ddof=1) if len(w) >= 2 else baseline_vol

    rvol = np.zeros(num_days)
    for day in range(num_days):
        s = max(0, day - vol_lookback + 1)
        w = cs[s:day + 1]
        rvol[day] = np.std(w, ddof=1) if len(w) >= 2 else baseline_vol

    position = 0
    for day in range(max(1, trade_start), trade_end):
        spread = cs[day]
        vol = rstd[day]
        et = k_entry * vol
        xt = k_exit * vol

        rv = rvol[day] if rvol[day] > 1e-10 else baseline_vol
        ps = min(max_position, max(1, int(target_risk / rv)))

        if position == 0:
            if spread > et:
                position = -1
                pair_actions[target, day] -= ps
                for i, si in enumerate(basket):
                    bsh = max(1, min(int(ps * abs(weights[i]) * hedge_ratio), ps))
                    if weights[i] > 0:
                        pair_actions[si, day] += bsh
                    else:
                        pair_actions[si, day] -= bsh
            elif spread < -et:
                position = 1
                pair_actions[target, day] += ps
                for i, si in enumerate(basket):
                    bsh = max(1, min(int(ps * abs(weights[i]) * hedge_ratio), ps))
                    if weights[i] > 0:
                        pair_actions[si, day] -= bsh
                    else:
                        pair_actions[si, day] += bsh
        elif position == 1:
            if spread > xt:
                position = 0
                pair_actions[target, day] -= ps
                for i, si in enumerate(basket):
                    bsh = max(1, min(int(ps * abs(weights[i]) * hedge_ratio), ps))
                    if weights[i] > 0:
                        pair_actions[si, day] += bsh
                    else:
                        pair_actions[si, day] -= bsh
        elif position == -1:
            if spread < -xt:
                position = 0
                pair_actions[target, day] += ps
                for i, si in enumerate(basket):
                    bsh = max(1, min(int(ps * abs(weights[i]) * hedge_ratio), ps))
                    if weights[i] > 0:
                        pair_actions[si, day] -= bsh
                    else:
                        pair_actions[si, day] += bsh

    return np.clip(pair_actions, -100, 100)


def _apply_circuit_breakers(prices, all_pair_actions,
                            pair_pnl_limit=-500.0,
                            portfolio_drawdown_limit=2000.0,
                            initial_cash=25000.0):
    """Circuit breaker logic, same as etf_arb_tuned.py."""
    num_stocks, num_days = prices.shape
    num_pairs = len(all_pair_actions)

    if num_pairs == 0:
        return np.zeros((num_stocks, num_days), dtype=np.float64)

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

            mtm = sum(positions[s] * prices[s, day] for s in range(num_stocks))
            pair_pnl = cumulative_cash + mtm

            if pair_pnl > peak_pnl:
                peak_pnl = pair_pnl

            if pair_pnl < pair_pnl_limit:
                pair_killed[pidx] = True
                pa[:, day + 1:] = 0

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
                    pass
                else:
                    day_actions[s] = 0

        final_actions[:, day] = day_actions

        for s in range(num_stocks):
            action = day_actions[s]
            if action != 0:
                cash -= action * prices[s, day]
                positions[s] += action

        port_value = cash + sum(positions[s] * prices[s, day] for s in range(num_stocks))

        if port_value > peak_value:
            peak_value = port_value

        drawdown = peak_value - port_value
        in_drawdown = drawdown > portfolio_drawdown_limit

    return final_actions


def get_actions(prices, n_chunks=5,
                ret_entry=0.02, ret_exit=0.008, ret_max_pos=20, ret_r2=0.80,
                adp_k_entry=1.0, adp_k_exit=0.2, adp_vol_window=40, adp_max_pos=20,
                pair_pnl_limit=-350.0, portfolio_drawdown_limit=2500.0):
    """
    Walk-forward retraining with expanding training window.

    Divides data into n_chunks equal segments. For each fold i (i=1..n_chunks-1):
      - Training window:  [0, boundary[i])  -- expanding
      - Trading window:   [boundary[i], boundary[i+1])  -- OOS chunk only

    The final fold's model (trained on the most data) also trades on all
    training-period days [1, boundary[n_chunks-1]), providing full timeline
    coverage while benefiting from periodic retraining on OOS chunks.
    """
    num_stocks, num_days = prices.shape

    chunk_size = num_days // n_chunks
    boundaries = [i * chunk_size for i in range(n_chunks + 1)]
    boundaries[-1] = num_days

    all_pair_actions = []

    # Store each fold's pairs for reuse
    fold_data = {}

    for fold in range(1, n_chunks):
        train_start = 0
        train_end = boundaries[fold]
        trade_start = boundaries[fold]
        trade_end = boundaries[fold + 1]

        if trade_start >= trade_end or train_end < 30:
            continue

        ret_pairs = _find_returns_pairs(prices, train_start, train_end, r2_cutoff=ret_r2)
        adp_pairs = _find_adaptive_pairs(prices, train_start, train_end)

        fold_data[fold] = {
            'ret_pairs': ret_pairs,
            'adp_pairs': adp_pairs,
            'train_start': train_start,
            'train_end': train_end,
        }

        # OOS trading for this fold
        for pair in ret_pairs:
            pa = _trade_returns_pair(
                pair, prices, train_start, train_end,
                trade_start, trade_end,
                entry_threshold=ret_entry, exit_threshold=ret_exit,
                max_position=ret_max_pos,
            )
            all_pair_actions.append(pa)

        for pair in adp_pairs:
            pa = _trade_adaptive_pair(
                pair, prices, train_start, train_end,
                trade_start, trade_end,
                k_entry=adp_k_entry, k_exit=adp_k_exit,
                vol_window=adp_vol_window, max_position=adp_max_pos,
            )
            all_pair_actions.append(pa)

    # Final fold model also trades in-sample (covering the training period)
    last_fold = n_chunks - 1
    if last_fold in fold_data:
        fd = fold_data[last_fold]
        ts, te = fd['train_start'], fd['train_end']

        for pair in fd['ret_pairs']:
            pa = _trade_returns_pair(
                pair, prices, ts, te,
                1, te,  # trade on training period
                entry_threshold=ret_entry, exit_threshold=ret_exit,
                max_position=ret_max_pos,
            )
            all_pair_actions.append(pa)

        for pair in fd['adp_pairs']:
            pa = _trade_adaptive_pair(
                pair, prices, ts, te,
                1, te,  # trade on training period
                k_entry=adp_k_entry, k_exit=adp_k_exit,
                vol_window=adp_vol_window, max_position=adp_max_pos,
            )
            all_pair_actions.append(pa)

    final_actions = _apply_circuit_breakers(
        prices, all_pair_actions,
        pair_pnl_limit=pair_pnl_limit,
        portfolio_drawdown_limit=portfolio_drawdown_limit,
    )
    return final_actions
