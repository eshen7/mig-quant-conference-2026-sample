"""
PCA Residual Mean Reversion Strategy
=====================================

Structurally different from pair/basket stat arb:

1. Compute PCA on stock returns during training period (K principal components)
2. For each stock, residual = actual return - PCA-predicted return
   These residuals are idiosyncratic moves stripped of common factor exposure
3. Accumulate residuals over time and compute rolling z-scores
4. Trade mean reversion on z-score:
   - z > entry_z  -> short (idiosyncratic overperformance reverts)
   - z < -entry_z -> long  (idiosyncratic underperformance reverts)
   - |z| < exit_z -> flatten
5. Volatility-scaled position sizing (from etf_arb_tuned)
6. Circuit breakers: per-stock PnL kill switch + portfolio drawdown protection
"""

import numpy as np


def _apply_circuit_breakers(prices, stock_actions_list,
                            stock_pnl_limit=-500.0,
                            portfolio_drawdown_limit=2000.0,
                            initial_cash=25000.0):
    """
    Circuit breaker logic matching etf_arb_tuned.

    stock_actions_list: list of (num_stocks, num_days) arrays, one per traded stock.
    """
    num_stocks, num_days = prices.shape
    num_entries = len(stock_actions_list)

    if num_entries == 0:
        return np.zeros((num_stocks, num_days), dtype=np.float64)

    # --- Per-stock circuit breaker ---
    for eidx in range(num_entries):
        sa = stock_actions_list[eidx]
        positions = np.zeros(num_stocks)
        cumulative_cash = 0.0

        for day in range(num_days):
            for s in range(num_stocks):
                action = sa[s, day]
                if action != 0:
                    cumulative_cash -= action * prices[s, day]
                    positions[s] += action

            mtm = 0.0
            for s in range(num_stocks):
                mtm += positions[s] * prices[s, day]
            entry_pnl = cumulative_cash + mtm

            if entry_pnl < stock_pnl_limit:
                sa[:, day + 1:] = 0
                break

    # --- Combine and apply portfolio-level circuit breaker ---
    combined = np.zeros((num_stocks, num_days), dtype=np.float64)
    for sa in stock_actions_list:
        combined += sa
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
                    pass  # reducing position - allow
                else:
                    day_actions[s] = 0  # increasing position - block

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
        in_drawdown = drawdown > portfolio_drawdown_limit

    return final_actions


def get_actions(prices,
                n_components=5,
                train_frac=0.85,
                entry_z=1.5,
                exit_z=0.3,
                max_position=15,
                vol_lookback=20,
                zscore_lookback=40,
                residual_lookback=60,
                max_traded_stocks=15):
    """
    PCA Residual Mean Reversion.

    Parameters
    ----------
    prices : ndarray, shape (num_stocks, num_days)
    n_components : int — number of PCA factors to extract
    train_frac : float — fraction of days used for training
    entry_z : float — z-score threshold to open a position
    exit_z : float — z-score threshold to close a position
    max_position : int — max shares per stock
    vol_lookback : int — window for rolling volatility (position sizing)
    zscore_lookback : int — window for rolling z-score of cumulative residual
    residual_lookback : int — window for rolling mean/std of residuals
    max_traded_stocks : int — cap on number of stocks traded
    """
    num_stocks, num_days = prices.shape
    train_end = int(num_days * train_frac)

    # --- Step 1: Compute daily log returns ---
    log_prices = np.log(prices + 1e-10)
    daily_returns = np.diff(log_prices, axis=1)  # (num_stocks, num_days-1)

    train_returns = daily_returns[:, :train_end - 1]  # training period returns

    # --- Step 2: PCA on training returns ---
    # Center the returns (each stock zero-mean over training)
    mean_returns = train_returns.mean(axis=1, keepdims=True)
    centered = train_returns - mean_returns  # (num_stocks, train_days)

    # Covariance matrix of stocks (stocks x stocks)
    cov_matrix = np.cov(centered)  # (num_stocks, num_stocks)

    # Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # eigh returns ascending order; take last n_components
    n_comp = min(n_components, num_stocks)
    idx = np.argsort(eigenvalues)[::-1][:n_comp]
    components = eigenvectors[:, idx]  # (num_stocks, n_comp) — factor loadings

    # Factor loadings matrix: each stock's exposure to each factor
    # For each stock i, predicted return = sum_k (loading_ik * factor_k_t)
    # factor_k_t = components[:, k]^T @ returns_t  (cross-sectional)

    # --- Step 3: Compute residuals over entire period ---
    # For each day, project cross-sectional returns onto factors, get predicted
    all_returns = daily_returns  # (num_stocks, num_days-1)
    mean_ret = mean_returns.flatten()  # training mean per stock

    residuals = np.zeros_like(all_returns)  # (num_stocks, num_days-1)

    for t in range(all_returns.shape[1]):
        r_t = all_returns[:, t]  # (num_stocks,)
        r_centered = r_t - mean_ret
        # Project onto PCA space: factor scores
        factor_scores = components.T @ r_centered  # (n_comp,)
        # Reconstruct
        r_predicted = components @ factor_scores + mean_ret
        residuals[:, t] = r_t - r_predicted

    # --- Step 4: Cumulative residuals and rolling z-scores ---
    cum_residuals = np.cumsum(residuals, axis=1)  # (num_stocks, num_days-1)

    # --- Step 5: Select stocks with strongest mean-reverting residuals ---
    # Use training period residual volatility as a proxy for tradability
    train_residuals = residuals[:, :train_end - 1]
    residual_vol = np.std(train_residuals, axis=1, ddof=1)  # per stock

    # Compute Hurst exponent approximation via variance ratio
    # If variance ratio < 1, residuals are mean-reverting
    variance_ratios = np.ones(num_stocks)
    for s in range(num_stocks):
        r = train_residuals[s]
        if len(r) < 20:
            continue
        var1 = np.var(r, ddof=1)
        if var1 < 1e-15:
            continue
        # Variance of 5-day sums vs 5*variance of 1-day
        n5 = len(r) // 5 * 5
        r5 = r[:n5].reshape(-1, 5).sum(axis=1)
        var5 = np.var(r5, ddof=1)
        variance_ratios[s] = var5 / (5 * var1) if var1 > 0 else 1.0

    # Select stocks with low variance ratio (mean-reverting) and decent vol
    # Filter: need some minimum volatility to trade
    min_vol = np.percentile(residual_vol[residual_vol > 0], 20)
    tradable = (residual_vol > min_vol) & (variance_ratios < 1.0)

    # Rank by variance ratio (lower = more mean-reverting)
    stock_scores = np.where(tradable, variance_ratios, 999.0)
    ranked_stocks = np.argsort(stock_scores)[:max_traded_stocks]
    ranked_stocks = ranked_stocks[stock_scores[ranked_stocks] < 999.0]

    if len(ranked_stocks) == 0:
        # Fallback: trade all stocks
        ranked_stocks = np.argsort(variance_ratios)[:max_traded_stocks]

    # --- Step 6: Generate trading signals per stock ---
    stock_actions_list = []

    for stock_idx in ranked_stocks:
        stock_actions = np.zeros((num_stocks, num_days), dtype=np.float64)

        cum_res = cum_residuals[stock_idx]  # (num_days-1,)
        stock_res_vol = residual_vol[stock_idx]

        if stock_res_vol < 1e-10:
            continue

        # Baseline vol for position sizing
        baseline_vol = stock_res_vol
        target_risk = baseline_vol * max_position

        position = 0  # +1 long, -1 short, 0 flat

        for t in range(1, len(cum_res)):
            day = t + 1  # actions array index (day 0 has no return)

            if day >= num_days:
                break

            # Rolling z-score of cumulative residual
            start_z = max(0, t - zscore_lookback + 1)
            window = cum_res[start_z:t + 1]
            if len(window) < 5:
                continue

            z_mean = np.mean(window)
            z_std = np.std(window, ddof=1)
            if z_std < 1e-10:
                continue

            z = (cum_res[t] - z_mean) / z_std

            # Rolling vol for position sizing
            start_v = max(0, t - vol_lookback + 1)
            res_window = residuals[stock_idx, start_v:t + 1]
            if len(res_window) >= 2:
                rolling_vol = np.std(res_window, ddof=1)
            else:
                rolling_vol = baseline_vol

            rv = rolling_vol if rolling_vol > 1e-10 else baseline_vol
            pos_size = min(max_position, max(1, int(target_risk / rv)))

            if position == 0:
                if z > entry_z:
                    # Idiosyncratic overperformance -> short for reversion
                    position = -1
                    stock_actions[stock_idx, day] -= pos_size
                elif z < -entry_z:
                    # Idiosyncratic underperformance -> long for reversion
                    position = 1
                    stock_actions[stock_idx, day] += pos_size
            elif position == 1:
                if z > -exit_z:
                    # Reverted enough, exit long
                    position = 0
                    stock_actions[stock_idx, day] -= pos_size
            elif position == -1:
                if z < exit_z:
                    # Reverted enough, exit short
                    position = 0
                    stock_actions[stock_idx, day] += pos_size

        stock_actions = np.clip(stock_actions, -100, 100)
        stock_actions_list.append(stock_actions)

    # --- Step 7: Apply circuit breakers ---
    final_actions = _apply_circuit_breakers(prices, stock_actions_list)

    return final_actions
