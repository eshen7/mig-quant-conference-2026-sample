import numpy as np
from itertools import combinations


def get_actions(prices, entry_threshold=0.05, exit_threshold=0.0, train_frac=0.85,
                max_basket_size=4, hedge_ratio=0.5, max_position=15, r2_cutoff=0.85):
    """
    ETF/Basket Statistical Arbitrage using LOG RETURNS instead of raw prices.

    Key difference from raw-price version:
    - Converts prices to cumulative log returns: log(P_t / P_0)
    - Runs OLS on cumulative log returns to find basket weights
    - Computes spread in return-space (invariant to price level changes)
    - Thresholds are in return-space (much smaller than dollar thresholds)

    This makes the model invariant to price level shifts (e.g., COVID regime change).
    """
    num_stocks, num_days = prices.shape
    actions = np.zeros((num_stocks, num_days), dtype=np.float64)

    train_end = int(num_days * train_frac)

    # Convert prices to cumulative log returns (relative to day 0)
    log_returns = np.log(prices / prices[:, 0:1])

    train_log_returns = log_returns[:, :train_end]

    # Compute correlation on daily log returns for pair selection
    daily_log_returns = np.diff(np.log(prices + 1e-10), axis=1)[:, :train_end - 1]
    corr_matrix = np.corrcoef(daily_log_returns)

    # Find basket pairs using OLS on cumulative log returns
    pairs = []

    for target in range(num_stocks):
        corrs = corr_matrix[target].copy()
        corrs[target] = -1  # exclude self

        top_indices = np.argsort(corrs)[::-1][:max_basket_size * 2]

        best_r2 = 0
        best_basket = None
        best_weights = None

        for bsize in range(2, max_basket_size + 1):
            for basket_indices in combinations(top_indices[:max_basket_size * 2], bsize):
                basket_indices = list(basket_indices)

                # OLS regression on CUMULATIVE LOG RETURNS
                X = train_log_returns[basket_indices, :].T  # (train_days, bsize)
                y = train_log_returns[target, :].T  # (train_days,)

                # Add intercept
                X_with_const = np.column_stack([X, np.ones(len(X))])

                try:
                    weights, residuals, rank, sv = np.linalg.lstsq(X_with_const, y, rcond=None)

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
                'target': target,
                'basket': best_basket,
                'weights': best_weights,  # includes intercept as last element
                'r2': best_r2,
            })

    # Sort by R2 and take top pairs (avoid overlapping targets)
    pairs.sort(key=lambda x: x['r2'], reverse=True)

    used_stocks = set()
    active_pairs = []
    for p in pairs:
        if p['target'] not in used_stocks:
            active_pairs.append(p)
            used_stocks.add(p['target'])
        if len(active_pairs) >= 10:
            break

    # Compute spreads in return-space and trade
    for pair in active_pairs:
        target = pair['target']
        basket = pair['basket']
        weights = pair['weights']

        # Compute spread in LOG RETURN space over full period
        spreads = np.zeros(num_days)
        for day in range(num_days):
            basket_val = sum(weights[i] * log_returns[basket[i], day] for i in range(len(basket)))
            basket_val += weights[-1]  # intercept
            spreads[day] = log_returns[target, day] - basket_val

        # Center spread using training mean
        train_spread = spreads[:train_end]
        spread_mean = np.mean(train_spread)
        centered_spreads = spreads - spread_mean

        # Trading logic with fixed thresholds (in return-space)
        position = 0  # +1 = long spread, -1 = short spread, 0 = flat

        for day in range(1, num_days):
            spread = centered_spreads[day]

            if position == 0:
                if spread > entry_threshold:
                    # Spread too high -> short spread (sell target, buy basket)
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
                    # Spread too low -> long spread (buy target, sell basket)
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

    # Clip actions to respect position limits
    actions = np.clip(actions, -100, 100)

    return actions
