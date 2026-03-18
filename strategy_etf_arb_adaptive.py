import numpy as np
from itertools import combinations


def get_actions(prices, k_entry=2.0, k_exit=0.25, vol_window=40,
                train_frac=0.85, max_basket_size=4, hedge_ratio=0.5, max_position=15):
    """
    ETF/Basket Statistical Arbitrage with Adaptive Volatility-Scaled Thresholds.

    Instead of fixed dollar thresholds, entry/exit thresholds scale with
    the rolling standard deviation of the spread:
        entry_threshold = k_entry * rolling_std(spread, vol_window)
        exit_threshold  = k_exit  * rolling_std(spread, vol_window)

    This adapts to regime changes: tight thresholds in calm markets,
    wide thresholds in volatile markets.
    """
    num_stocks, num_days = prices.shape
    actions = np.zeros((num_stocks, num_days), dtype=np.float64)

    train_end = int(num_days * train_frac)
    train_prices = prices[:, :train_end]

    # Compute log returns for correlation analysis
    log_returns = np.diff(np.log(train_prices + 1e-10), axis=1)

    # Correlation matrix
    corr_matrix = np.corrcoef(log_returns)

    # Find basket pairs: for each stock, find the best basket of correlated stocks
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

                X = train_prices[basket_indices, :].T
                y = train_prices[target, :].T

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

        if best_r2 > 0.85 and best_basket is not None:
            pairs.append({
                'target': target,
                'basket': best_basket,
                'weights': best_weights,
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

    # Trade each pair with adaptive thresholds
    for pair in active_pairs:
        target = pair['target']
        basket = pair['basket']
        weights = pair['weights']

        # Compute spread over full period
        spreads = np.zeros(num_days)
        for day in range(num_days):
            basket_val = sum(weights[i] * prices[basket[i], day] for i in range(len(basket)))
            basket_val += weights[-1]  # intercept
            spreads[day] = prices[target, day] - basket_val

        # Center spread using training mean
        train_spread = spreads[:train_end]
        spread_mean = np.mean(train_spread)
        centered_spreads = spreads - spread_mean

        # Compute rolling standard deviation of the centered spread
        rolling_std = np.zeros(num_days)
        for day in range(num_days):
            start = max(0, day - vol_window + 1)
            window_data = centered_spreads[start:day + 1]
            if len(window_data) >= 2:
                rolling_std[day] = np.std(window_data, ddof=1)
            else:
                rolling_std[day] = np.std(train_spread, ddof=1)  # fallback

        # Trading logic with adaptive thresholds
        position = 0  # +1 = long spread, -1 = short spread, 0 = flat

        for day in range(1, num_days):
            spread = centered_spreads[day]
            vol = rolling_std[day]

            # Adaptive thresholds
            entry_threshold = k_entry * vol
            exit_threshold = k_exit * vol

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

    # Clip actions to respect position limits
    actions = np.clip(actions, -100, 100)

    return actions
