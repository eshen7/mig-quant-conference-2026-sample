import numpy as np
from itertools import combinations


def get_actions(prices, window_size=120, retrain_freq=10, entry_threshold=1.0,
                exit_threshold=0.0, max_basket_size=3, hedge_ratio=0.5,
                max_position=15, r2_cutoff=0.85, top_k_corr=6):
    """
    ETF/Basket Statistical Arbitrage with Rolling Training Window.

    Instead of training once on a fixed fraction, retrains OLS weights every
    `retrain_freq` days using the last `window_size` days of data.

    Args:
        prices: np.ndarray of shape (num_stocks, num_days) with Open prices
        window_size: number of days in the rolling training window
        retrain_freq: retrain every N days
        entry_threshold: fixed dollar spread threshold to enter trades
        exit_threshold: spread level to exit trades
        max_basket_size: max number of stocks in a basket (2 or 3)
        hedge_ratio: fraction of basket to trade as hedge
        max_position: max shares per stock per trade signal
        r2_cutoff: minimum R² to qualify a pair
        top_k_corr: number of top correlated stocks to consider
    """
    num_stocks, num_days = prices.shape
    actions = np.zeros((num_stocks, num_days), dtype=np.float64)

    # We start trading after the first window completes
    trade_start = window_size

    # Precompute log prices for correlation
    log_prices = np.log(prices + 1e-10)

    # Track positions per pair: dict of target -> position state
    positions = {}  # target -> int (-1, 0, 1)

    # Current active pairs
    active_pairs = []
    last_train_day = -retrain_freq  # Force training on first eligible day

    for day in range(trade_start, num_days):
        # Check if we need to retrain
        if day - last_train_day >= retrain_freq:
            last_train_day = day

            # Training window
            t_start = day - window_size
            t_end = day
            train_prices = prices[:, t_start:t_end]

            # Compute log returns for correlation
            log_returns = np.diff(np.log(train_prices + 1e-10), axis=1)
            corr_matrix = np.corrcoef(log_returns)

            # Find pairs
            pairs = []
            for target in range(num_stocks):
                corrs = corr_matrix[target].copy()
                corrs[target] = -1

                top_indices = np.argsort(corrs)[::-1][:top_k_corr]

                best_r2 = 0
                best_basket = None
                best_weights = None

                for bsize in range(2, max_basket_size + 1):
                    for basket_indices in combinations(top_indices, bsize):
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

                if best_r2 > r2_cutoff and best_basket is not None:
                    # Compute spread stats on training window
                    spreads_train = np.zeros(window_size)
                    for d in range(window_size):
                        basket_val = sum(best_weights[i] * train_prices[best_basket[i], d]
                                         for i in range(len(best_basket)))
                        basket_val += best_weights[-1]
                        spreads_train[d] = train_prices[target, d] - basket_val

                    spread_mean = np.mean(spreads_train)
                    spread_std = np.std(spreads_train)

                    pairs.append({
                        'target': target,
                        'basket': best_basket,
                        'weights': best_weights,
                        'r2': best_r2,
                        'spread_mean': spread_mean,
                        'spread_std': spread_std,
                    })

            # Sort by R² and select non-overlapping targets
            pairs.sort(key=lambda x: x['r2'], reverse=True)
            used_stocks = set()
            new_active = []
            for p in pairs:
                if p['target'] not in used_stocks:
                    new_active.append(p)
                    used_stocks.add(p['target'])
                if len(new_active) >= 10:
                    break

            # Close positions for pairs that are no longer active
            old_targets = {p['target'] for p in active_pairs}
            new_targets = {p['target'] for p in new_active}
            for old_pair in active_pairs:
                t = old_pair['target']
                if t not in new_targets and positions.get(t, 0) != 0:
                    # Close this position
                    pos = positions[t]
                    basket = old_pair['basket']
                    weights = old_pair['weights']
                    if pos == 1:
                        # Was long spread: sell target, buy basket
                        actions[t, day] -= max_position
                        for i, stock_idx in enumerate(basket):
                            bs = int(max_position * abs(weights[i]) * hedge_ratio)
                            bs = max(1, min(bs, max_position))
                            if weights[i] > 0:
                                actions[stock_idx, day] += bs
                            else:
                                actions[stock_idx, day] -= bs
                    elif pos == -1:
                        # Was short spread: buy target, sell basket
                        actions[t, day] += max_position
                        for i, stock_idx in enumerate(basket):
                            bs = int(max_position * abs(weights[i]) * hedge_ratio)
                            bs = max(1, min(bs, max_position))
                            if weights[i] > 0:
                                actions[stock_idx, day] -= bs
                            else:
                                actions[stock_idx, day] += bs
                    positions[t] = 0

            active_pairs = new_active

        # Trade using current active pairs
        for pair in active_pairs:
            target = pair['target']
            basket = pair['basket']
            weights = pair['weights']
            spread_mean = pair['spread_mean']

            # Compute current spread
            basket_val = sum(weights[i] * prices[basket[i], day]
                             for i in range(len(basket)))
            basket_val += weights[-1]
            spread = prices[target, day] - basket_val - spread_mean

            pos = positions.get(target, 0)

            if pos == 0:
                if spread > entry_threshold:
                    positions[target] = -1
                    actions[target, day] -= max_position
                    for i, stock_idx in enumerate(basket):
                        bs = int(max_position * abs(weights[i]) * hedge_ratio)
                        bs = max(1, min(bs, max_position))
                        if weights[i] > 0:
                            actions[stock_idx, day] += bs
                        else:
                            actions[stock_idx, day] -= bs

                elif spread < -entry_threshold:
                    positions[target] = 1
                    actions[target, day] += max_position
                    for i, stock_idx in enumerate(basket):
                        bs = int(max_position * abs(weights[i]) * hedge_ratio)
                        bs = max(1, min(bs, max_position))
                        if weights[i] > 0:
                            actions[stock_idx, day] -= bs
                        else:
                            actions[stock_idx, day] += bs

            elif pos == 1:
                if spread > exit_threshold:
                    positions[target] = 0
                    actions[target, day] -= max_position
                    for i, stock_idx in enumerate(basket):
                        bs = int(max_position * abs(weights[i]) * hedge_ratio)
                        bs = max(1, min(bs, max_position))
                        if weights[i] > 0:
                            actions[stock_idx, day] += bs
                        else:
                            actions[stock_idx, day] -= bs

            elif pos == -1:
                if spread < exit_threshold:
                    positions[target] = 0
                    actions[target, day] += max_position
                    for i, stock_idx in enumerate(basket):
                        bs = int(max_position * abs(weights[i]) * hedge_ratio)
                        bs = max(1, min(bs, max_position))
                        if weights[i] > 0:
                            actions[stock_idx, day] -= bs
                        else:
                            actions[stock_idx, day] += bs

    actions = np.clip(actions, -100, 100)
    return actions
