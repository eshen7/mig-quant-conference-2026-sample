import numpy as np
from itertools import combinations


def get_actions(prices, window_size=180, retrain_every=20, k_entry=1.5, k_exit=0.25,
                vol_window=20, max_basket_size=3, hedge_ratio=0.5, max_position=15,
                r2_threshold=0.85, top_corr=6):
    """
    Combined ETF/Basket Statistical Arbitrage with:
    1. Returns-based regression (log returns, not raw prices)
    2. Rolling training window (retrain every R days)
    3. Adaptive thresholds (scaled by rolling spread volatility)
    """
    num_stocks, num_days = prices.shape
    actions = np.zeros((num_stocks, num_days), dtype=np.float64)

    # Precompute log prices for efficiency
    log_prices = np.log(prices + 1e-10)

    # We'll store active pairs and retrain periodically
    active_pairs = []
    last_train_day = -retrain_every  # force initial training

    # Track positions per pair
    pair_positions = {}  # pair_id -> position state

    for day in range(num_days):
        # Check if we need to retrain
        if day - last_train_day >= retrain_every and day >= window_size:
            last_train_day = day
            window_start = day - window_size
            window_end = day

            # Compute cumulative log returns within window (relative to window start)
            # log_ret[i, t] = log(price[i,t]) - log(price[i, window_start])
            log_ret = log_prices[:, window_start:window_end] - log_prices[:, window_start:window_start+1]

            # Compute daily log returns for correlation
            daily_log_ret = np.diff(log_prices[:, window_start:window_end], axis=1)

            # Correlation matrix on daily returns
            corr_matrix = np.corrcoef(daily_log_ret)

            # Find pairs
            new_pairs = []
            for target in range(num_stocks):
                corrs = corr_matrix[target].copy()
                corrs[target] = -1

                top_indices = np.argsort(corrs)[::-1][:top_corr]

                best_r2 = 0
                best_basket = None
                best_weights = None

                for bsize in range(2, max_basket_size + 1):
                    for basket_indices in combinations(top_indices[:top_corr], bsize):
                        basket_indices = list(basket_indices)

                        # OLS on cumulative log returns
                        X = log_ret[basket_indices, :].T  # (window_size, bsize)
                        y = log_ret[target, :].T

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

                if best_r2 > r2_threshold and best_basket is not None:
                    # Compute spread (in return space) over the window
                    X_full = log_ret[best_basket, :].T
                    X_full_c = np.column_stack([X_full, np.ones(len(X_full))])
                    spread_window = log_ret[target, :] - (X_full_c @ best_weights)

                    new_pairs.append({
                        'target': target,
                        'basket': best_basket,
                        'weights': best_weights,
                        'r2': best_r2,
                        'spread_std': np.std(spread_window[-vol_window:]) if len(spread_window) >= vol_window else np.std(spread_window),
                    })

            # Sort by R2, pick top non-overlapping
            new_pairs.sort(key=lambda x: x['r2'], reverse=True)
            used_stocks = set()
            active_pairs = []
            for p in new_pairs:
                if p['target'] not in used_stocks:
                    active_pairs.append(p)
                    used_stocks.add(p['target'])
                if len(active_pairs) >= 10:
                    break

            # Initialize positions for new pairs if not tracked
            for i, p in enumerate(active_pairs):
                key = (p['target'], tuple(p['basket']))
                if key not in pair_positions:
                    pair_positions[key] = 0

        # Trading logic
        if day < window_size:
            continue

        for pair in active_pairs:
            target = pair['target']
            basket = pair['basket']
            weights = pair['weights']
            key = (target, tuple(basket))

            # Compute current spread in return space
            # Use log returns relative to a recent anchor (start of current window)
            anchor = max(0, last_train_day - window_size)
            cur_log_ret_target = log_prices[target, day] - log_prices[target, anchor]
            cur_log_ret_basket = np.array([log_prices[b, day] - log_prices[b, anchor] for b in basket])
            basket_val = np.dot(weights[:-1], cur_log_ret_basket) + weights[-1]
            spread = cur_log_ret_target - basket_val

            # Adaptive threshold using spread_std
            spread_std = pair['spread_std']
            if spread_std < 1e-8:
                spread_std = 1e-4

            entry_thresh = k_entry * spread_std
            exit_thresh = k_exit * spread_std

            position = pair_positions.get(key, 0)

            if position == 0:
                if spread > entry_thresh:
                    # Short spread
                    position = -1
                    actions[target, day] -= max_position
                    for i, stock_idx in enumerate(basket):
                        basket_shares = int(max_position * abs(weights[i]) * hedge_ratio)
                        basket_shares = max(1, min(basket_shares, max_position))
                        if weights[i] > 0:
                            actions[stock_idx, day] += basket_shares
                        else:
                            actions[stock_idx, day] -= basket_shares

                elif spread < -entry_thresh:
                    # Long spread
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
                if spread > exit_thresh:
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
                if spread < -exit_thresh:
                    position = 0
                    actions[target, day] += max_position
                    for i, stock_idx in enumerate(basket):
                        basket_shares = int(max_position * abs(weights[i]) * hedge_ratio)
                        basket_shares = max(1, min(basket_shares, max_position))
                        if weights[i] > 0:
                            actions[stock_idx, day] -= basket_shares
                        else:
                            actions[stock_idx, day] += basket_shares

            pair_positions[key] = position

    actions = np.clip(actions, -100, 100)
    return actions
