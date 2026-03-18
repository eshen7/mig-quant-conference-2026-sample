import numpy as np
from itertools import combinations


def get_actions(prices, entry_threshold=1.5, exit_threshold=0.0, train_frac=0.70,
                max_basket_size=4, hedge_ratio=0.5, max_position=15):
    """
    ETF/Basket Statistical Arbitrage with Fixed Thresholds.

    Inspired by Prosperity 3 Picnic Baskets approach:
    - Find groups of correlated stocks forming synthetic baskets
    - Trade spread between a stock and its synthetic basket value
    - Use FIXED thresholds (not z-score normalization)
    - Half-hedge basket exposure

    Args:
        prices: np.ndarray of shape (num_stocks, num_days) with Open prices
        entry_threshold: fixed dollar spread threshold to enter trades
        exit_threshold: spread level to exit trades (typically 0 or small)
        train_frac: fraction of data used for training
        max_basket_size: max number of stocks in a basket
        hedge_ratio: fraction of basket to trade as hedge (0.5 = half-hedge)
        max_position: max shares per stock per trade signal
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
    # that can be used to replicate its price movements
    pairs = []

    for target in range(num_stocks):
        # Find most correlated stocks
        corrs = corr_matrix[target].copy()
        corrs[target] = -1  # exclude self

        # Get top correlated stocks
        top_indices = np.argsort(corrs)[::-1][:max_basket_size * 2]

        best_r2 = 0
        best_basket = None
        best_weights = None

        # Try different basket sizes
        for bsize in range(2, max_basket_size + 1):
            for basket_indices in combinations(top_indices[:max_basket_size * 2], bsize):
                basket_indices = list(basket_indices)

                # OLS regression: target_price ~ basket_prices
                X = train_prices[basket_indices, :].T  # (train_days, bsize)
                y = train_prices[target, :].T  # (train_days,)

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

        if best_r2 > 0.85 and best_basket is not None:
            pairs.append({
                'target': target,
                'basket': best_basket,
                'weights': best_weights,  # includes intercept as last element
                'r2': best_r2,
            })

    # Sort by R2 and take top pairs (avoid overlapping targets)
    pairs.sort(key=lambda x: x['r2'], reverse=True)

    # Compute spreads and trade
    used_stocks = set()
    active_pairs = []
    for p in pairs:
        if p['target'] not in used_stocks:
            active_pairs.append(p)
            used_stocks.add(p['target'])
        if len(active_pairs) >= 10:
            break

    # Track positions for each pair
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

        # Compute training spread statistics for threshold calibration
        train_spread = spreads[:train_end]
        spread_mean = np.mean(train_spread)
        spread_std = np.std(train_spread)

        # The spread should be mean-reverting; center it
        centered_spreads = spreads - spread_mean

        # Trading logic with fixed thresholds
        position = 0  # +1 = long spread, -1 = short spread, 0 = flat

        for day in range(1, num_days):
            spread = centered_spreads[day]

            if position == 0:
                # Entry signals
                if spread > entry_threshold:
                    # Spread too high -> short spread (sell target, buy basket)
                    position = -1
                    # Sell target stock
                    actions[target, day] -= max_position
                    # Buy basket stocks (half-hedge)
                    for i, stock_idx in enumerate(basket):
                        # Weight determines how many shares to buy
                        basket_shares = int(max_position * abs(weights[i]) * hedge_ratio)
                        basket_shares = max(1, min(basket_shares, max_position))
                        if weights[i] > 0:
                            actions[stock_idx, day] += basket_shares
                        else:
                            actions[stock_idx, day] -= basket_shares

                elif spread < -entry_threshold:
                    # Spread too low -> long spread (buy target, sell basket)
                    position = 1
                    # Buy target stock
                    actions[target, day] += max_position
                    # Sell basket stocks (half-hedge)
                    for i, stock_idx in enumerate(basket):
                        basket_shares = int(max_position * abs(weights[i]) * hedge_ratio)
                        basket_shares = max(1, min(basket_shares, max_position))
                        if weights[i] > 0:
                            actions[stock_idx, day] -= basket_shares
                        else:
                            actions[stock_idx, day] += basket_shares

            elif position == 1:
                # Long spread position - exit when spread crosses above exit_threshold
                if spread > exit_threshold:
                    position = 0
                    # Close: sell target
                    actions[target, day] -= max_position
                    # Close: buy back basket
                    for i, stock_idx in enumerate(basket):
                        basket_shares = int(max_position * abs(weights[i]) * hedge_ratio)
                        basket_shares = max(1, min(basket_shares, max_position))
                        if weights[i] > 0:
                            actions[stock_idx, day] += basket_shares
                        else:
                            actions[stock_idx, day] -= basket_shares

            elif position == -1:
                # Short spread position - exit when spread crosses below exit_threshold
                if spread < exit_threshold:
                    position = 0
                    # Close: buy back target
                    actions[target, day] += max_position
                    # Close: sell basket
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
