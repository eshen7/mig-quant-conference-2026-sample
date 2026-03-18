import numpy as np


def get_actions(prices, ema_span=12, threshold=0.35, dollar_per_trade=400,
                autocorr_cutoff=-0.02, train_frac=0.70):
    """
    Mean Reversion with EMA strategy.

    Parameters
    ----------
    prices : np.ndarray, shape (num_stocks, num_days)
        Open prices for each stock on each day.
    ema_span : int
        EMA span (number of days) for the fast EMA.
    threshold : float
        Fixed dollar deviation from EMA required to trigger a trade.
    dollar_per_trade : float
        Dollar amount to allocate per trade signal.
    autocorr_cutoff : float
        Maximum lag-1 autocorrelation of returns to qualify as mean-reverting.
    train_frac : float
        Fraction of data used for training (identifying mean-reverting stocks).

    Returns
    -------
    actions : np.ndarray, shape (num_stocks, num_days)
    """
    num_stocks, num_days = prices.shape
    actions = np.zeros_like(prices, dtype=int)

    train_end = int(num_days * train_frac)

    # --- Step 1: Identify mean-reverting stocks using training data ---
    mean_reverting = np.zeros(num_stocks, dtype=bool)
    for i in range(num_stocks):
        train_prices = prices[i, :train_end]
        returns = np.diff(train_prices) / train_prices[:-1]
        returns = returns[np.isfinite(returns)]
        if len(returns) < 20:
            continue
        # Lag-1 autocorrelation
        r_mean = returns.mean()
        r_var = np.var(returns)
        if r_var == 0:
            continue
        autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        if autocorr < autocorr_cutoff:
            mean_reverting[i] = True

    # --- Step 2: Compute EMA and generate signals for ALL days ---
    alpha = 2.0 / (ema_span + 1)
    positions = np.zeros(num_stocks, dtype=int)

    for i in range(num_stocks):
        if not mean_reverting[i]:
            continue

        # Initialize EMA with first price
        ema = prices[i, 0]

        for d in range(num_days):
            price = prices[i, d]
            ema = alpha * price + (1 - alpha) * ema
            deviation = price - ema

            if d == 0:
                continue

            # Number of shares based on dollar allocation
            shares = int(dollar_per_trade / price) if price > 0 else 0
            shares = min(shares, 100)  # cap at position limit

            if deviation < -threshold and positions[i] < 100:
                # Price below EMA -> buy (mean reversion up)
                buy_qty = min(shares, 100 - positions[i])
                if buy_qty > 0:
                    actions[i, d] = buy_qty
                    positions[i] += buy_qty

            elif deviation > threshold and positions[i] > -100:
                # Price above EMA -> sell/short (mean reversion down)
                sell_qty = min(shares, 100 + positions[i])
                if sell_qty > 0:
                    actions[i, d] = -sell_qty
                    positions[i] -= sell_qty

            elif abs(deviation) < threshold * 0.3:
                # Price reverted to EMA -> flatten position
                if positions[i] > 0:
                    actions[i, d] = -positions[i]
                    positions[i] = 0
                elif positions[i] < 0:
                    actions[i, d] = -positions[i]  # positive = cover
                    positions[i] = 0

    return actions
