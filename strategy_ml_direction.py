"""
MIG Quant Competition — ML Direction Prediction Strategy
=========================================================

Logistic-regression based directional prediction inspired by Prosperity 3
Macarons approach. Trains one model per stock on the first 70% of data to
predict next-day price direction (up/down). Features are derived purely from
price data: multi-horizon returns, rolling volatility, price vs. rolling mean,
and RSI. Trades only when model confidence exceeds a threshold.

Parameters tuned on dev_data_30 with 0.1% fees and 100-share position limit.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- Tuned parameters ---
TRAIN_FRAC = 0.70
CONFIDENCE_BUY = 0.54       # predict_proba > this to buy
CONFIDENCE_SELL = 0.46      # predict_proba < this to sell
DOLLAR_PER_TRADE = 800      # dollar exposure per trade
POS_LIMIT = 100             # max shares per stock
LOOKBACK = 20               # max lookback for features
FEE_RATE = 0.001            # 0.1% transaction fee


def _compute_rsi(prices_1d, period=14):
    """Compute RSI for a 1D price series."""
    n = len(prices_1d)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi
    deltas = np.diff(prices_1d)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for t in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[t]) / period
        avg_loss = (avg_loss * (period - 1) + losses[t]) / period
        if avg_loss == 0:
            rsi[t + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[t + 1] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def _build_features(price_series):
    """Build feature matrix from a single stock's price series.

    Returns: features (num_days, num_features), labels (num_days,)
    Labels: 1 if next-day return > 0, else 0.
    """
    n = len(price_series)
    # Returns at various horizons
    ret_1 = np.full(n, np.nan)
    ret_5 = np.full(n, np.nan)
    ret_20 = np.full(n, np.nan)
    ret_1[1:] = price_series[1:] / price_series[:-1] - 1
    ret_5[5:] = price_series[5:] / price_series[:-5] - 1
    ret_20[20:] = price_series[20:] / price_series[:-20] - 1

    # Rolling volatility (20-day)
    vol_20 = np.full(n, np.nan)
    for t in range(20, n):
        window = price_series[t-19:t+1]
        log_ret = np.log(window[1:] / window[:-1])
        vol_20[t] = np.std(log_ret)

    # Price relative to 20-day SMA (mean reversion indicator)
    sma_20 = np.full(n, np.nan)
    for t in range(19, n):
        sma_20[t] = np.mean(price_series[t-19:t+1])
    price_vs_sma = (price_series - sma_20) / sma_20

    # Price relative to 5-day SMA (short-term momentum)
    sma_5 = np.full(n, np.nan)
    for t in range(4, n):
        sma_5[t] = np.mean(price_series[t-4:t+1])
    price_vs_sma5 = (price_series - sma_5) / sma_5

    # RSI
    rsi = _compute_rsi(price_series, period=14)
    # Normalize RSI to [-1, 1] range
    rsi_norm = (rsi - 50.0) / 50.0

    # Rolling 5-day volatility
    vol_5 = np.full(n, np.nan)
    for t in range(5, n):
        window = price_series[t-4:t+1]
        log_ret = np.log(window[1:] / window[:-1])
        vol_5[t] = np.std(log_ret)

    # Volatility ratio (short / long) — regime indicator
    vol_ratio = vol_5 / vol_20

    # Stack features
    features = np.column_stack([
        ret_1,
        ret_5,
        ret_20,
        vol_20,
        vol_5,
        vol_ratio,
        price_vs_sma,
        price_vs_sma5,
        rsi_norm,
    ])

    # Labels: next-day return direction
    labels = np.full(n, np.nan)
    labels[:-1] = (price_series[1:] > price_series[:-1]).astype(float)

    return features, labels


def get_actions(prices: np.ndarray,
                confidence_buy: float = CONFIDENCE_BUY,
                confidence_sell: float = CONFIDENCE_SELL,
                dollar_per_trade: float = DOLLAR_PER_TRADE) -> np.ndarray:
    """
    ML direction prediction strategy.

    Parameters
    ----------
    prices : np.ndarray of shape (num_stocks, num_days)
    confidence_buy : float – threshold above which to go long
    confidence_sell : float – threshold below which to go short
    dollar_per_trade : float – dollar amount per trade

    Returns
    -------
    actions : np.ndarray of same shape as prices
    """
    num_stocks, num_days = prices.shape
    split = int(num_days * TRAIN_FRAC)
    actions = np.zeros((num_stocks, num_days))

    models = {}
    scalers = {}
    feature_names = [
        'ret_1d', 'ret_5d', 'ret_20d',
        'vol_20d', 'vol_5d', 'vol_ratio',
        'price_vs_sma20', 'price_vs_sma5', 'rsi_norm',
    ]

    # Train one model per stock
    for i in range(num_stocks):
        features, labels = _build_features(prices[i])

        # Use training window only
        train_feat = features[:split]
        train_labels = labels[:split]

        # Drop rows with NaN
        valid = ~(np.isnan(train_feat).any(axis=1) | np.isnan(train_labels))
        X_train = train_feat[valid]
        y_train = train_labels[valid]

        if len(X_train) < 50 or y_train.sum() < 10 or (len(y_train) - y_train.sum()) < 10:
            continue  # skip stocks with insufficient data

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = LogisticRegression(
            C=0.1,              # strong regularization to avoid overfit
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            random_state=42,
        )
        model.fit(X_train_scaled, y_train)

        models[i] = model
        scalers[i] = scaler

    # Generate actions for test period
    positions = np.zeros(num_stocks)  # track current position

    for day in range(split, num_days):
        for i in range(num_stocks):
            if i not in models:
                continue

            features, _ = _build_features(prices[i, :day+1])
            feat_today = features[-1:]

            if np.isnan(feat_today).any():
                continue

            feat_scaled = scalers[i].transform(feat_today)
            prob_up = models[i].predict_proba(feat_scaled)[0, 1]

            price = prices[i, day]
            max_shares = min(int(dollar_per_trade / price), POS_LIMIT)

            if prob_up > confidence_buy:
                # Want to be long: buy up to max_shares from current position
                desired = max_shares
                trade = desired - positions[i]
                if trade > 0:
                    trade = min(trade, POS_LIMIT - positions[i])
                    actions[i, day] = trade
                    positions[i] += trade
            elif prob_up < confidence_sell:
                # Want to be short: sell/short up to max_shares
                desired = -max_shares
                trade = desired - positions[i]
                if trade < 0:
                    trade = max(trade, -(POS_LIMIT + positions[i]))
                    actions[i, day] = trade
                    positions[i] += trade
            else:
                # Neutral: flatten position
                if positions[i] != 0:
                    actions[i, day] = -positions[i]
                    positions[i] = 0

    return actions


def get_feature_importances(prices: np.ndarray):
    """Train models and return feature importances for analysis."""
    num_stocks, num_days = prices.shape
    split = int(num_days * TRAIN_FRAC)

    feature_names = [
        'ret_1d', 'ret_5d', 'ret_20d',
        'vol_20d', 'vol_5d', 'vol_ratio',
        'price_vs_sma20', 'price_vs_sma5', 'rsi_norm',
    ]

    all_importances = {}
    all_accuracies = {}

    for i in range(num_stocks):
        features, labels = _build_features(prices[i])
        train_feat = features[:split]
        train_labels = labels[:split]
        test_feat = features[split:]
        test_labels = labels[split:]

        valid_train = ~(np.isnan(train_feat).any(axis=1) | np.isnan(train_labels))
        valid_test = ~(np.isnan(test_feat).any(axis=1) | np.isnan(test_labels))

        X_train = train_feat[valid_train]
        y_train = train_labels[valid_train]
        X_test = test_feat[valid_test]
        y_test = test_labels[valid_test]

        if len(X_train) < 50:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs',
                                   max_iter=1000, random_state=42)
        model.fit(X_train_s, y_train)

        train_acc = model.score(X_train_s, y_train)
        test_acc = model.score(X_test_s, y_test)

        all_importances[i] = dict(zip(feature_names, model.coef_[0]))
        all_accuracies[i] = {'train_acc': train_acc, 'test_acc': test_acc}

    return all_importances, all_accuracies, feature_names
