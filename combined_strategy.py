"""
Combined Multi-Strategy (Self-Contained, No Look-Ahead)
========================================================

Three structurally different alpha sources with priority-based position
allocation, all in one file with no external imports beyond numpy.

1. ETF Stat Arb — hardcoded basket weights trained on dev_data (no future look)
2. Cross-sectional Momentum — buy winners, sell losers (no training needed)
3. Short-term Reversal — trade cross-sectional reversals (no training needed)

All strategies use only past prices to make current decisions.
Position limits enforced: max 100 shares per stock.
"""

import numpy as np


# ==========================================================================
# HARDCODED ETF ARB PAIRS (trained on dev_data_30.csv, first 85% of days)
# ==========================================================================

# Returns-based pairs: spread = log_return[target] - (w0*log_return[b0] + ... + intercept)
# Trained on dev_data_30.csv (first 85% = 855 days)
RETURNS_PAIRS = [
    {"target": 11, "basket": [21, 20, 9, 26], "weights": [0.537806204, 0.2428688445, 0.2306298356, 0.214035629, -0.0592758885], "spread_mean": 0.0, "spread_std": 0.0328824048},
    {"target": 29, "basket": [1, 23, 24, 3], "weights": [0.538481775, 0.1535130369, 0.4544151957, -0.2206271069, -0.0008206919], "spread_mean": 0.0, "spread_std": 0.0350049269},
    {"target": 1, "basket": [29, 25, 17, 3], "weights": [0.7300417695, 0.4573108715, -0.2062956887, 0.189908529, 0.1072925616], "spread_mean": 0.0, "spread_std": 0.0412931386},
    {"target": 13, "basket": [10, 15, 22, 21], "weights": [0.6980578761, 0.3347392963, -0.3768864197, 0.3719270172, -0.0208704679], "spread_mean": 0.0, "spread_std": 0.0493938816},
    {"target": 20, "basket": [19, 11, 21, 22], "weights": [0.8924802239, 1.2762364923, -1.5182523462, 0.6799656723, -0.0449951269], "spread_mean": 0.0, "spread_std": 0.0894626183},
    {"target": 14, "basket": [15, 4, 24, 28], "weights": [0.2646058902, 0.6748267161, 0.3281501514, -0.2346664781, -0.1023587158], "spread_mean": 0.0, "spread_std": 0.0618500374},
    {"target": 24, "basket": [22, 21, 3, 14], "weights": [-0.3918788745, 0.442793546, 0.4315225099, 0.3854884166, -0.0941317091], "spread_mean": 0.0, "spread_std": 0.0491266943},
    {"target": 19, "basket": [20, 22, 11, 28], "weights": [0.3535234274, -0.5871373256, 0.6165058279, -0.1178918497, 0.0283081521], "spread_mean": 0.0, "spread_std": 0.0536997916},
    {"target": 10, "basket": [13, 15, 4, 26], "weights": [0.6872529466, -0.2014788675, 0.5305105846, -0.1901801449, -0.0415481796], "spread_mean": 0.0, "spread_std": 0.0614058266},
    {"target": 7, "basket": [27, 9, 20, 28], "weights": [0.3382884514, 0.7742051056, 0.7742510499, -0.8199014422, 0.0098290289], "spread_mean": 0.0, "spread_std": 0.1291654332},
]

# Adaptive pairs: spread = price[target] - (w0*price[b0] + ... + intercept)
# Trained on dev_data_30.csv (first 85% = 855 days)
ADAPTIVE_PAIRS = [
    {"target": 13, "basket": [10, 15, 22, 21], "weights": [1.2544701939, 13.459756086, -0.3073198127, 0.5079402051, -2.0887010513], "spread_mean": 0.0, "spread_std": 9.8956582641},
    {"target": 29, "basket": [1, 23, 24, 3], "weights": [0.136590228, 0.0536512171, 0.3066332784, -0.299333816, 2.716449188], "spread_mean": 0.0, "spread_std": 1.797865678},
    {"target": 11, "basket": [21, 20, 9, 26], "weights": [0.1697162845, 0.1716043203, 0.5064031941, 0.0837946709, -11.1516208879], "spread_mean": 0.0, "spread_std": 1.1642800931},
    {"target": 1, "basket": [29, 25, 17, 3], "weights": [2.7990663944, 0.7790748677, -1.2927183703, 0.8835064304, -23.3208889936], "spread_mean": 0.0, "spread_std": 8.6293866339},
    {"target": 14, "basket": [15, 4, 24, 28], "weights": [3.4919522143, 0.8244134101, 0.190538651, -1.1480155836, -1.78205559], "spread_mean": 0.0, "spread_std": 3.1560527011},
    {"target": 10, "basket": [13, 15, 26, 22], "weights": [0.6421528904, -6.0100123438, -0.2058542237, 0.1570923291, 12.4995618485], "spread_mean": 0.0, "spread_std": 7.0938348693},
    {"target": 20, "basket": [19, 11, 21, 27], "weights": [0.522334964, 2.5801878001, -0.5597266504, -0.0701528801, 0.3925712919], "spread_mean": 0.0, "spread_std": 4.3229636427},
    {"target": 24, "basket": [22, 21, 3, 14], "weights": [-0.1539634426, 0.387283019, 0.6340598747, 0.5136915469, 14.4527162587], "spread_mean": 0.0, "spread_std": 3.5019821916},
    {"target": 4, "basket": [14, 10, 26, 24], "weights": [0.3570996655, 0.1214184699, 0.0920976614, -0.115184626, 6.4932569447], "spread_mean": 0.0, "spread_std": 2.319010442},
    {"target": 25, "basket": [29, 3, 2, 26], "weights": [1.104984772, -0.2592399834, 1.835130402, -0.2385338716, -2.8879183936], "spread_mean": 0.0, "spread_std": 5.0019589501},
]


# ==========================================================================
# STRATEGY 1: ETF Stat Arb (hardcoded weights, no look-ahead)
# ==========================================================================

def _etf_arb_actions(prices):
    """
    ETF stat arb using pre-trained basket weights.

    Returns-based sub-strategy: uses log returns and hardcoded OLS weights.
    Adaptive sub-strategy: uses raw prices and hardcoded OLS weights.

    No future data is used — spreads are computed causally day-by-day,
    and the rolling volatility lookback only uses past data.
    """
    num_stocks, num_days = prices.shape

    # --- Shared params ---
    ENTRY_THRESH_RET = 0.02
    EXIT_THRESH_RET = 0.008
    K_ENTRY_ADP = 1.0
    K_EXIT_ADP = 0.2
    VOL_WINDOW_ADP = 40
    MAX_POSITION = 20
    HEDGE_RATIO = 0.5
    VOL_LOOKBACK = 20
    PAIR_PNL_LIMIT = -500.0
    PORTFOLIO_DD_LIMIT = 2000.0

    # Compute log returns from day 0 (used by returns-based pairs)
    log_returns = np.log(prices / prices[:, 0:1])

    all_pair_actions = []

    # --- Returns-based pairs ---
    for pair in RETURNS_PAIRS:
        target = pair["target"]
        basket = pair["basket"]
        weights = pair["weights"]
        spread_mean = pair["spread_mean"]

        pair_actions = np.zeros((num_stocks, num_days), dtype=np.float64)

        # Compute spread using only past/current data (log returns from day 0)
        spreads = np.zeros(num_days)
        for day in range(num_days):
            basket_val = sum(weights[i] * log_returns[basket[i], day] for i in range(len(basket)))
            basket_val += weights[-1]  # intercept
            spreads[day] = log_returns[target, day] - basket_val

        centered_spreads = spreads - spread_mean

        # Rolling vol for position sizing (only uses past data)
        baseline_vol = pair["spread_std"]
        target_risk = baseline_vol * MAX_POSITION

        rolling_vol = np.full(num_days, baseline_vol)
        for day in range(1, num_days):
            start = max(0, day - VOL_LOOKBACK + 1)
            window = centered_spreads[start:day + 1]
            if len(window) >= 2:
                rolling_vol[day] = np.std(window, ddof=1)

        position = 0
        for day in range(1, num_days):
            spread = centered_spreads[day]
            rv = rolling_vol[day] if rolling_vol[day] > 1e-10 else baseline_vol
            pos_size = min(MAX_POSITION, max(1, int(target_risk / rv)))

            if position == 0:
                if spread > ENTRY_THRESH_RET:
                    position = -1
                    pair_actions[target, day] -= pos_size
                    for i, si in enumerate(basket):
                        bs = max(1, min(pos_size, int(pos_size * abs(weights[i]) * HEDGE_RATIO)))
                        if weights[i] > 0:
                            pair_actions[si, day] += bs
                        else:
                            pair_actions[si, day] -= bs
                elif spread < -ENTRY_THRESH_RET:
                    position = 1
                    pair_actions[target, day] += pos_size
                    for i, si in enumerate(basket):
                        bs = max(1, min(pos_size, int(pos_size * abs(weights[i]) * HEDGE_RATIO)))
                        if weights[i] > 0:
                            pair_actions[si, day] -= bs
                        else:
                            pair_actions[si, day] += bs
            elif position == 1:
                if spread > EXIT_THRESH_RET:
                    position = 0
                    pair_actions[target, day] -= pos_size
                    for i, si in enumerate(basket):
                        bs = max(1, min(pos_size, int(pos_size * abs(weights[i]) * HEDGE_RATIO)))
                        if weights[i] > 0:
                            pair_actions[si, day] += bs
                        else:
                            pair_actions[si, day] -= bs
            elif position == -1:
                if spread < EXIT_THRESH_RET:
                    position = 0
                    pair_actions[target, day] += pos_size
                    for i, si in enumerate(basket):
                        bs = max(1, min(pos_size, int(pos_size * abs(weights[i]) * HEDGE_RATIO)))
                        if weights[i] > 0:
                            pair_actions[si, day] -= bs
                        else:
                            pair_actions[si, day] += bs

        all_pair_actions.append(np.clip(pair_actions, -100, 100))

    # --- Adaptive pairs ---
    for pair in ADAPTIVE_PAIRS:
        target = pair["target"]
        basket = pair["basket"]
        weights = pair["weights"]
        spread_mean = pair["spread_mean"]

        pair_actions = np.zeros((num_stocks, num_days), dtype=np.float64)

        # Spread using raw prices
        spreads = np.zeros(num_days)
        for day in range(num_days):
            basket_val = sum(weights[i] * prices[basket[i], day] for i in range(len(basket)))
            basket_val += weights[-1]
            spreads[day] = prices[target, day] - basket_val

        centered_spreads = spreads - spread_mean
        baseline_vol = pair["spread_std"]
        target_risk = baseline_vol * MAX_POSITION

        # Rolling std for adaptive thresholds
        rolling_std = np.full(num_days, baseline_vol)
        for day in range(1, num_days):
            start = max(0, day - VOL_WINDOW_ADP + 1)
            window = centered_spreads[start:day + 1]
            if len(window) >= 2:
                rolling_std[day] = np.std(window, ddof=1)

        # Rolling vol for position sizing
        rolling_vol = np.full(num_days, baseline_vol)
        for day in range(1, num_days):
            start = max(0, day - VOL_LOOKBACK + 1)
            window = centered_spreads[start:day + 1]
            if len(window) >= 2:
                rolling_vol[day] = np.std(window, ddof=1)

        position = 0
        for day in range(1, num_days):
            spread = centered_spreads[day]
            vol = rolling_std[day]
            entry_t = K_ENTRY_ADP * vol
            exit_t = K_EXIT_ADP * vol

            rv = rolling_vol[day] if rolling_vol[day] > 1e-10 else baseline_vol
            pos_size = min(MAX_POSITION, max(1, int(target_risk / rv)))

            if position == 0:
                if spread > entry_t:
                    position = -1
                    pair_actions[target, day] -= pos_size
                    for i, si in enumerate(basket):
                        bs = max(1, min(pos_size, int(pos_size * abs(weights[i]) * HEDGE_RATIO)))
                        if weights[i] > 0:
                            pair_actions[si, day] += bs
                        else:
                            pair_actions[si, day] -= bs
                elif spread < -entry_t:
                    position = 1
                    pair_actions[target, day] += pos_size
                    for i, si in enumerate(basket):
                        bs = max(1, min(pos_size, int(pos_size * abs(weights[i]) * HEDGE_RATIO)))
                        if weights[i] > 0:
                            pair_actions[si, day] -= bs
                        else:
                            pair_actions[si, day] += bs
            elif position == 1:
                if spread > exit_t:
                    position = 0
                    pair_actions[target, day] -= pos_size
                    for i, si in enumerate(basket):
                        bs = max(1, min(pos_size, int(pos_size * abs(weights[i]) * HEDGE_RATIO)))
                        if weights[i] > 0:
                            pair_actions[si, day] += bs
                        else:
                            pair_actions[si, day] -= bs
            elif position == -1:
                if spread < -exit_t:
                    position = 0
                    pair_actions[target, day] += pos_size
                    for i, si in enumerate(basket):
                        bs = max(1, min(pos_size, int(pos_size * abs(weights[i]) * HEDGE_RATIO)))
                        if weights[i] > 0:
                            pair_actions[si, day] -= bs
                        else:
                            pair_actions[si, day] += bs

        all_pair_actions.append(np.clip(pair_actions, -100, 100))

    # --- Circuit breakers ---
    # Per-pair kill switch
    for pidx in range(len(all_pair_actions)):
        pa = all_pair_actions[pidx]
        positions = np.zeros(num_stocks)
        cum_cash = 0.0
        killed = False

        for day in range(num_days):
            if killed:
                pa[:, day:] = 0
                break
            for s in range(num_stocks):
                a = pa[s, day]
                if a != 0:
                    cum_cash -= a * prices[s, day]
                    positions[s] += a
            mtm = sum(positions[s] * prices[s, day] for s in range(num_stocks))
            if cum_cash + mtm < PAIR_PNL_LIMIT:
                killed = True
                pa[:, day + 1:] = 0

    # Combine all pairs
    combined = np.zeros((num_stocks, num_days), dtype=np.float64)
    for pa in all_pair_actions:
        combined += pa
    combined = np.clip(combined, -100, 100)

    # Portfolio-level drawdown protection
    positions = np.zeros(num_stocks)
    cash = 25000.0
    peak_value = 25000.0
    in_drawdown = False
    final = np.zeros((num_stocks, num_days), dtype=np.float64)

    for day in range(num_days):
        day_act = combined[:, day].copy()
        if in_drawdown:
            for s in range(num_stocks):
                if day_act[s] == 0:
                    continue
                new_pos = positions[s] + day_act[s]
                if abs(new_pos) >= abs(positions[s]):
                    day_act[s] = 0  # block increases

        final[:, day] = day_act
        for s in range(num_stocks):
            if day_act[s] != 0:
                cash -= day_act[s] * prices[s, day]
                positions[s] += day_act[s]
        port_val = cash + sum(positions[s] * prices[s, day] for s in range(num_stocks))
        if port_val > peak_value:
            peak_value = port_val
        in_drawdown = (peak_value - port_val) > PORTFOLIO_DD_LIMIT

    return final


# ==========================================================================
# STRATEGY 2: Cross-Sectional Momentum (no training needed)
# ==========================================================================

def _momentum_actions(prices):
    """
    Rank stocks by past 60-day returns (skip last 5 days).
    Long top 6, short bottom 6. Rebalance every 5 days.
    Uses only past prices — no look-ahead.
    """
    num_stocks, num_days = prices.shape
    actions = np.zeros((num_stocks, num_days), dtype=np.float64)

    LOOKBACK = 60
    SKIP = 5
    REBAL_FREQ = 5
    LONG_K = 6
    SHORT_K = 6
    SHARES = 6

    start_day = LOOKBACK + SKIP
    positions = np.zeros(num_stocks)

    for day in range(start_day, num_days):
        if (day - start_day) % REBAL_FREQ != 0:
            continue

        past_price = prices[:, day - LOOKBACK - SKIP]
        recent_price = prices[:, day - SKIP]

        valid = past_price > 0
        momentum = np.full(num_stocks, -np.inf)
        momentum[valid] = (recent_price[valid] - past_price[valid]) / past_price[valid]

        ranked = np.argsort(momentum)
        target = np.zeros(num_stocks)
        target[ranked[-LONG_K:]] = SHARES
        target[ranked[:SHORT_K]] = -SHARES

        delta = target - positions
        actions[:, day] = delta
        positions = target.copy()

    return actions


# ==========================================================================
# STRATEGY 3: Short-Term Cross-Sectional Reversal (no training needed)
# ==========================================================================

def _reversal_actions(prices):
    """
    Buy 3-day losers, short 3-day winners. Hold 2 days.
    Only trades when cross-sectional dispersion > 0.5%.
    Uses only past prices — no look-ahead.
    """
    num_stocks, num_days = prices.shape
    actions = np.zeros((num_stocks, num_days), dtype=np.float64)

    LOOKBACK = 3
    HOLD = 2
    N_LONG = 5
    N_SHORT = 5
    MAX_SH = 6
    DISP_THRESH = 0.005
    WARMUP = 20

    open_pos = []  # (open_day, stock_idx, shares)

    for day in range(WARMUP, num_days):
        # Close expired positions
        still_open = []
        for (od, si, sh) in open_pos:
            if day - od >= HOLD:
                actions[si, day] -= sh
            else:
                still_open.append((od, si, sh))
        open_pos = still_open

        if day < LOOKBACK:
            continue

        rets = np.zeros(num_stocks)
        for s in range(num_stocks):
            if prices[s, day - LOOKBACK] > 0:
                rets[s] = (prices[s, day] - prices[s, day - LOOKBACK]) / prices[s, day - LOOKBACK]

        if np.std(rets) < DISP_THRESH:
            continue

        ranked = np.argsort(rets)
        longs = ranked[:N_LONG]
        shorts = ranked[-N_SHORT:]

        long_sh = np.full(N_LONG, MAX_SH, dtype=int)
        short_sh = np.full(N_SHORT, MAX_SH, dtype=int)

        # Dollar-neutral adjustment
        ld = np.sum(long_sh * prices[longs, day])
        sd = np.sum(short_sh * prices[shorts, day])
        if ld > 0 and sd > 0:
            if sd / ld > 1.2:
                short_sh = np.maximum(1, (short_sh * ld / sd).astype(int))
            elif sd / ld < 0.8:
                long_sh = np.maximum(1, (long_sh * sd / ld).astype(int))

        for i, s in enumerate(longs):
            actions[s, day] += long_sh[i]
            open_pos.append((day, s, int(long_sh[i])))
        for i, s in enumerate(shorts):
            actions[s, day] -= short_sh[i]
            open_pos.append((day, s, -int(short_sh[i])))

    return actions


# ==========================================================================
# COMBINED: Priority-based position allocation
# ==========================================================================

def get_actions(prices: np.ndarray) -> np.ndarray:
    """
    Combined multi-strategy with priority-based position limit enforcement.

    ETF arb (primary) gets full allocation first.
    Momentum and reversal fill remaining capacity.
    Max 100 shares per stock enforced.
    """
    num_stocks, num_days = prices.shape
    POS_LIMIT = 100

    # Generate actions from each strategy (ordered by priority)
    strategy_actions = [
        _etf_arb_actions(prices),
        _momentum_actions(prices),
        _reversal_actions(prices),
    ]

    final_actions = np.zeros((num_stocks, num_days), dtype=np.float64)
    positions = np.zeros(num_stocks)

    for day in range(num_days):
        day_combined = np.zeros(num_stocks)

        for acts in strategy_actions:
            act = acts[:, day].copy()
            for s in range(num_stocks):
                if act[s] == 0:
                    continue
                new_pos = positions[s] + day_combined[s] + act[s]
                if abs(new_pos) > POS_LIMIT:
                    if act[s] > 0:
                        room = POS_LIMIT - (positions[s] + day_combined[s])
                        act[s] = max(0, min(act[s], room))
                    else:
                        room = -POS_LIMIT - (positions[s] + day_combined[s])
                        act[s] = min(0, max(act[s], room))
            day_combined += act

        day_combined = np.clip(day_combined, -POS_LIMIT, POS_LIMIT)
        final_actions[:, day] = day_combined
        positions += day_combined

    return final_actions
