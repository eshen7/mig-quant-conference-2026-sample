import numpy as np
import pandas as pd
from itertools import product
from backtester import Backtester
from strategy_etf_arb_returns import get_actions

# Load data
df = pd.read_csv('dev_data_30.csv', parse_dates=['Date'], date_format='%m/%d/%y', thousands=',')
prices_df = df.pivot(index='Ticker', columns='Date', values='Open').sort_index().ffill(axis=1).dropna(axis=1)
prices = prices_df.values

print(f"Prices shape: {prices.shape}")
print(f"Tickers: {list(prices_df.index)}")

# Grid search parameters
entry_thresholds = [0.01, 0.02, 0.05, 0.1, 0.2]
max_positions = [5, 10, 15, 20]
r2_cutoffs = [0.80, 0.85, 0.90, 0.95]

best_pnl = -np.inf
best_params = None
results = []

total = len(entry_thresholds) * len(max_positions) * len(r2_cutoffs)
count = 0

for entry_thresh, max_pos, r2_cut in product(entry_thresholds, max_positions, r2_cutoffs):
    count += 1
    try:
        actions = get_actions(prices, entry_threshold=entry_thresh, max_position=max_pos, r2_cutoff=r2_cut)

        # Apply 0.1% fee adjustment
        trade_volume = np.sum(np.abs(actions) * prices)
        fees = trade_volume * 0.001

        bt = Backtester(prices, actions, cash=25000)
        port_values, pnl = bt.eval_actions()

        if pnl is not None:
            net_pnl = pnl - fees
            results.append((entry_thresh, max_pos, r2_cut, pnl, fees, net_pnl))

            if net_pnl > best_pnl:
                best_pnl = net_pnl
                best_params = (entry_thresh, max_pos, r2_cut)

            if count % 10 == 0:
                print(f"[{count}/{total}] entry={entry_thresh}, max_pos={max_pos}, r2_cut={r2_cut} -> PnL={pnl:.2f}, fees={fees:.2f}, net={net_pnl:.2f}")
    except Exception as e:
        print(f"[{count}/{total}] FAILED: entry={entry_thresh}, max_pos={max_pos}, r2_cut={r2_cut} -> {e}")

print("\n" + "="*80)
print("TOP 10 RESULTS (by net PnL):")
print("="*80)
results.sort(key=lambda x: x[5], reverse=True)
for i, (et, mp, r2c, pnl, fees, net) in enumerate(results[:10]):
    print(f"  {i+1}. entry_threshold={et}, max_position={mp}, r2_cutoff={r2c}")
    print(f"     PnL={pnl:.2f}, fees={fees:.2f}, NET PnL={net:.2f}")

print(f"\nBEST: entry_threshold={best_params[0]}, max_position={best_params[1]}, r2_cutoff={best_params[2]}")
print(f"BEST NET PnL: ${best_pnl:.2f}")
