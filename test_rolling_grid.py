import numpy as np
import pandas as pd
from itertools import product
from backtester import Backtester
from strategy_etf_arb_rolling import get_actions
import time

# Load data
df = pd.read_csv('dev_data_30.csv', parse_dates=['Date'], date_format='%m/%d/%y', thousands=',')
prices_df = df.pivot(index='Ticker', columns='Date', values='Open').sort_index().ffill(axis=1).dropna(axis=1)
prices = prices_df.values

print(f"Prices shape: {prices.shape}")
print(f"Tickers: {list(prices_df.index)}")

# Grid search parameters
window_sizes = [60, 120, 180, 252]
retrain_freqs = [5, 10, 20]
entry_thresholds = [0.5, 1.0, 2.0, 3.0]
max_positions = [10, 15, 20]

total = len(window_sizes) * len(retrain_freqs) * len(entry_thresholds) * len(max_positions)
print(f"\nTotal configurations: {total}")

best_pnl = -np.inf
best_params = None
results = []

count = 0
t0 = time.time()

for ws, rf, et, mp in product(window_sizes, retrain_freqs, entry_thresholds, max_positions):
    count += 1
    try:
        actions = get_actions(prices, window_size=ws, retrain_freq=rf,
                              entry_threshold=et, max_position=mp)

        # Apply 0.1% fee penalty
        trade_volume = np.abs(actions)
        fee_cost = np.sum(trade_volume * prices * 0.001)

        bt = Backtester(prices, actions, cash=25000)
        port_values, pnl = bt.eval_actions()

        if pnl is not None:
            net_pnl = pnl - fee_cost
            results.append((ws, rf, et, mp, pnl, fee_cost, net_pnl))

            if net_pnl > best_pnl:
                best_pnl = net_pnl
                best_params = (ws, rf, et, mp)

            if count % 20 == 0:
                elapsed = time.time() - t0
                print(f"  [{count}/{total}] {elapsed:.1f}s | Best so far: net_pnl={best_pnl:.2f} params={best_params}")
    except Exception as e:
        print(f"  Error with ws={ws} rf={rf} et={et} mp={mp}: {e}")

elapsed = time.time() - t0
print(f"\n{'='*80}")
print(f"Grid search complete in {elapsed:.1f}s")
print(f"{'='*80}")

# Sort results by net PnL
results.sort(key=lambda x: x[-1], reverse=True)

print(f"\nTop 10 configurations:")
print(f"{'WinSize':>8} {'Retrain':>8} {'Entry':>8} {'MaxPos':>8} {'RawPnL':>10} {'Fees':>10} {'NetPnL':>10}")
print("-" * 72)
for ws, rf, et, mp, pnl, fees, net in results[:10]:
    print(f"{ws:>8} {rf:>8} {et:>8.1f} {mp:>8} {pnl:>10.2f} {fees:>10.2f} {net:>10.2f}")

print(f"\nBest parameters: window_size={best_params[0]}, retrain_freq={best_params[1]}, "
      f"entry_threshold={best_params[2]}, max_position={best_params[3]}")
print(f"Best net PnL (after 0.1% fees): ${best_pnl:.2f}")
