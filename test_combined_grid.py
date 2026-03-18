import numpy as np
import pandas as pd
import time
from itertools import product
from backtester import Backtester
from strategy_etf_arb_combined import get_actions

# Load data
df = pd.read_csv('dev_data_30.csv', parse_dates=['Date'], date_format='%m/%d/%y', thousands=',')
prices_df = df.pivot(index='Ticker', columns='Date', values='Open').sort_index().ffill(axis=1).dropna(axis=1)
prices = prices_df.values
print(f"Prices shape: {prices.shape}")

# Coarse grid search
param_grid = {
    'window_size': [120, 180, 252],
    'retrain_every': [10, 20],
    'k_entry': [1.0, 1.5, 2.0, 2.5],
    'k_exit': [0.0, 0.25, 0.5],
    'vol_window': [20, 40],
    'max_position': [10, 15, 20],
}

keys = list(param_grid.keys())
values = list(param_grid.values())
combos = list(product(*values))
print(f"Total combinations: {len(combos)}")

best_pnl = -np.inf
best_params = None
results = []

t0 = time.time()
for idx, combo in enumerate(combos):
    params = dict(zip(keys, combo))
    try:
        actions = get_actions(prices, **params)
        bt = Backtester(prices, actions, cash=25000)

        # Suppress print
        import io, sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        port_values, pnl = bt.eval_actions()
        sys.stdout = old_stdout

        if pnl is not None:
            # Apply 0.1% fee
            total_trades = np.sum(np.abs(actions))
            # Fee = 0.1% of trade value; approximate with average price
            avg_price = np.mean(prices)
            fee = total_trades * avg_price * 0.001
            net_pnl = pnl - fee

            results.append((net_pnl, pnl, fee, params))
            if net_pnl > best_pnl:
                best_pnl = net_pnl
                best_params = params
                print(f"[{idx+1}/{len(combos)}] NEW BEST net_pnl={net_pnl:.2f} (gross={pnl:.2f}, fee={fee:.2f}) params={params}")
    except Exception as e:
        import sys
        sys.stdout = sys.__stdout__
        print(f"[{idx+1}/{len(combos)}] ERROR: {e} params={params}")

    if (idx+1) % 50 == 0:
        elapsed = time.time() - t0
        print(f"  ... {idx+1}/{len(combos)} done, elapsed {elapsed:.1f}s")

elapsed = time.time() - t0
print(f"\n=== COARSE GRID DONE in {elapsed:.1f}s ===")
print(f"Best net PnL: {best_pnl:.2f}")
print(f"Best params: {best_params}")

# Show top 10
results.sort(key=lambda x: x[0], reverse=True)
print("\nTop 10 results:")
for i, (net, gross, fee, p) in enumerate(results[:10]):
    print(f"  {i+1}. net={net:.2f} gross={gross:.2f} fee={fee:.2f} {p}")
