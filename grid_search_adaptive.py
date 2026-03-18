import numpy as np
import pandas as pd
from itertools import product
from backtester import Backtester
from strategy_etf_arb_adaptive import get_actions

# Load data
df = pd.read_csv('dev_data_30.csv', parse_dates=['Date'], date_format='%m/%d/%y', thousands=',')
prices_df = df.pivot(index='Ticker', columns='Date', values='Open').sort_index().ffill(axis=1).dropna(axis=1)
prices = prices_df.values

print(f"Prices shape: {prices.shape}")
print(f"Tickers: {list(prices_df.index)}")

# Grid search parameters
k_entry_vals = [1.0, 1.5, 2.0, 2.5, 3.0]
k_exit_vals = [0.0, 0.25, 0.5]
vol_window_vals = [20, 40, 60]
max_position_vals = [10, 15, 20]

results = []
total = len(k_entry_vals) * len(k_exit_vals) * len(vol_window_vals) * len(max_position_vals)
print(f"Total combinations: {total}\n")

best_pnl = -float('inf')
best_params = None
count = 0

for k_entry, k_exit, vol_window, max_pos in product(k_entry_vals, k_exit_vals, vol_window_vals, max_position_vals):
    count += 1
    actions = get_actions(prices, k_entry=k_entry, k_exit=k_exit,
                          vol_window=vol_window, max_position=max_pos)

    bt = Backtester(prices, actions, cash=25000)

    # Suppress print from backtester
    import io, sys
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    port_values, pnl = bt.eval_actions()
    sys.stdout = old_stdout

    if pnl is not None:
        # Apply 0.1% fee on all trades
        trade_volume = np.sum(np.abs(actions) * prices)
        fees = trade_volume * 0.001
        net_pnl = pnl - fees

        results.append({
            'k_entry': k_entry, 'k_exit': k_exit,
            'vol_window': vol_window, 'max_position': max_pos,
            'gross_pnl': pnl, 'fees': fees, 'net_pnl': net_pnl
        })

        if net_pnl > best_pnl:
            best_pnl = net_pnl
            best_params = (k_entry, k_exit, vol_window, max_pos)
            print(f"[{count}/{total}] NEW BEST: k_entry={k_entry}, k_exit={k_exit}, "
                  f"vol_window={vol_window}, max_pos={max_pos} -> "
                  f"gross={pnl:.2f}, fees={fees:.2f}, net={net_pnl:.2f}")
    else:
        print(f"[{count}/{total}] FAILED: k_entry={k_entry}, k_exit={k_exit}, "
              f"vol_window={vol_window}, max_pos={max_pos}")

# Sort and show top 10
results.sort(key=lambda x: x['net_pnl'], reverse=True)
print("\n" + "="*80)
print("TOP 10 PARAMETER COMBINATIONS BY NET PNL (after 0.1% fees)")
print("="*80)
for i, r in enumerate(results[:10]):
    print(f"{i+1}. k_entry={r['k_entry']}, k_exit={r['k_exit']}, "
          f"vol_window={r['vol_window']}, max_pos={r['max_position']} | "
          f"gross={r['gross_pnl']:.2f}, fees={r['fees']:.2f}, NET={r['net_pnl']:.2f}")

print(f"\nBEST: k_entry={best_params[0]}, k_exit={best_params[1]}, "
      f"vol_window={best_params[2]}, max_pos={best_params[3]}")
print(f"BEST NET PNL: ${best_pnl:.2f}")
