"""
Test script for ML Direction Prediction strategy.
Loads data, trains models, prints feature importances, runs backtester,
and tries different confidence thresholds.
"""

import numpy as np
import pandas as pd
from backtester import Backtester
from strategy_ml_direction import get_actions, get_feature_importances

# ─── 1. Load data ───────────────────────────────────────────────────────────
df = pd.read_csv('dev_data_30.csv', parse_dates=['Date'],
                 date_format='%m/%d/%y', thousands=',')
prices_df = (df.pivot(index='Ticker', columns='Date', values='Open')
             .sort_index().ffill(axis=1).dropna(axis=1))
prices = prices_df.values
tickers = prices_df.index.tolist()

print(f"Data shape: {prices.shape} ({len(tickers)} stocks, {prices.shape[1]} days)")
print(f"Tickers: {tickers}")
print(f"Train/test split: {int(prices.shape[1]*0.70)} / {prices.shape[1] - int(prices.shape[1]*0.70)} days")
print()

# ─── 2. Feature importances & model quality ──────────────────────────────────
print("=" * 70)
print("FEATURE IMPORTANCES & MODEL ACCURACY")
print("=" * 70)

importances, accuracies, feature_names = get_feature_importances(prices)

# Average importances across stocks
avg_imp = {f: 0.0 for f in feature_names}
for stock_imp in importances.values():
    for f, v in stock_imp.items():
        avg_imp[f] += v / len(importances)

print(f"\nModels trained: {len(importances)} / {len(tickers)}")
print(f"\nAverage feature coefficients (across all stocks):")
for f in sorted(avg_imp, key=lambda x: abs(avg_imp[x]), reverse=True):
    print(f"  {f:20s}: {avg_imp[f]:+.4f}")

print(f"\nPer-stock accuracy:")
for i in sorted(accuracies.keys()):
    acc = accuracies[i]
    print(f"  {tickers[i]:6s}  train={acc['train_acc']:.3f}  test={acc['test_acc']:.3f}")

avg_train = np.mean([a['train_acc'] for a in accuracies.values()])
avg_test = np.mean([a['test_acc'] for a in accuracies.values()])
print(f"\n  Average  train={avg_train:.3f}  test={avg_test:.3f}")
print()

# ─── 3. Run with default parameters ─────────────────────────────────────────
print("=" * 70)
print("DEFAULT PARAMETERS (buy>0.54, sell<0.46)")
print("=" * 70)
actions = get_actions(prices)
bt = Backtester(prices, actions, cash=25000)
port_values, pnl = bt.eval_actions()
if pnl is not None:
    ret_pct = (pnl / 25000) * 100
    print(f"Return: {ret_pct:.2f}%")
    # Estimate fee impact
    total_trades = np.abs(actions).sum()
    avg_price = prices.mean()
    est_fees = total_trades * avg_price * 0.001
    print(f"Total trades (shares): {total_trades:.0f}, Est. fees: ${est_fees:.2f}")
print()

# ─── 4. Try different confidence thresholds ──────────────────────────────────
print("=" * 70)
print("THRESHOLD SWEEP")
print("=" * 70)

thresholds = [
    (0.51, 0.49, "Aggressive (0.51/0.49)"),
    (0.52, 0.48, "Moderate-Agg (0.52/0.48)"),
    (0.53, 0.47, "Moderate (0.53/0.47)"),
    (0.54, 0.46, "Default (0.54/0.46)"),
    (0.55, 0.45, "Conservative (0.55/0.45)"),
    (0.56, 0.44, "Very Conserv (0.56/0.44)"),
    (0.58, 0.42, "Ultra Conserv (0.58/0.42)"),
    (0.60, 0.40, "Extreme (0.60/0.40)"),
]

best_pnl = -np.inf
best_params = None

for buy_th, sell_th, label in thresholds:
    actions = get_actions(prices, confidence_buy=buy_th, confidence_sell=sell_th)
    bt = Backtester(prices, actions, cash=25000)
    port_values, pnl = bt.eval_actions()
    if pnl is not None:
        ret_pct = (pnl / 25000) * 100
        total_trades = np.abs(actions).sum()
        print(f"  {label:30s}  PnL=${pnl:>10.2f}  Return={ret_pct:>7.2f}%  Trades={total_trades:>6.0f}")
        if pnl > best_pnl:
            best_pnl = pnl
            best_params = (buy_th, sell_th, label)

print()
if best_params:
    print(f"Best: {best_params[2]} with PnL=${best_pnl:.2f} ({best_pnl/250:.2f}% return)")

# ─── 5. Dollar-per-trade sweep with best threshold ──────────────────────────
print()
print("=" * 70)
print(f"DOLLAR-PER-TRADE SWEEP (using {best_params[2] if best_params else 'default'})")
print("=" * 70)

buy_th = best_params[0] if best_params else 0.54
sell_th = best_params[1] if best_params else 0.46

for dpt in [200, 400, 600, 800, 1000, 1500, 2000]:
    actions = get_actions(prices, confidence_buy=buy_th, confidence_sell=sell_th, dollar_per_trade=dpt)
    bt = Backtester(prices, actions, cash=25000)
    port_values, pnl = bt.eval_actions()
    if pnl is not None:
        ret_pct = (pnl / 25000) * 100
        total_trades = np.abs(actions).sum()
        print(f"  DPT=${dpt:>5d}  PnL=${pnl:>10.2f}  Return={ret_pct:>7.2f}%  Trades={total_trades:>6.0f}")
