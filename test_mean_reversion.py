import numpy as np
import pandas as pd
from backtester import Backtester
from strategy_mean_reversion import get_actions

# --- Load data ---
df = pd.read_csv('dev_data_30.csv', parse_dates=['Date'], date_format='%m/%d/%y', thousands=',')
prices_df = df.pivot(index='Ticker', columns='Date', values='Open').sort_index().ffill(axis=1).dropna(axis=1)
prices = prices_df.values
tickers = prices_df.index.tolist()

num_stocks, num_days = prices.shape
print(f"Data: {num_stocks} stocks, {num_days} days")
print(f"Tickers: {tickers}")
print(f"Price range example (stock 0): {prices[0].min():.2f} - {prices[0].max():.2f}")
print()

# --- Step 1: Analyze autocorrelation ---
train_end = int(num_days * 0.70)
print("=== Autocorrelation Analysis (training period) ===")
autocorrs = []
for i in range(num_stocks):
    train_prices = prices[i, :train_end]
    returns = np.diff(train_prices) / train_prices[:-1]
    returns = returns[np.isfinite(returns)]
    if len(returns) < 20:
        autocorrs.append(0)
        continue
    ac = np.corrcoef(returns[:-1], returns[1:])[0, 1]
    autocorrs.append(ac)
    marker = " <-- mean reverting" if ac < -0.02 else ""
    print(f"  {tickers[i]:>6s}: autocorr = {ac:+.4f}{marker}")
print()

# --- Step 2: Grid search ---
print("=== Grid Search ===")
best_pnl = -np.inf
best_params = {}

ema_spans = [5, 8, 10, 12, 15, 20]
thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80]
dollar_amounts = [300, 400, 500]
autocorr_cutoffs = [-0.01, -0.02, -0.03, -0.05]

# Coarse grid first
results = []
for ema_span in ema_spans:
    for thresh in thresholds:
        for dollar in dollar_amounts:
            for ac_cut in autocorr_cutoffs:
                actions = get_actions(prices, ema_span=ema_span, threshold=thresh,
                                      dollar_per_trade=dollar, autocorr_cutoff=ac_cut)

                # Estimate fee impact
                total_trades = np.sum(np.abs(actions) > 0)
                trade_values = np.sum(np.abs(actions) * prices)
                fee_cost = trade_values * 0.001  # 0.1% fees

                bt = Backtester(prices, actions, cash=25000)
                import io, sys
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                port_values, pnl = bt.eval_actions()
                sys.stdout = old_stdout

                if pnl is not None:
                    fee_adj_pnl = pnl - fee_cost
                    results.append({
                        'ema_span': ema_span, 'threshold': thresh,
                        'dollar': dollar, 'ac_cut': ac_cut,
                        'pnl': pnl, 'fee_cost': fee_cost,
                        'fee_adj_pnl': fee_adj_pnl,
                        'num_trades': total_trades,
                        'final_value': port_values[-1] if port_values else 0,
                    })

                    if fee_adj_pnl > best_pnl:
                        best_pnl = fee_adj_pnl
                        best_params = {'ema_span': ema_span, 'threshold': thresh,
                                       'dollar': dollar, 'ac_cut': ac_cut}

# Sort and show top 10
results.sort(key=lambda x: x['fee_adj_pnl'], reverse=True)
print(f"\nTop 10 parameter combinations (fee-adjusted PnL):")
print(f"{'EMA':>5} {'Thresh':>7} {'Dollar':>7} {'AC_cut':>7} {'PnL':>10} {'Fees':>10} {'Adj PnL':>10} {'Trades':>8}")
for r in results[:10]:
    print(f"{r['ema_span']:>5} {r['threshold']:>7.2f} {r['dollar']:>7} {r['ac_cut']:>7.2f} "
          f"{r['pnl']:>10.2f} {r['fee_cost']:>10.2f} {r['fee_adj_pnl']:>10.2f} {r['num_trades']:>8}")

print(f"\nBest params: {best_params}")
print(f"Best fee-adjusted PnL: ${best_pnl:.2f}")

# --- Step 3: Run best strategy and show details ---
print("\n=== Final Run with Best Parameters ===")
actions = get_actions(prices, **best_params)
bt = Backtester(prices, actions, cash=25000)
port_values, pnl = bt.eval_actions()

total_trades = np.sum(np.abs(actions) > 0)
trade_values = np.sum(np.abs(actions) * prices)
fee_cost = trade_values * 0.001
print(f"Estimated fee cost: ${fee_cost:.2f}")
print(f"Fee-adjusted PnL: ${pnl - fee_cost:.2f}")
print(f"Total trade events: {total_trades}")
print(f"Return: {(pnl / 25000) * 100:.2f}%")
print(f"Fee-adjusted Return: {((pnl - fee_cost) / 25000) * 100:.2f}%")

# Per-stock breakdown
print("\n=== Per-Stock Activity ===")
for i in range(num_stocks):
    stock_trades = np.sum(np.abs(actions[i]) > 0)
    if stock_trades > 0:
        stock_buys = np.sum(actions[i] > 0)
        stock_sells = np.sum(actions[i] < 0)
        print(f"  {tickers[i]:>6s}: {stock_trades:>4} trades ({stock_buys} buys, {stock_sells} sells)")
