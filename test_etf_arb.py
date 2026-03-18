import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Users/edisonshen/Documents/mig2026/mig-quant-conference-2026-sample')

from backtester import Backtester
from strategy_etf_arb import get_actions


def load_data():
    df = pd.read_csv('/Users/edisonshen/Documents/mig2026/mig-quant-conference-2026-sample/dev_data_30.csv',
                      parse_dates=['Date'], date_format='%m/%d/%y', thousands=',')
    prices_df = df.pivot(index='Ticker', columns='Date', values='Open').sort_index().ffill(axis=1).dropna(axis=1)
    return prices_df.values, prices_df.index.tolist()


def compute_fees(prices, actions, fee_rate=0.001):
    """Compute total trading fees: 0.1% of trade value per trade."""
    actions_rounded = np.round(actions).astype(int)
    total_fees = 0.0
    for stock in range(prices.shape[0]):
        for day in range(prices.shape[1]):
            if actions_rounded[stock, day] != 0:
                total_fees += abs(actions_rounded[stock, day]) * prices[stock, day] * fee_rate
    return total_fees


def run_backtest(prices, entry_threshold, exit_threshold=0.0, max_position=15,
                 hedge_ratio=0.5, verbose=True):
    actions = get_actions(prices, entry_threshold=entry_threshold,
                         exit_threshold=exit_threshold,
                         max_position=max_position,
                         hedge_ratio=hedge_ratio)

    bt = Backtester(prices, actions, cash=25000)
    port_values, pnl = bt.eval_actions()

    if port_values is None:
        return None, None, None

    fees = compute_fees(prices, actions)
    fee_adjusted_pnl = pnl - fees

    if verbose:
        print(f"  Gross PnL: ${pnl:.2f}")
        print(f"  Total Fees: ${fees:.2f}")
        print(f"  Fee-Adjusted PnL: ${fee_adjusted_pnl:.2f}")
        print(f"  Return: {fee_adjusted_pnl / 25000 * 100:.2f}%")
        num_trades = np.sum(np.abs(np.round(actions).astype(int)) > 0)
        print(f"  Number of trade actions: {num_trades}")

    return port_values, pnl, fee_adjusted_pnl


def grid_search(prices):
    print("=" * 60)
    print("GRID SEARCH OVER PARAMETERS")
    print("=" * 60)

    results = []

    entry_thresholds = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    exit_thresholds = [0.0, 0.1, 0.2]
    max_positions = [5, 10, 15, 20]
    hedge_ratios = [0.3, 0.5, 0.7]

    for entry_t in entry_thresholds:
        for exit_t in exit_thresholds:
            for max_pos in max_positions:
                for hr in hedge_ratios:
                    try:
                        actions = get_actions(prices, entry_threshold=entry_t,
                                            exit_threshold=exit_t,
                                            max_position=max_pos,
                                            hedge_ratio=hr)
                        bt = Backtester(prices, actions, cash=25000)
                        port_values, pnl = bt.eval_actions()
                        if port_values is None:
                            continue
                        fees = compute_fees(prices, actions)
                        fee_adj_pnl = pnl - fees
                        results.append({
                            'entry_threshold': entry_t,
                            'exit_threshold': exit_t,
                            'max_position': max_pos,
                            'hedge_ratio': hr,
                            'gross_pnl': pnl,
                            'fees': fees,
                            'fee_adj_pnl': fee_adj_pnl,
                            'return_pct': fee_adj_pnl / 25000 * 100
                        })
                    except Exception as e:
                        pass

    # Sort by fee-adjusted PnL
    results.sort(key=lambda x: x['fee_adj_pnl'], reverse=True)

    print(f"\nTop 15 parameter combinations:")
    print(f"{'Entry':>8} {'Exit':>6} {'MaxPos':>6} {'Hedge':>6} {'Gross PnL':>12} {'Fees':>10} {'Net PnL':>12} {'Return%':>10}")
    print("-" * 80)
    for r in results[:15]:
        print(f"{r['entry_threshold']:>8.2f} {r['exit_threshold']:>6.2f} {r['max_position']:>6d} "
              f"{r['hedge_ratio']:>6.2f} {r['gross_pnl']:>12.2f} {r['fees']:>10.2f} "
              f"{r['fee_adj_pnl']:>12.2f} {r['return_pct']:>10.2f}%")

    # Find robust region: look at parameters where nearby values also perform well
    print("\n\nROBUSTNESS ANALYSIS:")
    print("Looking for 'flat regions' of good performance...")

    # Group by entry_threshold and compute mean performance
    from collections import defaultdict
    by_entry = defaultdict(list)
    for r in results:
        by_entry[r['entry_threshold']].append(r['fee_adj_pnl'])

    print(f"\n{'Entry Threshold':>16} {'Mean Net PnL':>12} {'Std':>10} {'Min':>10} {'Max':>10} {'Count':>6}")
    for et in sorted(by_entry.keys()):
        vals = by_entry[et]
        print(f"{et:>16.2f} {np.mean(vals):>12.2f} {np.std(vals):>10.2f} {np.min(vals):>10.2f} {np.max(vals):>10.2f} {len(vals):>6d}")

    return results


if __name__ == '__main__':
    prices, tickers = load_data()
    print(f"Loaded data: {prices.shape[0]} stocks, {prices.shape[1]} days")
    print(f"Tickers: {tickers}")
    print()

    # Run with default parameters first
    print("=" * 60)
    print("DEFAULT PARAMETERS (entry=1.5, exit=0.0)")
    print("=" * 60)
    run_backtest(prices, entry_threshold=1.5, exit_threshold=0.0)
    print()

    # Grid search
    results = grid_search(prices)

    # Run best parameters
    if results:
        best = results[0]
        print("\n" + "=" * 60)
        print(f"BEST PARAMETERS")
        print(f"  Entry threshold: {best['entry_threshold']}")
        print(f"  Exit threshold: {best['exit_threshold']}")
        print(f"  Max position: {best['max_position']}")
        print(f"  Hedge ratio: {best['hedge_ratio']}")
        print("=" * 60)
        run_backtest(prices, entry_threshold=best['entry_threshold'],
                    exit_threshold=best['exit_threshold'],
                    max_position=best['max_position'],
                    hedge_ratio=best['hedge_ratio'])
