"""
backtest_full.py — Extended backtest using real ticker data (2010–2025)
======================================================================

1. Downloads real price data via yfinance for all 30 tickers (2010–2025)
2. Marks the dev_data period (2017-03-15 to 2021-03-12) as the training window
3. Runs etf_arb_tuned with train_start/train_end fixed to the dev_data boundary
4. Reports pre-sample, in-sample, and out-of-sample metrics separately
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from backtester import Backtester
from etf_arb_tuned import get_actions

# ── Ticker mapping (anonymous -> real) ──────────────────────────────────────
TICKER_MAP = {
    # Confirmed (corr=1.0000)
    'AB': 'ORCL',
    'AC': 'CSCO',
    'AD': 'TSM',
    'D':  'PLAY',
    'F':  'JBLU',
    'G':  'KLAC',
    'H':  'CFG',
    'I':  'EA',
    'J':  'ASML',
    'K':  'AAPL',
    'L':  'NVDA',
    'M':  'TSLA',
    'N':  'PFE',
    'O':  'BMY',
    'P':  'XOM',
    'Q':  'SLB',
    'R':  'JPM',
    'S':  'GS',
    'T':  'MCD',
    'U':  'NKE',
    'V':  'PG',
    'W':  'CAT',
    'X':  'BA',
    'Y':  'FCX',
    'Z':  'NEE',
    'AA': 'AMT',
    # Unidentified (excluded from OOS)
    # 'A':  ???  (6.70->47.07, min=1.49)
    # 'B':  ???  (8.80->40.88)
    # 'C':  ???  (7.85->18.84, min=3.07)
    # 'E':  ???  (28.83->34.95, range 20-38)
}

# Anonymous tickers with confirmed mappings only
ANON_TICKERS = sorted(TICKER_MAP.keys())
REAL_TICKERS = [TICKER_MAP[t] for t in ANON_TICKERS]

# All 30 anonymous tickers in the dev data (including unidentified ones)
ALL_ANON = sorted(['A','AA','AB','AC','AD','B','C','D','E','F','G','H','I','J','K',
                   'L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])
# Indices of confirmed tickers in the full 30-stock matrix
CONFIRMED_INDICES = [ALL_ANON.index(t) for t in ANON_TICKERS]

# ── Dev data boundaries (for marking training window) ───────────────────────
DEV_START = pd.Timestamp('2017-03-15')
DEV_END   = pd.Timestamp('2021-03-12')

# ── Download full 2010–2025 dataset ─────────────────────────────────────────
DOWNLOAD_START = '2010-01-01'
DOWNLOAD_END   = '2025-03-19'

print(f"Downloading data for {len(set(REAL_TICKERS))} unique tickers ({DOWNLOAD_START} to {DOWNLOAD_END})...")
unique_tickers = list(set(REAL_TICKERS))
raw_data = yf.download(unique_tickers, start=DOWNLOAD_START, end=DOWNLOAD_END, auto_adjust=False)

# Extract Open prices per anonymous ticker
frames = []
for anon, real in zip(ANON_TICKERS, REAL_TICKERS):
    try:
        opens = raw_data['Open'][real].dropna()
        frames.append(opens)
    except KeyError:
        print(f"  WARNING: No data for {anon} ({real})")
        frames.append(pd.Series(dtype=float))

# Find common dates where all tickers have data
date_sets = [set(s.index) for s in frames if len(s) > 0]
common_dates = sorted(set.intersection(*date_sets))
print(f"Common trading days across all tickers: {len(common_dates)}")
print(f"Date range: {common_dates[0].date()} to {common_dates[-1].date()}")

# Build price matrix (num_stocks x num_days)
prices = np.zeros((len(ANON_TICKERS), len(common_dates)))
for i, series in enumerate(frames):
    for j, date in enumerate(common_dates):
        prices[i, j] = series[date]

dates_arr = pd.DatetimeIndex(common_dates)

# ── Identify training window indices ───────────────────────────────────────
# Find the closest dates to DEV_START and DEV_END
train_start_idx = np.searchsorted(dates_arr, DEV_START)
train_end_idx = np.searchsorted(dates_arr, DEV_END, side='right')

# Clamp to valid range
train_start_idx = min(train_start_idx, len(dates_arr) - 1)
train_end_idx = min(train_end_idx, len(dates_arr))

num_total = len(common_dates)

print(f"\nPrice matrix: {prices.shape[0]} stocks x {num_total} days")
print(f"Pre-sample:  days 0–{train_start_idx-1} ({dates_arr[0].date()} to {dates_arr[train_start_idx-1].date()})")
print(f"Training:    days {train_start_idx}–{train_end_idx-1} ({dates_arr[train_start_idx].date()} to {dates_arr[train_end_idx-1].date()})")
print(f"OOS:         days {train_end_idx}–{num_total-1} ({dates_arr[train_end_idx].date()} to {dates_arr[-1].date()})")

# ── Run strategy ────────────────────────────────────────────────────────────
print(f"\nRunning etf_arb_tuned with train_start={train_start_idx}, train_end={train_end_idx}...")
actions = get_actions(prices, train_start=train_start_idx, train_end=train_end_idx)

# ── Backtest ────────────────────────────────────────────────────────────────
bt = Backtester(prices, actions, cash=25000)
port_values, pnl = bt.eval_actions()
port_values = np.array(port_values)

# ── Split metrics ───────────────────────────────────────────────────────────
def calc_metrics(pv):
    if len(pv) < 2:
        return 0, 0, 0
    daily = np.diff(pv)
    sharpe = np.mean(daily) / (np.std(daily) + 1e-10) * np.sqrt(252)
    running_max = np.maximum.accumulate(pv)
    max_dd = np.max((running_max - pv) / running_max)
    period_pnl = pv[-1] - pv[0]
    return period_pnl, sharpe, max_dd

pv_pre = port_values[:train_start_idx]
pv_train = port_values[train_start_idx:train_end_idx]
pv_oos = port_values[train_end_idx:]

pnl_pre, sharpe_pre, dd_pre = calc_metrics(pv_pre)
pnl_train, sharpe_train, dd_train = calc_metrics(pv_train)
pnl_oos, sharpe_oos, dd_oos = calc_metrics(pv_oos)

acts_pre = np.count_nonzero(actions[:, :train_start_idx])
acts_train = np.count_nonzero(actions[:, train_start_idx:train_end_idx])
acts_oos = np.count_nonzero(actions[:, train_end_idx:])

print(f"\n{'='*70}")
print(f"{'Period':<20s} {'Days':>6s} {'PnL':>10s} {'Sharpe':>8s} {'MaxDD':>8s} {'Trades':>8s}")
print(f"{'-'*70}")
print(f"{'Pre-Sample':<20s} {train_start_idx:>6d} ${pnl_pre:>9,.0f} {sharpe_pre:>8.2f} {dd_pre:>7.1%} {acts_pre:>8d}")
print(f"{'In-Sample (train)':<20s} {train_end_idx-train_start_idx:>6d} ${pnl_train:>9,.0f} {sharpe_train:>8.2f} {dd_train:>7.1%} {acts_train:>8d}")
print(f"{'Out-of-Sample':<20s} {num_total-train_end_idx:>6d} ${pnl_oos:>9,.0f} {sharpe_oos:>8.2f} {dd_oos:>7.1%} {acts_oos:>8d}")
print(f"{'-'*70}")
print(f"{'Total':<20s} {num_total:>6d} ${pnl:>9,.0f}")
print(f"{'='*70}")

# ── Plot ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(18, 13), sharex=True,
                         gridspec_kw={'height_ratios': [2, 1.2, 1]})

# Portfolio Value
axes[0].plot(dates_arr, port_values, color='#10b981', linewidth=1.5)
axes[0].axhline(25_000, color='#64748b', linewidth=0.8, linestyle='--', label='Starting Capital')
axes[0].axvline(dates_arr[train_start_idx], color='#38bdf8', linewidth=1.5, linestyle='--',
                label=f'Train Start ({dates_arr[train_start_idx].date()})')
axes[0].axvline(dates_arr[train_end_idx], color='#f97316', linewidth=1.5, linestyle='--',
                label=f'OOS Start ({dates_arr[train_end_idx].date()})')
axes[0].axvspan(dates_arr[0], dates_arr[train_start_idx], alpha=0.05, color='#38bdf8', label='Pre-Sample')
axes[0].axvspan(dates_arr[train_start_idx], dates_arr[train_end_idx-1], alpha=0.05, color='#a78bfa', label='Training')
axes[0].axvspan(dates_arr[train_end_idx], dates_arr[-1], alpha=0.05, color='#f97316', label='OOS')
axes[0].set_ylabel('Portfolio Value ($)')
axes[0].set_title('ETF Arb Tuned — Full Backtest (2010–2025)', fontsize=14)
axes[0].legend(loc='upper left', fontsize=8, ncol=3)
axes[0].grid(True, alpha=0.2)

# Cumulative PnL
pnl_curve = port_values - 25_000
pos_mask = pnl_curve >= 0
axes[1].fill_between(dates_arr, pnl_curve, where=pos_mask, color='#10b981', alpha=0.4)
axes[1].fill_between(dates_arr, pnl_curve, where=~pos_mask, color='#f43f5e', alpha=0.4)
axes[1].plot(dates_arr, pnl_curve, color='#10b981', linewidth=1.2)
axes[1].axhline(0, color='#64748b', linewidth=0.8, linestyle='--')
axes[1].axvline(dates_arr[train_start_idx], color='#38bdf8', linewidth=1.5, linestyle='--')
axes[1].axvline(dates_arr[train_end_idx], color='#f97316', linewidth=1.5, linestyle='--')
axes[1].set_ylabel('Cumulative PnL ($)')
axes[1].set_title('Cumulative PnL', fontsize=12)
axes[1].grid(True, alpha=0.2)

# Drawdown
running_max = np.maximum.accumulate(port_values)
dd_curve = (running_max - port_values) / running_max * 100
axes[2].fill_between(dates_arr, -dd_curve, color='#f43f5e', alpha=0.4)
axes[2].axhline(0, color='#64748b', linewidth=0.8)
axes[2].axvline(dates_arr[train_start_idx], color='#38bdf8', linewidth=1.5, linestyle='--')
axes[2].axvline(dates_arr[train_end_idx], color='#f97316', linewidth=1.5, linestyle='--')
axes[2].set_ylabel('Drawdown (%)')
axes[2].set_title('Drawdown', fontsize=12)
axes[2].grid(True, alpha=0.2)
axes[2].xaxis.set_major_locator(mdates.YearLocator())
axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=30, ha='right')

plt.tight_layout()
plt.savefig('backtest_full_results.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to backtest_full_results.png")
