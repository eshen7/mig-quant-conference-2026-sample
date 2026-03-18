# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a starter repo for the **MIG Algo Competition** — a quant trading competition where teams build algorithmic trading strategies. The goal is to implement a `get_actions(prices)` function that generates buy/sell signals for a portfolio of stocks, maximizing PnL starting from $25,000.

- **Competition docs**: https://mig-algo-challenge.vercel.app/
- **Discord**: https://discord.gg/depR4xR2

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Architecture

### Strategy Interface

All strategies must implement a single function:

```python
def get_actions(prices: np.ndarray) -> np.ndarray
```

- **Input**: `prices` — shape `(num_stocks, num_days)`, Open prices only. Rows are stocks sorted alphabetically by ticker.
- **Output**: `actions` — same shape. `+N` = buy N shares, `-N` = sell/short N shares, `0` = hold. Rounded to integers (no fractional shares).

### Key Files

- **`sample_strategy.py`** — Reference implementation of `get_actions` (5/20-day MA crossover). This is the submission template.
- **`backtester.py`** — Local replica of the server-side backtester. Handles long/short positions, FIFO short covering, and portfolio value tracking. Used via `Backtester(prices, actions, cash=25000).eval_actions()`.
- **`notebooks/01_data_exploration.ipynb`** — Data loading patterns, price distributions, correlations, volatility analysis.
- **`notebooks/02_strategy_development.ipynb`** — Strategy development workflow with backtesting and visualization. Contains a `backtest()` helper and `compare_strategies()` for testing multiple strategies.

### Data Format

CSV files (`dev_data_10.csv`, `dev_data_30.csv`) contain OHLCV data with columns: `Ticker, Date, Open, High, Low, Close, Adj. Close, Volume`. Dates use `%m/%d/%y` format with comma-formatted volume. Load with:

```python
df = pd.read_csv('dev_data_30.csv', parse_dates=['Date'], date_format='%m/%d/%y', thousands=',')
```

To build the prices matrix for `get_actions`:
```python
prices_df = df.pivot(index='Ticker', columns='Date', values='Open').sort_index()
prices = prices_df.values  # shape: (num_stocks, num_days)
```

### Competition Constraints

- Starting capital: $25,000
- Runtime limit: 60 seconds, memory limit: 512 MB
- No network access or file I/O in the sandbox
- Strategy must be deterministic
- Sandbox packages: numpy>=1.26, pandas>=2.0, scipy, scikit-learn>=1.3, statsmodels, ta-lib>=0.6.5, numba, joblib

### Backtester Behavior

- Actions are rounded to integers before execution
- Buys require sufficient cash; insufficient cash silently skips the trade
- Selling more than held shares automatically opens short positions
- Short covering uses FIFO order
- Portfolio going negative (from excessive shorting) terminates the backtest with failure
