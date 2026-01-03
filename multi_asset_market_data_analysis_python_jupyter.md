# Multi-Asset Market Data Analysis (Python | Jupyter) 
# Assets: Equities, FX, Crypto, Commodities  
# Skills: Data ingestion, cleaning, returns, volatility, correlation, visualization, basic signals

1. Project Overview

This notebook demonstrates a **step-by-step workflow** for analyzing multiple financial assets using Python. The goal is to transform **raw price data** into **actionable insights** using statistical and visual techniques commonly used by **Financial Data Analysts, Trading Analysts, and Quant interns**.


# 2. Environment Setup

# 2.1 Install Required Libraries

!pip install yfinance pandas numpy matplotlib seaborn scipy statsmodels

# 2.2 Import Libraries

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

plt.style.use('seaborn-v0_8')

## 3. Asset Selection

We select representative assets across markets:

| Asset Class | Ticker | Description |
|-----------|-------|-------------|
| Equity | AAPL | Apple Inc |
| FX | EURUSD=X | Euro / Dollar |
| Crypto | BTC-USD | Bitcoin |
| Commodity | GC=F | Gold Futures |

tickers = {
    'AAPL': 'Equity',
    'EURUSD=X': 'FX',
    'BTC-USD': 'Crypto',
    'GC=F': 'Commodity'
}


## 4. Data Acquisition

start_date = '2018-01-01'
end_date = '2025-01-01'

data = yf.download(list(tickers.keys()), start=start_date, end=end_date)['Adj Close']
data.head()

# 5. Data Cleaning & Quality Checks

# 5.1 Missing Data Inspection

data.isna().sum()

# 5.2 Handling Missing Values

data = data.fillna(method='ffill').dropna()

# 6. Return Computation

# 6.1 Log Returns (Preferred for Quant Analysis)

log_returns = np.log(data / data.shift(1)).dropna()
log_returns.head()

# 6.2 Cumulative Returns

cumulative_returns = log_returns.cumsum().apply(np.exp)

cumulative_returns.plot(figsize=(12,6))
plt.title('Cumulative Returns Across Asset Classes')
plt.show()

# 7. Descriptive Statistics

stats = pd.DataFrame({
    'Mean': log_returns.mean() * 252,
    'Volatility': log_returns.std() * np.sqrt(252),
    'Skewness': log_returns.apply(skew),
    'Kurtosis': log_returns.apply(kurtosis)
})

stats

# 8. Volatility Analysis
# 8.1 Rolling Volatility (30-Day)

rolling_vol = log_returns.rolling(30).std() * np.sqrt(252)

rolling_vol.plot(figsize=(12,6))
plt.title('30-Day Rolling Annualized Volatility')
plt.show()

# 9. Correlation & Diversification
# 9.1 Correlation Matrix

corr = log_returns.corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Asset Correlation Matrix')
plt.show()

# Insight: Low or negative correlation improves diversification.


# 10. Drawdown Analysis

def compute_drawdown(series):
    cumulative = series.cumsum().apply(np.exp)
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown

plt.figure(figsize=(12,6))
for asset in log_returns.columns:
    plt.plot(compute_drawdown(log_returns[asset]), label=asset)

plt.legend()
plt.title('Drawdown Analysis')
plt.show()

# 11. Simple Signal Example (Momentum)
# 11.1 20/100 Moving Average Signal

signals = {}

for asset in data.columns:
    ma_short = data[asset].rolling(20).mean()
    ma_long = data[asset].rolling(100).mean()
    signals[asset] = np.where(ma_short > ma_long, 1, -1)

signals = pd.DataFrame(signals, index=data.index)

# 11.2 Strategy Returns

strategy_returns = signals.shift(1) * log_returns

(strategy_returns.cumsum().apply(np.exp)).plot(figsize=(12,6))
plt.title('Momentum Strategy Performance')
plt.show()

# 12. Risk-Adjusted Performance

sharpe_ratio = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
sharpe_ratio

## 13. Key Takeaways

- Multi-asset analysis improves risk awareness
- Log returns are essential for modeling
- Correlation drives portfolio construction
- Even simple signals can be evaluated quantitatively