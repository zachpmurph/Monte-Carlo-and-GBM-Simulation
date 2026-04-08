import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime
from random import random

start_date = datetime(year=2020, month=1, day=1)
end_date = datetime(year=2024, month=12, day=31)

tickers = ['AAPL', 'MSFT', 'VOO', 'QQQ', 'ORCL', 'NVDA', 'LLY', 'JNJ', 'AMZN', 'WMT']

data = yf.download(
    tickers=tickers,
    start=start_date,
    end=end_date,
    interval="1d",
    group_by="ticker",
    auto_adjust=True,
    progress=False
)
daily_returns = pd.DataFrame()

for ticker in tickers:
    daily_returns[ticker] = data[(ticker, 'Close')].pct_change().dropna()

daily_return_mean = daily_returns.mean()
annual_returns = daily_return_mean *252

daily_cov_matrix = daily_returns.cov()
annual_cov_matrix = daily_cov_matrix * 252

n = 10000
columns = ['Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'AAPL', 'MSFT', 'VOO', 'SYP', 'ORCL', 'NVDA', 'LLY', 'JNJ', 'AMZN', 'WMT']
df = pd.DataFrame(columns = columns)

for i in range(n):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)

    annual_return = np.sum(annual_returns * weights)
    annual_volatility = np.sqrt(weights.T @ annual_cov_matrix @ weights)
    sharpe_ratio = annual_return/annual_volatility

    df.loc[i] = [annual_return, annual_volatility, sharpe_ratio, weights[0], weights[1], weights[2], weights[3], weights[4], weights[5], weights[6], weights[7], weights[8], weights[9]]


sns.histplot(df['Sharpe Ratio'], kde=True, stat='probability')
plt.title('Sharpe Ratio Distribution')
plt.xlabel('Sharpe Ratio')
plt.ylabel('Probability')
plt.show()

max_sharpe_portfolio = df.loc[df['Sharpe Ratio'].idxmax(), :]
print(max_sharpe_portfolio)