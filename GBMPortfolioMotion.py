import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime
import math

#IMPORTING DATA
start_date = datetime(year=2000, month=1, day=1)
end_date = datetime(year=2024, month=12, day=31)
tickers = ['AAPL', 'MSFT', 'VOO', 'QQQ', 'ORCL', 'NVDA', 'LLY', 'JNJ', 'AMZN', 'WMT']


data = yf.download(tickers=tickers, start=start_date, end=end_date, auto_adjust=True, progress=False, interval = '1d', group_by='ticker')
#CREATES DATA FRAME WITH DAILY RETURNS FOR EACH STOCK
daily_returns = pd.DataFrame()
for ticker in tickers:
    daily_returns[ticker] = data[(ticker, 'Close')].pct_change()
    
#start of Efficient Frontier Simulation
daily_return_mean = daily_returns.mean()
annual_returns = daily_return_mean *252
returns_corr = daily_returns.corr()
L = np.linalg.cholesky(returns_corr)

daily_cov_matrix = daily_returns.cov()
annual_cov_matrix = daily_cov_matrix * 252

trials = 10000
columns = ['Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'AAPL', 'MSFT', 'VOO', 'SYP', 'ORCL', 'NVDA', 'LLY', 'JNJ', 'AMZN', 'WMT']
df = pd.DataFrame(columns = columns)

for i in range(trials):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)

    annual_return = np.sum(annual_returns * weights)
    annual_volatility = np.sqrt(weights.T @ annual_cov_matrix @ weights)
    sharpe_ratio = annual_return/annual_volatility

    df.loc[i] = [annual_return, annual_volatility, sharpe_ratio, weights[0], weights[1], weights[2], weights[3], weights[4], weights[5], weights[6], weights[7], weights[8], weights[9]]


max_sharpe_portfolio = df.loc[df['Sharpe Ratio'].idxmax(), :]
max_weights = np.array(max_sharpe_portfolio.iloc[3:])

#START OF BROWNIAN MOTION MODEL
#Roughly Optimal weights from Efficient Frontier Simulation
#annual weights, because GBM takes them and then converts to daily returns
return_means = daily_returns.mean().values *252
volatilities = daily_returns.std().values * np.sqrt(252)

#Length and num of sim
print("GBM SIMULATION FOR CONTINOUS PORTFOLIO WEIGHTS")
days_prediction = int(input("How many days?"))
n=1000

#create array of total returns of simulation
zeros = np.zeros((days_prediction+1, n))
gbm = pd.DataFrame(zeros)

#creates dataframe with all values, except the first value at 0.
array_data = np.zeros((days_prediction+1, len(tickers)))
columns = tickers
gbm_sharpe = np.zeros(n)
gbm_drawdown = np.zeros(n)

for sim in range(n):
    #resets gbm_df every time
    gbm_df = pd.DataFrame(array_data, columns = columns)
    for ticker in tickers:
        gbm_df.loc[0, ticker] = data[(ticker, 'Close')].iloc[-1]

    #Calculates Price for each Day
    for day in range(days_prediction):
        Z_indep = np.random.normal(size=len(tickers))
        rand_norm = L @ Z_indep
        gbm_df.loc[day+1] = gbm_df.loc[day] * np.exp((np.sqrt(1/252)*rand_norm*volatilities)+(return_means - ((volatilities**2)/2))*(1/252))

    #calculates portfolio variance
    sim_vol = np.sqrt(max_weights.T @ gbm_df.cov() @ max_weights)
    
    #creates TOTAL columna and PERCENT RETURN so that the returns across multiple portfolios are easier to compare. 
    gbm_weighted = gbm_df*max_weights
    gbm_df['TOTAL'] = gbm_weighted.sum(axis=1)
    gbm_df['PERCENT RETURN'] = gbm_df['TOTAL']/gbm_df.loc[0, 'TOTAL']-1
    gbm[sim] = gbm_df['PERCENT RETURN']

    #makes np array that contains maxdrawdown
    gbm_drawdown[sim] = ((gbm_df['TOTAL']-gbm_df['TOTAL'].cummax())/gbm_df['TOTAL'].cummax()).min()
    
    #makes np array that contains sharpe ratio
    gbm_sharpe[sim] = gbm_df['TOTAL'].iloc[-1]/gbm_df['TOTAL'].iloc[0]



##TODO
    #Make shocks correlated
print('BEGIN PROCESSING')
target_return = float(input('Target Return: '))
#RETURNS OF MONTE CARLO GBM MODEL
    #VaR and CVaR (CVaR/Expected Shortfall)
prob = np.mean(gbm.iloc[-1]>target_return)
print(prob)
    #Fan Chart (percentile bands over time)

#COMPUTE
    #Mean Portfolio Return (float)
mean_portfolios_return = gbm.iloc[-1].mean()

    #Volatility of final returns
portfolios_volatility = gbm.iloc[-1].std()

    #Sharpe Ratio for each simulation
        #plot distributions
sns.histplot(gbm_sharpe, kde=True, stat='probability')
plt.title('Sharpe Ratio')
plt.xlabel('Sharpe Ratio')

plt.ylabel('Probability')
plt.show()
    #Calculate Drawdown(max peak to trough loss)
        #summarize mean drawdown, max drawdown, and distribution of drawdowns
        #cummax()
sns.histplot(gbm_drawdown, kde=True, stat='probability')
plt.title('Drawdown')
plt.xlabel('Drawdown')
plt.ylabel('Probability')
plt.show()
    #Tail risk measures
        #Value at Risk - VaR 95% = 5th percentile of data
sns.histplot(gbm.iloc[-1], kde=True, stat='probability')
plt.title('VaR')
plt.xlabel('VaR')
plt.ylabel('Probability')
plt.show()


#Plots the 100 simulations
for col in gbm.columns:
    plt.plot(gbm.index, gbm[col])

plt.show()


