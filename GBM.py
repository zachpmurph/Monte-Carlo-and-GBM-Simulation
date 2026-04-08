import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime
import math


start_date = datetime(year=2010, month=1, day=1)
end_date = datetime(year=2024, month=12, day=31)

data = yf.download('AAPL', start=start_date, end=end_date, auto_adjust=True, progress=False, interval = '1d')

data['Daily Return'] = data['Close'].pct_change()
data['Daily Volatility'] = data['Daily Return'].std()

volatility = data['Daily Volatility'].mean()
drift = data['Daily Return'].mean()


days_prediction = 252
n=1000

array_data = np.zeros((days_prediction+1, n))


columns = [f"col{i}" for i in range(n)]

gbm_df = pd.DataFrame(array_data, columns = columns)
gbm_df.loc[0] = data['Close'].iloc[-1, 0]

#Calculates Price for each Day
for day in range(days_prediction):
    rand_norm = np.random.normal(size=n)
    gbm_df.loc[day+1] = gbm_df.loc[day] * np.exp((np.sqrt(1/252)*rand_norm*volatility)+(drift - ((volatility**2)/2))*(1/252))


#Plots the 100 simulations        
for col in gbm_df.columns:
    plt.plot(gbm_df.index, gbm_df[col])

plt.show()



