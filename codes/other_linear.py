import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

AAPL_price = pd.read_csv('AAPL.csv',usecols=['Date', 'Close'])
SPY_price = pd.read_csv('SPY.csv',usecols=['Date', 'Close'])

X = sm.add_constant(SPY_price['Close'])

model = sm.OLS(AAPL_price['Close'],X)
results = model.fit()
plt.scatter(SPY_price['Close'],AAPL_price['Close'],alpha=0.3)
y_predict = results.params[0] + results.params[1]*SPY_price['Close']

plt.plot(SPY_price['Close'],y_predict, linewidth=3)
plt.xlim(240,350)
plt.ylim(100,350)
plt.xlabel('SPY_price')
plt.ylabel('AAPL_price')
plt.title('OLS Regression')

print(results.summary())