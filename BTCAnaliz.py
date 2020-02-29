import quandl,datetime,math
from matplotlib import style
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

style.use('ggplot')
quandl.ApiConfig.api_key = "26CGFfrrz4KU9xDzmbJY"

df = quandl.get("BITSTAMP/USD")
df['HL_PCT'] = (df['High']-df['Low']) / df['Last'] * 100.0
df['ASK-BID_PCT'] = (df['Ask'] - df['Bid']) /df['Ask'] * 100.0

df = df[['High','Low','Volume','Last','HL_PCT','ASK-BID_PCT']]

forecast_out = int(math.ceil(len(df) * 0.01))
forecast_col = 'Last'

df['Label'] = df[forecast_col].shift(-forecast_out)

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_lately = X[-forecast_out:]
X = X[:-forecast_out]
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train,y_train)
accuracy = regressor.score(X_test,y_test)

forecast_set = regressor.predict(X_lately)
df['Forecast']  = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for i in range(len(df.columns)-1) ] + [i]

df['Last'].plot()
df['Forecast'].plot()
plt.xlabel('Date')
plt.ylabel('Price(USD)')
plt.legend(loc=4)
plt.show()











