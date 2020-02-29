import numpy as np
import pandas as pd
import datetime,math
import matplotlib.pyplot as plt
import matplotlib.style
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import svm


matplotlib.style.use('ggplot')

df = pd.read_csv('KLNMA.csv')

for i in df['Date']:
    q = i
    j = datetime.datetime.strptime(i, '%d-%m-%Y')
    df['Date'].loc[df['Date'].values == q] = j


df.index = df['Date']



for i in df['Volume']:
    if i.endswith('K'):
        q = i
        i = i[:-1]
        i = i.replace(',','.')
        i = float(i)
        i *= 1000
        df['Volume'].loc[df['Volume'].values == q] = str(i)
    elif i.endswith('M'):
        y = i
        i = i[:-1]
        i = i.replace(',','.')
        i = float(i)
        i *= 1000000
        df['Volume'].loc[df['Volume'].values == y] = str(i)

df['Volume'] = [float(i) for i in df['Volume']]

df['High'] = [i.replace(',','.') for i in df['High']]
df['High'] = [float(i) for i in df['High']]
df['Low'] = [i.replace(',','.') for i in df['Low']]
df['Low'] = [float(i) for i in df['Low']]
df['Open'] = [i.replace(',','.') for i in df['Open']]
df['Open'] = [float(i) for i in df['Open']]
df['Close'] = [i.replace(',','.') for i in df['Close']]
df['Close'] = [float(i) for i in df['Close']]

forecast_col = 'Close'
forecast_out = int(math.ceil(len(df)*0.005))

df['HL_PRC'] = (df['High']-df['Low']) / df['Close'] * 100.0
df['PRC_CHG'] = (df['Close']-df['Open']) / df['Open'] * 100.0

df = df[['Open','Close','HL_PRC','PRC_CHG','Volume']]

df['Label'] = df[forecast_col].shift(-forecast_out)

X = df.iloc[:,:-1]
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

y = df.iloc[:,-1]
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train,y_train)
print(regressor.score(X_test,y_test))

forecast_set = regressor.predict(X_lately)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day


for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for i in range(len(df.columns)-1)] + [i]

df['Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Tarih')
plt.ylabel('Fiyat')
plt.show()


















