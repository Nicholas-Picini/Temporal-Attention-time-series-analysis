import Models
import keras
import talib
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from datetime import datetime
from fancyimpute import KNN
from keras.models import load_model
from statsmodels.tsa.arima_model import ARIMA

np.random.seed(123)


ticker = 'AAPL'
stock = pdr.get_data_yahoo(ticker.upper(), start='2009-01-14', end=str(datetime.now().date()))
cor = pdr.get_data_yahoo('MSFT', start='2009-01-14', end=str(datetime.now().date()))
cor = cor["Close"]
retruns = np.log(stock["Close"] / stock["Close"].shift(1))

cor = cor.rename("Corr")
retruns = retruns.rename("retruns")

stock = stock.join(cor)
stock = stock.join(retruns)

gdp = pdr.DataReader('GDP', 'fred', start='2009-01-14', end=str(datetime.now().date()))
unem = pdr.DataReader('UNRATE', 'fred', start='2009-01-14', end=str(datetime.now().date()))
payem = pdr.DataReader('PAYEMS', 'fred', start='2009-01-14', end=str(datetime.now().date()))


stock = stock.merge(gdp,how='left', left_on=stock.index, right_on=gdp.index)
stock = stock.fillna(method='bfill')
stock = stock.fillna(method='ffill')
stock = stock.set_index('key_0', drop=True)

stock = stock.merge(unem,how='left', left_on=stock.index, right_on=unem.index)
stock = stock.fillna(method='bfill')
stock = stock.fillna(method='ffill')
stock = stock.set_index('key_0', drop=True)

stock = stock.merge(payem,how='left', left_on=stock.index, right_on=payem.index)
stock = stock.fillna(method='bfill')
stock = stock.fillna(method='ffill')
stock = stock.set_index('key_0', drop=True)


# VIX Data
from datapackage import Package
package = Package('https://datahub.io/core/finance-vix/datapackage.json')

for resource in package.resources:
    if resource.descriptor['datahub']['type'] == 'derived/csv':
        vix = pd.DataFrame(resource.read())
vix = vix.iloc[1267:,:]
vix = vix.set_index(vix.iloc[:,0], drop=True)
vix = vix[4]
vix = pd.to_numeric(vix, errors='coerce')


stock = pd.concat([stock, vix], axis=1)
stock = stock.rename(columns={stock.columns[11]: "VIX"})

up = pd.DataFrame(data={"UP":np.where(stock["Close"].shift(-1) > stock["Close"], 1, 0)})
dn = pd.DataFrame(data={"DN":np.where(stock["Close"].shift(-1) < stock["Close"], 1, 0)})

lbls = up.join(dn)

lbls_train = lbls.iloc[0:round(len(lbls)*0.8),:]
lbls_test = lbls.iloc[round(len(lbls)*0.8):,:]


cols = list(stock)[0:13]

#Preprocess data
stock = stock[cols].astype(str)
for i in cols:
    for j in range(0,len(stock)):
        stock[i][j] = stock[i][j].replace(",","")

stock = stock.astype(float)

periods = np.double(np.array(list(range(0, len(stock['Close'])))))

HT = talib.HT_TRENDMODE(stock['Close'])
rsi = talib.RSI(stock['Close'], timeperiod=5)
wma = talib.WMA(stock['Close'], timeperiod=20)
mavp = talib.MAVP(stock['Close'], periods, minperiod=2, maxperiod=30, matype=0)
upperband, middleband, lowerband = talib.BBANDS(stock['Close'], timeperiod=10, nbdevup=2, nbdevdn=2, matype=0)
Roc = talib.ROC(stock['Close'], timeperiod=5)
Atr = talib.ATR(stock['High'], stock['Low'], stock['Close'], timeperiod=10)
div = stock.Close - wma
voldiff = stock.Volume.diff()
VolROC = (stock.Volume - stock.Volume.shift(1)) / stock.Volume
opendiff = stock.Open - stock.Open.shift(1)


X = stock.Close.values
size = int(len(X) * 0.66)
train, test = X[0:size], X
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(0,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
arima = pd.DataFrame(predictions)
arima.iloc[0,:] = None


HT = pd.DataFrame(data={'TRENDMODE':HT})
wma = pd.DataFrame(data={'WMA':wma})
rsi = pd.DataFrame(data={'RSI':rsi})
mavp = pd.DataFrame(data={'MAVP':mavp})
upperband = pd.DataFrame(data={'upperband':upperband})
middleband = pd.DataFrame(data={'middleband':middleband})
lowerband = pd.DataFrame(data={'lowerband':lowerband})
Roc = pd.DataFrame(data={'Roc':Roc})
Atr = pd.DataFrame(data={'Atr':Atr})
div = pd.DataFrame(data={'Div':div})
voldiff = pd.DataFrame(data={'Volume Diff':voldiff})
VolROC = pd.DataFrame(data={'Volume ROC':VolROC})
opendiff = pd.DataFrame(data={'Open Diff':opendiff})
arima = np.array(arima)

pctrank = lambda x: pd.Series(x).rank(pct=True).iloc[-1]
test = stock['GDP']
test1 = stock['UNRATE']
rollingrank = test.rolling(window=50).apply(pctrank)
rollingrank1 = test1.rolling(window=50).apply(pctrank)
rollingrank = rollingrank.rename("GDPROL")
rollingrank1 = rollingrank1.rename("UNROL")
rollingrank = rollingrank.fillna(method='bfill')
rollingrank1 = rollingrank1.fillna(method='bfill')


stock = stock.join(HT)
stock = stock.join(wma)
stock = stock.join(rsi)
stock = stock.join(mavp)
stock = stock.join(upperband)
stock = stock.join(middleband)
stock = stock.join(lowerband)
stock = stock.join(Roc)
stock = stock.join(Atr)
stock = stock.join(rollingrank)
stock = stock.join(rollingrank1)
stock = stock.join(div)
stock = stock.join(voldiff)
stock = stock.join(VolROC)
stock = stock.join(opendiff)


# Impute missing values using KNN
stock = stock.as_matrix()
stock = np.append(stock, arima, 1)
stock = KNN(k=15).fit_transform(stock)

stock = pd.DataFrame(stock)

stock_train = stock.iloc[0:round(len(stock)*0.8),:]
stock_test = stock.iloc[round(len(stock)*0.8):,:]

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(stock_train)
sc_predict = MinMaxScaler()
test_set_scaled = sc_predict.fit_transform(stock_test)

X_train = []
y_train = []

n_future = 3  # Number of time steps predicted
n_past = 20  # Number of time steps used in preditcion

for i in range(n_past, len(training_set_scaled) - n_future + 1):
    X_train.append(training_set_scaled[i - n_past:i, 0:stock.shape[1]])
    y_train.append(training_set_scaled[i+n_future-1:i + n_future, 3])

X_train = np.array(X_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], stock.shape[1]))

lbls_train = lbls_train.iloc[:-n_past]


# 1 hidden layer network with input: n_past x num_features, hidden 120x5, output 2x1
template = [[n_past, stock.shape[1]], [120,5], [2,1]]


# get Bilinear model
projection_regularizer = None
projection_constraint = keras.constraints.max_norm(3.0,axis=0)
attention_regularizer = None
attention_constraint = keras.constraints.max_norm(5.0, axis=1)
dropout = 0.1


model = Models.TABL(template, dropout, projection_regularizer, projection_constraint,
                    attention_regularizer, attention_constraint)
model.summary()

# create class weight
class_weight = {0 : 1e6/300.0,
                1 : 1e6/400.0,
                2 : 1e6/300.0}


# training          # remove .iloc[2:] for single day
model.fit(X_train, lbls_train.iloc[2:], batch_size=256, epochs=100, class_weight=class_weight)

model.save('model.h5')

pred = model.predict(X_train)

total = pd.concat((stock_train, stock_test), axis=0)
inputs = total[len(total) - len(stock_test) - n_past:].values
inputs = sc_predict.transform(inputs)
add = np.zeros((n_past + (n_future - 1), stock.shape[1]))
inputs = np.vstack((inputs, add))

X_test = []
for i in range(n_past, len(inputs) - n_future + 1):
    X_test.append(inputs[i - n_past:i, 0:stock.shape[1]])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], stock.shape[1]))


pred2 = model.predict(X_test)
pred2 = pred2[n_past:]

for i in range(0, pred2.shape[0]):
    for j in range(0, pred2.shape[1]):
        if pred2[i][j] > 0.5:
            pred2[i][j] = 1
        else:
            pred2[i][j] = 0
  
pred2_ = np.array(pd.DataFrame(pred2).iloc[:-2])
lbls_test_ = lbls_test.iloc[2:]
                                 # remove .iloc[2:] for single day
acc = np.where(pred2_ == np.array(lbls_test_), 1, 0)

acc1 = acc[:,0]
acc2 = acc[:,1]

print('OOS Up Accuracy: ' + str(round((list(acc1).count(1) / (len(acc1)-1))*100, 2)) + '%')
print('OOS Down Accuracy: ' + str(round((list(acc2).count(1) / (len(acc2)-1))*100, 2)) + '%')
