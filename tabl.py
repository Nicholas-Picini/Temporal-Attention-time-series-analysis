import Models
import keras
import talib
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from datetime import datetime
from fancyimpute import KNN
from statsmodels.tsa.arima_model import ARIMA

np.random.seed(123)


ticker = 'AAPL'
stock = pdr.get_data_yahoo(ticker.upper(), start='2009-01-14', end=str(datetime.now().date()))

# VIX Data
from datapackage import Package
package = Package('https://datahub.io/core/finance-vix/datapackage.json')

for resource in package.resources:
    if resource.descriptor['datahub']['type'] == 'derived/csv':
        vix = pd.DataFrame(resource.read())
vix = vix.iloc[1267:,:]
vix = vix.set_index(vix.iloc[:,0], drop=True)
vix = vix[4]

stock = pd.concat([stock, vix], axis=1)


up = pd.DataFrame(data={"UP":np.where(stock["Close"].shift(-1) > stock["Close"], 1, 0)})
dn = pd.DataFrame(data={"DN":np.where(stock["Close"].shift(-1) < stock["Close"], 1, 0)})
lbls = up.join(dn)

lbls_train = lbls.iloc[0:round(len(lbls)*0.8),:]
lbls_test = lbls.iloc[round(len(lbls)*0.8):,:]


cols = list(stock)[0:7]
 
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
arima = np.array(arima)


stock = stock.join(HT)
stock = stock.join(wma)
stock = stock.join(rsi)
stock = stock.join(mavp)
stock = stock.join(upperband)
stock = stock.join(middleband)
stock = stock.join(lowerband)
stock = stock.join(Roc)
stock = stock.join(Atr)


# Impute missing values using KNN
stock = stock.as_matrix() 
stock = np.append(stock, arima, 1)
stock = KNN(k=7).fit_transform(stock)

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

n_future = 1  # Number of time steps predicted
n_past = 20  # Number of time steps used in preditcion

for i in range(n_past, len(training_set_scaled) - n_future + 1):
    X_train.append(training_set_scaled[i - n_past:i, 0:17])
    y_train.append(training_set_scaled[i+n_future-1:i + n_future, 3])

X_train = np.array(X_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 17))

lbls_train = lbls_train.iloc[:-20]


# 1 hidden layer network with input: 20x10, hidden 120x5, output 3x1
template = [[20,17], [120,5], [2,1]]


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


# training
model.fit(X_train, lbls_train, batch_size=256, epochs=100, class_weight=class_weight)

pred = model.predict(X_train)

total = pd.concat((stock_train, stock_test), axis=0)
inputs = total[len(total) - len(stock_test) - 20:].values
inputs = sc_predict.transform(inputs)
add = np.zeros((n_past, 17))
inputs = np.vstack((inputs, add))

X_test = []
for i in range(n_past, len(inputs) - n_future + 1):
    X_test.append(inputs[i - n_past:i, 0:17])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 17))


pred2 = model.predict(X_test)
pred2 = pred2[20:]

for i in range(0, pred2.shape[0]):
    for j in range(0, pred2.shape[1]):
        if pred2[i][j] > 0.5:
            pred2[i][j] = 1
        else:
            pred2[i][j] = 0

acc = np.where(pred2 == np.array(lbls_test), 1, 0)
acc1 = acc[:,0]
acc2 = acc[:,1]

print('OOS Up Accuracy: ' + str(round((list(acc1).count(1) / len(acc1))*100, 2)) + '%')
print('OOS Down Accuracy: ' + str(round((list(acc2).count(1) / len(acc2))*100, 2)) + '%')
