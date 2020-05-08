# Temporal-Attention-time-series-analysis
A Temporal Attention augmented Bilinear Network for Directional Prediction of Financial Time Series

A bilinear network with temporal attention proposed by Dat Thanh Tran, Alexandros Iosifidis, Juho Kanniainen, and Moncef Gabbouj 
in their insightful paper:

[Temporal Attention-Augmented Bilinear Network for Financial Time-Series Data Analysis](https://ieeexplore.ieee.org/document/8476227)

This work presents my modification of their original program ([which can be found here](https://github.com/viebboy/TABL)) to work with daily stock data as opposed to limit order book information they use in their paper.

A number of features are generated for the stock including RSI, Bollinger Bands, Moving-Average with Variable Period, ROC, ATR, Weighted Moving-Average, Daily VIX, Hilbert-Transform Trendmode, and ARIMA forecast. The network topology will automatically update to the appropriate dimension for any added feature. 
