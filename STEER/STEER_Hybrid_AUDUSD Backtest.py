from BloombergAPI import Bloomberg
import xgboost as xgb
import math
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from matplotlib.offsetbox import AnchoredText
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
import statsmodels.tsa.stattools as ts 
import backtrader as bt
from backtesting import Backtest, Strategy



# 2y swaps, relative yield curve steepness or CDS, local equity performance, global equity performance, commodity prices

def steer_ols(swaps="", FX="", local_equity="", yc="", global_eq="", cb="", days=390, weights=[1], G10=False, US=False):
    dates = [(dt.date.today()+dt.timedelta(days=-days)).strftime("%Y%m%d"), dt.date.today().strftime("%Y%m%d")]
    swaps_temp = Bloomberg(swaps).fetch_timeseries_data(["px last"], dates, col_names = swaps, reshape=True, dropna=False, resample=False).ffill()
    fx = Bloomberg([FX]).fetch_timeseries_data(["px last"], dates, col_names = FX, reshape=True, dropna=False, resample=False).ffill()
    fx2 = Bloomberg([FX]).fetch_timeseries_data(["px open"], dates, col_names = FX, reshape=True, dropna=False, resample=False).ffill()
    fx3 = Bloomberg([FX]).fetch_timeseries_data(["px high"], dates, col_names = FX, reshape=True, dropna=False, resample=False).ffill()
    fx4 = Bloomberg([FX]).fetch_timeseries_data(["px low"], dates, col_names = FX, reshape=True, dropna=False, resample=False).ffill()

    fx.rename(columns={"U": "Close"}, inplace=True)
    fx2.rename(columns={"U": "Open"}, inplace=True)
    fx3.rename(columns={"U": "High"}, inplace=True)
    fx4.rename(columns={"U": "Low"}, inplace=True)
    if (G10):
        yc_cds_temp = Bloomberg(yc).fetch_timeseries_data(["px last"], dates, col_names = yc, reshape=True, dropna=False, resample=False).ffill()

    else:
        yc_cds_temp = Bloomberg([yc]).fetch_timeseries_data(["px last"], dates, col_names = yc, reshape=True, dropna=False, resample=False).ffill()
    local_index = Bloomberg([local_equity]).fetch_timeseries_data(["px last"], dates, col_names = local_equity, reshape=True, dropna=False, resample=False).ffill()
    MSCI = Bloomberg([global_eq]).fetch_timeseries_data(["px last"], dates, col_names = global_eq, reshape=True, dropna=False, resample=False).ffill()
    basket_temp =  Bloomberg(cb).fetch_timeseries_data(["px last"], dates, col_names = cb, reshape=True, dropna=False, resample=False).ffill()
    basket_temp.interpolate(method='time')

    if (G10):
        yc_cds = pd.DataFrame()

        # Need to change these numbers based on order of ticker symbols
        yc_cds["First Spread"] = yc_cds_temp.iloc[:, 1] - yc_cds_temp.iloc[:, 0]
        yc_cds["Second Spread"] = yc_cds_temp.iloc[:, 2] - yc_cds_temp.iloc[:, 3]
        yc = yc_cds.loc[:, "First Spread"] - yc_cds.loc[:, "Second Spread"]
        yc = yc.to_frame()
        yc.rename(columns={0: "Differential 10y/3M Spread"}, inplace=True)
        
        yc_cds_temp = yc

        basket = basket_temp.iloc[:, 0] * weights[0]

    else: 
        basket = basket_temp.iloc[:, 0] * weights[0]
        basket = basket_temp.iloc[:, 0] * weights[0]
        for i in range (1, len(weights)):
            basket += basket_temp.iloc[:, i] * weights[i]
    
    swaps = swaps_temp.iloc[:, 0] - swaps_temp.iloc[:, 1]
    swaps = swaps.to_frame()
    swaps.rename(columns={0: "IR Differential"}, inplace=True)

        
    basket = basket.to_frame()
    basket.rename(columns={0: "basket"}, inplace=True)


    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    X = basket.join(swaps, how="left")
    X = X.join(yc_cds_temp, how="left")
    X = X.join(local_index, how="left")
    X = X.join(MSCI, how="left")
    X = X.join(fx, how="left")
    
    X.ffill(inplace=True)
    fx = X.loc[:, "Close"].to_frame()
    X.drop(columns=["Close"], inplace=True)


    # Preprocessing

    poly = PolynomialFeatures(degree=1, include_bias=True)
    scaler = StandardScaler()

    X = poly.fit_transform(X)
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, index=fx.index)

    # LinReg

    model = LinearRegression(fit_intercept=True)
    model.fit(X, fx)
    y_pred = model.predict(X)
    y_pred = pd.DataFrame(y_pred, index=X.index)

    resid = fx.iloc[:, 0] - y_pred.iloc[:, 0]
    resid = resid.to_frame()
    resid.rename(columns={0: "Resid"}, inplace=True)
    mean = resid.loc[:, "Resid"].mean()
    std = resid.loc[:, "Resid"].std()
    z = (resid.iloc[:, 0] - mean) / std
    z = z.to_frame()


    y = y_pred.join(fx, how="left")
    y.rename(columns={0: "yhat", "Close": "fx"}, inplace=True)

    positive_z = z.index[z["Resid"] >= 1.5].tolist()
    negative_z = z.index[z["Resid"] <= -1.5].tolist()


    zeros = pd.DataFrame(0, index=y_pred.index, columns=["Signal"])
    df = fx2.join(fx3, how="left")
    df = df.join(fx4, how="left")
    df = df.join(fx, how="left")


    df = df.join(zeros, how="left")
    df.loc[positive_z, "Signal"] = 1
    df.loc[negative_z, "Signal"] = -1

    return df, z, y

df, z, y = steer_ols(swaps=["USSO2 Curncy", "ADSO2 Curncy"], FX="USDAUD Curncy", yc=["GTAUD10YR Corp", "ADSOC Curncy", "USGG10YR Index", "USSOC BGN Curncy"], local_equity="ASX Index", global_eq="MXWO Index", cb=["CL1 COMB Comdty"], weights=[1], G10=True)

df["PCT"] = (y.loc[:, "yhat"] - y.loc[:, "fx"])/y.loc[:, "fx"]
df = df.interpolate(method='time')
df["Index"] = df.index

from datetime import datetime
from datetime import date
date_format = "%Y-%m-%d"


class STEER(Strategy):

    def init(self):
        pass
    def next(self):
        current_signal = self.data.Signal[-1]
        current_price = self.data.Close[-1]
        current_pct = self.data.PCT[-1]
        
        trades = self.trades
        def parse_trades(trades):
            for x in trades:
                a = datetime.strptime(datetime.strftime(x.entry_time, date_format), date_format)
                b = np.datetime_as_string(self.data.Index[-1])
                d = b[:10]
                c = datetime.strptime(d, date_format)

                delta = c - a
              
                if delta.days > 20:
                    x.close()


        parse_trades(trades)
        if current_signal == 1 and abs(current_pct) > 0.01:
        
            self.sell(size=-1000, sl = current_price + 0.5 * current_price * (current_pct))
        
        elif current_signal == -1 and abs(current_pct) > 0.01:
            
            self.buy(size=1000, sl = current_price - 0.5 * current_price * (current_pct))

bt = Backtest(df, STEER, cash=10000)
stats = bt.run()
print(stats)
bt.plot()


