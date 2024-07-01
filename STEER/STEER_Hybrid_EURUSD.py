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


# 2y swaps, relative yield curve steepness or CDS, local equity performance, global equity performance, commodity prices

def steer_ols(swaps="", FX="", local_equity="", yc="", global_eq="", cb="", days=378, weights=[1], G10=False, US=False):
    dates = [(dt.date.today()+dt.timedelta(days=-days)).strftime("%Y%m%d"), dt.date.today().strftime("%Y%m%d")]
    swaps_temp = Bloomberg(swaps).fetch_timeseries_data(["px last"], dates, col_names = swaps, reshape=True, dropna=False, resample=False).ffill()
    fx = Bloomberg([FX]).fetch_timeseries_data(["px last"], dates, col_names = FX, reshape=True, dropna=False, resample=False).ffill()
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
        for i in range (1, len(weights)):
            basket += basket_temp.iloc[:, i] * weights[i]
    else: 
        basket = basket_temp.iloc[:, 0] * weights[0]
    
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
    fx = X.loc[:, "E"].to_frame()
    X.drop(columns=["E"], inplace=True)


    # Preprocessing

    poly = PolynomialFeatures(degree=2, include_bias=True)
    scaler = StandardScaler()

    X = poly.fit_transform(X)
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, index=fx.index)

    # LinReg

    model = LinearRegression(fit_intercept=True)
    model.fit(X, fx)
    y_pred = model.predict(X)
    y_pred = pd.DataFrame(y_pred, index=X.index)


    ax.plot(y_pred, label="Predicted STEER")
    ax.plot(fx, color="red", label="Spot")
    ax.set_ylabel("FX Rate")

    corr = model.score(X, fx)

    at = AnchoredText(
        f"{corr:.4f}",
        prop=dict(size="large"),
        frameon=True,
        loc="upper left",
    )
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)

    ax.legend(loc=1)


    plt.savefig("EURUSD_STEER.png")
    plt.show()  




steer_ols(swaps=["EUSA2 Curncy", "USSO2 Curncy"], FX="EURUSD Curncy", yc=["GTEUR10Y Govt", "EESWEC Curncy", "USGG10YR Index", "USSOC BGN Curncy"], local_equity="VTI US Equity", global_eq="MXWO Index", cb=["CL1 COMB Comdty"], weights=[1], G10=True)