from BloombergAPI import Bloomberg
import xgboost as xgb
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from matplotlib.offsetbox import AnchoredText
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
import statsmodels.tsa.stattools as ts 
import plotly.express as px


# 2y swaps, relative yield curve steepness or CDS, local equity performance, global equity performance, commodity prices

def steer_ols(swaps="", FX="", local_equity="", yc="", global_eq="", cb="", days=379, weights=[]):
    dates = [(dt.date.today()+dt.timedelta(days=-days)).strftime("%Y%m%d"), dt.date.today().strftime("%Y%m%d")]
    swaps_temp = Bloomberg(swaps).fetch_timeseries_data(["px last"], dates, col_names = swaps, reshape=True, dropna=False, resample=False).ffill()
    fx = Bloomberg([FX]).fetch_timeseries_data(["px last"], dates, col_names = FX, reshape=True, dropna=False, resample=False).ffill()
    yc_cds = Bloomberg([yc]).fetch_timeseries_data(["px last"], dates, col_names = yc, reshape=True, dropna=False, resample=False).ffill()
    local_index = Bloomberg([local_equity]).fetch_timeseries_data(["px last"], dates, col_names = local_equity, reshape=True, dropna=False, resample=False).ffill()
    MSCI = Bloomberg([global_eq]).fetch_timeseries_data(["px last"], dates, col_names = global_eq, reshape=True, dropna=False, resample=False).ffill()
    basket_temp =  Bloomberg(cb).fetch_timeseries_data(["px last"], dates, col_names = cb, reshape=True, dropna=False, resample=False).ffill()
    basket_temp.interpolate(method='time')

    swaps = swaps_temp.iloc[:, 0] - swaps_temp.iloc[:, 1]
    swaps = swaps.to_frame()
    swaps.rename(columns={0: "IR Differential"})

    basket = basket_temp.iloc[:, 0] * weights[0]
    for i in range (1, len(weights)):
        basket += basket_temp.iloc[:, i] * weights[i]
        
    basket = basket.to_frame()
    basket.rename(columns={0: "basket"})
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    X = basket.join(swaps, how="left")
    X = X.join(yc_cds, how="left")
    X = X.join(local_index, how="left")
    X = X.join(MSCI, how="left")
    X = X.join(fx, how="left")
    X.ffill(inplace=True)
    fx = X.loc[:, "U"].to_frame()
    X.drop(columns=["U"], inplace=True)
    X.rename(columns={0: "A"}, inplace=True)

    poly = PolynomialFeatures(degree=2, include_bias=True)
    scaler = StandardScaler()

    X = poly.fit_transform(X)
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, index=fx.index)

    # X.rename(columns={0: "Intercept", 1: "Cmdty Basket", 2: "2y Swaps", 3: "3m/10y Spread", 4: "Local Index", 5: "Global Index"}, inplace=True)

    model = LinearRegression()
    model.fit(X, fx)
    y = model.predict(X)
    y = pd.DataFrame(y, index=X.index)
    ax.plot(y, label="FX Rate")
    ax.set_title("STEER vs Spot")
    ax.plot(fx, color="red", label="Spot")
    ax.set_ylabel("FX Rate")

    corr=model.score(X, fx)

    ax.legend(loc=1)

    at = AnchoredText(
        f"{corr:.4f}",
        prop=dict(size="large"),
        frameon=True,
        loc="upper left",
    )
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)



    plt.savefig("USDZAR_STEER.png")
    plt.show()



    



steer_ols(swaps=["USOSFR3 Curncy", "SASW3 Curncy"], FX="USDZAR Curncy", yc="CSOAF1U5 Curncy", local_equity="JALSH Index", global_eq="MXWO Index", cb=["XAU Curncy", "WBCOIRON Index", "XPT Curncy", "WBCOCSAF Index"], weights=[0.232, 0.232, 0.188, 0.235])