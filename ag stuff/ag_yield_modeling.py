import os
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np 
import datetime as dt
import xgboost as xgb
from matplotlib.offsetbox import AnchoredText
import heapq
import math 

Path = "ag stuff"
os.chdir(Path)

""" 
Helpful Links:
https://geo.fas.usda.gov/GADAS/index.html#
https://glam1.gsfc.nasa.gov/
https://www.mdpi.com/2072-4292/13/21/4227
https://glam1.gsfc.nasa.gov/api/doc/db/versions#default-db
"""

def initialize_df(df, cols, week, start_yr, suffix, var):
    temp = df[df["end_code"].eq(week) & (df["short_desc"].eq(var))]
    temp = temp[temp["location_desc"].isin(cols)]
    temp = temp[temp["year"].ge(start_yr)]
    temp.reset_index(inplace=True)
    temp = temp.pivot_table(index='year', columns='location_desc', values="Value")
    temp = temp.add_suffix(suffix)
    return temp 


"""
Get crop mask from: https://glam1.gsfc.nasa.gov/api/doc/db/versions#default-db
"""
def get_ndvi(version, mask, start_month, num_months, ts_type='seasonal', mcv=None, ids=27258):
    df = pd.read_csv(f"https://glam1.gsfc.nasa.gov/api/gettbl/v4?sat=MOD&version={version}&layer=NDVI&mask={mask}&shape=ADM&ids={ids}&ts_type={ts_type}&start_month={start_month}&num_months={num_months}&format=csv", skiprows=14).ffill()
    df = df[df["ORDINAL DATE"].ge("2000-01-01") & df["ORDINAL DATE"].lt("2024-01-01")]
    return df 


"""
The model runs one default variable, specified by the overlay_var. This can be "maturity" or "harvested" depending on the crop.
Pass array of 1s and 0s indicating which additional variables to include.

Below are the available variables, in order:
1. PCT Excellent at current week
2. PCT Good at current week
3. PCT Change in Excellent (pct @ week - pct @ week - 10)
4. PCT Change in Good (pct @ week - pct @ week - 10)
5. Acres planted/year
6. Drought data for basket of states
"""


def get_data(crop, week, overlay_var, start_yr=2000, end_yr=2023, var_arr=[]):
    

    df = get_ndvi("v15", "USDA-NASS-CDL_2018-2023_corn-50pp", 1, 12)
    
    idx = list()
    for i in range(end_yr - start_yr + 1):
        idx.append(start_yr + i)
    NDVI = list()

    # Data taken from USDA Quickstats

    dropped_cols = ["Program", "Week Ending", "Ag District", "Ag District Code", "Watershed", "Commodity", "Data Item", "watershed_code", "County", "County ANSI", "Zip Code", "Region", "Domain", "Domain Category", "CV (%)"]
    yields=pd.read_csv(f"data/annual_{crop}_yields.csv")
    yields.drop(columns=dropped_cols, inplace=True)
    yields = yields.iloc[::-1]
    yields.reset_index(inplace=True, drop=True)

    overlay = pd.read_csv(f"data/{crop}_{overlay_var}_wk{week}.csv")
    overlay.drop(columns=dropped_cols, inplace=True)
    overlay = overlay.iloc[::-1]
    overlay.reset_index(inplace=True, drop=True)
    overlay = overlay.pivot(index="Year", columns="State", values="Value")
    overlay = overlay.add_suffix(" MATURITY")
    
    progress_hist = pd.read_csv("data/progress_hist.csv")
    additional_vars = {
                    0: f"{crop} - CONDITION, MEASURED IN PCT EXCELLENT",
                    1: f"{crop} - CONDITION, MEASURED IN PCT GOOD",
                    2: f"{crop} - CONDITION, MEASURED IN PCT EXCELLENT",
                    3: f"{crop} - CONDITION, MEASURED IN PCT GOOD",
                    4: "PCT CHNGE EXCELLENT",
                    5: "PCT CHNGE GOOD",
                    6: "Acres planted/year",
                    7: "Drought data" 
    }


    if (crop == "CORN"):
        states = ["WISCONSIN", "SOUTH DAKOTA", "OHIO", "NEBRASKA", "MISSOURI", "MINNESOTA", "KANSAS", "IOWA", "INDIANA", "ILLINOIS"]
        threshold = 0.56
    elif (crop == "SOYBEAN"):
        threshold = 0.58
        states = ["SOUTH DAKOTA", "OHIO", "NORTH DAKOTA", "NEBRASKA", "MISSOURI", "MINNESOTA", "IOWA", "INDIANA", "ILLINOIS", "ARKANSAS"]
    elif (crop == "WHEAT, WINTER"):
        threshold = 0.3
        states = ["WASHINGTON", "SOUTH DAKOTA", "OKLAHOMA", "NEBRASKA", "MONTANA", "MISSOURI", "KANSAS", "ILLINOIS", "IDAHO", "COLORADO"]

    y = pd.DataFrame(columns=states)
    cols = states

    for i in range(6):
        if var_arr[i] == 1:
            if i == 2:
                excellent_24 = initialize_df(progress_hist, cols, week - 10, start_yr, " EXCELLENT PCT CHNGE", additional_vars[2])
                excellent_34 = initialize_df(progress_hist, cols, week, start_yr, " EXCELLENT PCT CHNGE", additional_vars[2])
                pct_chnge_excellent = excellent_34 - excellent_24
                overlay = pd.concat([overlay, pct_chnge_excellent], axis=1)
            elif i == 3:
                good_24 = initialize_df(progress_hist, cols, week - 10, start_yr, " GOOD PCT CHNGE", additional_vars[1])
                good_34 = initialize_df(progress_hist, cols, week, start_yr, " GOOD PCT CHNGE", additional_vars[1])
                pct_chnge_good = good_34 - good_24
                overlay = pd.concat([overlay, pct_chnge_good], axis=1)
            elif i == 4:
                planted = pd.read_csv(f"data/annual_{crop}_planted.csv")
                planted.drop(columns=dropped_cols, inplace=True)
                planted = planted.iloc[::-1]
                planted.reset_index(inplace=True, drop=True)
                planted = planted.pivot(index="Year", columns="State", values="Value")
                planted = planted.add_suffix(" Planted")
                overlay = pd.concat([overlay, planted], axis=1)
            elif i == 5:
                drought_index = pd.read_csv(f"data/{crop}_Drought.csv", index_col=0, parse_dates=True)
                drought_index = drought_index.loc[:, ["D0", "D1", "D2", "D3", "D4", "None"]].iloc[::-1].reset_index()
                drought_index['MapDate'] = drought_index['MapDate'].dt.year
                drought_index = drought_index.groupby("MapDate", as_index=True, sort=False)[["D0", "D1", "D2", "D3", "D4", "None"]].mean()
                drought_index.index.name = None
                drought_index.drop(index=2024, inplace=True)
                overlay = pd.concat([overlay, drought_index], axis=1)
            elif i == 0:
                temp = initialize_df(progress_hist, cols, week, start_yr, " EXCELLENT", additional_vars[i])
                overlay = pd.concat([overlay, temp], axis=1)   
            else:
                temp = initialize_df(progress_hist, cols, week, start_yr, " GOOD", additional_vars[i])
                overlay = pd.concat([overlay, temp], axis=1)



    count=0
    for i in range(df.shape[0]):
        if (int(df.loc[i, "ORDINAL DATE"][-3:]) == 329):
            NDVI.append(count)
            count=0
        elif (df.loc[i, "SAMPLE VALUE"] >= threshold): 
            count+=df.loc[i, "SAMPLE VALUE"]   
    
    for i in y.columns:
        y[i] = yields[yields["State"] == i]["Value"].values
    
    x = pd.concat([pd.DataFrame(NDVI, columns=["NDVI"], index=idx), overlay], axis=1)
    return y, x

# Takes "CORN" or "WHEAT, WINTER". Harvest weeks are 34/35/36 and 31, respectively. 
# Overlay variable, either "harvested" or "maturity"
y, x = get_data("CORN", 34, "maturity", 2000, 2023, [1, 1, 1, 1, 1, 1])


vhi_links = ["https://www.star.nesdis.noaa.gov/smcd/emb/vci/VH/get_TS_admin.php?provinceID=14&country=USA&yearlyTag=Yearly&type=Parea_VHI&TagCropland=MAIZ&year1=1982&year2=2024",
             "https://www.star.nesdis.noaa.gov/smcd/emb/vci/VH/get_TS_admin.php?provinceID=15&country=USA&yearlyTag=Yearly&type=Parea_VHI&TagCropland=MAIZ&year1=1982&year2=2024",
             "https://www.star.nesdis.noaa.gov/smcd/emb/vci/VH/get_TS_admin.php?provinceID=16&country=USA&yearlyTag=Yearly&type=Parea_VHI&TagCropland=MAIZ&year1=1982&year2=2024",
             "https://www.star.nesdis.noaa.gov/smcd/emb/vci/VH/get_TS_admin.php?provinceID=17&country=USA&yearlyTag=Yearly&type=Parea_VCI&TagCropland=MAIZ&year1=1982&year2=2024",
             "https://www.star.nesdis.noaa.gov/smcd/emb/vci/VH/get_TS_admin.php?provinceID=24&country=USA&yearlyTag=Yearly&type=Parea_VHI&TagCropland=MAIZ&year1=1982&year2=2024",
             "https://www.star.nesdis.noaa.gov/smcd/emb/vci/VH/get_TS_admin.php?provinceID=26&country=USA&yearlyTag=Yearly&type=Parea_VHI&TagCropland=MAIZ&year1=1982&year2=2024",
             "https://www.star.nesdis.noaa.gov/smcd/emb/vci/VH/get_TS_admin.php?provinceID=28&country=USA&yearlyTag=Yearly&type=Parea_VHI&TagCropland=MAIZ&year1=1982&year2=2024",
             "https://www.star.nesdis.noaa.gov/smcd/emb/vci/VH/get_TS_admin.php?provinceID=36&country=USA&yearlyTag=Yearly&type=Parea_VHI&TagCropland=MAIZ&year1=1982&year2=2024",
             "https://www.star.nesdis.noaa.gov/smcd/emb/vci/VH/get_TS_admin.php?provinceID=42&country=USA&yearlyTag=Yearly&type=Parea_VHI&TagCropland=MAIZ&year1=1982&year2=2024",
             "https://www.star.nesdis.noaa.gov/smcd/emb/vci/VH/get_TS_admin.php?provinceID=50&country=USA&yearlyTag=Yearly&type=Parea_VHI&TagCropland=MAIZ&year1=1982&year2=2024"]


# Extra VHI data
def get_vhi_data(links):
    for i in range(len(links)):
        global x 
        vhi = pd.read_csv(links[i], skiprows=3)
        vhi = vhi.iloc[list(range(17, 41)), :]
        vhi.reset_index(inplace=True)
        vhi.set_index(vhi.columns[1], drop=True, inplace=True)
        vhi.drop(columns="index", inplace=True)
        vhi.index.name = None
        VHI_bins = list(range(0, 105, 5))
        vhi.drop(columns="Unnamed: 22", inplace=True)
        vhi.columns = VHI_bins
        vhi = vhi.add_prefix(y.columns[i] + " VHI Bin")
        vhi.index = x.index 
        x = pd.concat([x, vhi], axis=1)

x.fillna(0, inplace=True)
cols = x.columns
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
x = pd.DataFrame(x, columns=cols)


fig, axs = plt.subplots(5, 2)


def feature_importance(model, df, ax, num_features, plot=False):
        importance = model.coef_
        keys = list(df.keys())
        weights = {}
        for i,v in enumerate(importance):
            print("Feature: %s, Score: %.5f" % (keys[i],v))
            weights[keys[i]] = abs(v)
        if plot:
            ax.bar([x for x in range(len(importance))], height=importance, color='b')
            tickvals = range(0, len(importance))
            cols = df.columns
            ax.set_xticks(ticks=tickvals, labels=cols, rotation=45, fontsize='xx-small', fontstretch='extra-condensed')
        
        largest_features = heapq.nlargest(num_features, weights, key=weights.get)
        return largest_features


"""
Hybrid model of ridge on training data and gradient boosted regressor on residuals
Will need to change params/models if using different framework.
"""

def get_best_model(X, y, ridge=True):
    xgb_params = {
    "learning_rate": np.arange(0.001, 0.2, 0.01),
    "gamma": range(0, 8),
    "n_estimators": range(800, 1200),
    "max_depth": range(1, 12),
    "reg_lambda": np.arange(0.1, 10, 0.1),
    "colsample_bynode": np.arange(0.1, 1, 0.1)
    }

    ridge_params = {
        "alpha": np.arange(0, 5, 0.1),
    }

    if not ridge:
        optimized_model = RandomizedSearchCV(param_distributions=xgb_params, estimator=xgb.XGBRegressor(), scoring='neg_mean_squared_error', verbose=1, random_state=42)
        optimized_model.fit(X, y)
        print("Best Parameters:", optimized_model.best_params_)
    else: 
        optimized_model = RandomizedSearchCV(param_distributions=ridge_params, estimator=Ridge(solver='svd'), scoring='neg_mean_squared_error', verbose=1, random_state=42)
        optimized_model.fit(X, y)
        print("Best Parameters:", optimized_model.best_params_)
    return optimized_model



def reg_plot(x, y, title, ax=None):
    X_train = x.iloc[0:15]
    X_test = x.iloc[15:20]
    X_cv = x.iloc[20:]
    y_train = y.iloc[0:15]
    y_test = y.iloc[15:20]
    y_cv = y.iloc[20:]


    model = Ridge(solver="svd", alpha=1.3)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    weighted_features = feature_importance(model, X_train, ax, math.ceil(X_train.shape[1]/2))

    residuals = y_train - y_train_pred[0]
    residuals = pd.DataFrame(residuals, index=y_train.index)

    kwargs = {
        "monotone_constraints": "(1, 1, 1, -1, 1, 1)"
    }

    resid_model = xgb.XGBRegressor()
    resid_model.fit(X_train.loc[:, weighted_features], residuals)



    ax.scatter(y_train.index, y_train)
    ax.plot(y_train.index, y_train_pred)
    pred2 = resid_model.predict(X_test.loc[:, weighted_features]) + y_test_pred
    pred2 = pd.DataFrame(pred2, index=y_test.index)
    ax.scatter(y_test.index, y_test)
    ax.plot(y_test.index, pred2)
    
    resid_cv = y_test - y_test_pred
    resid_cv = pd.DataFrame(resid_cv, index=y_test.index)

    resid_cv_model = xgb.XGBRegressor()
    resid_cv_model.fit(X_test.loc[:, weighted_features], resid_cv)
    resid_cv_pred = resid_cv_model.predict(X_cv.loc[:, weighted_features])
    pred3 = model.predict(X_cv)

    pred3 = resid_cv_pred + pred3
    pred3 = pd.DataFrame(pred3, index=y_cv.index)
    ax.scatter(y_cv.index, y_cv)
    ax.plot(y_cv.index, pred3)

    ax.set_title(title)

    s = np.sqrt(mean_squared_error(y_cv, pred3))
    at = AnchoredText(
        f"{s:.2f}",
        prop=dict(size="large"),
        frameon=True,
        loc="upper left",
    )
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)

    return ax 

for i, ax in enumerate(fig.axes):
    col_name = y.columns[i]
    state_data = x.filter(regex=col_name)
    ndvi_data = x.loc[:, "NDVI"]
    drought_data = x.loc[:, ["D1", "D2", "D3", "D4"]]
    input = pd.concat([ndvi_data, state_data], axis=1, ignore_index=False)
    input = pd.concat([input, drought_data], axis=1, ignore_index=False)

    reg_plot(input, y.iloc[:, i], col_name, ax)

plt.subplots_adjust(hspace=0.5)
plt.show()







