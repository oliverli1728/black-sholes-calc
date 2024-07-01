import os
import sys
import joblib
import shutil
from tempfile import mkdtemp
temp_dir = mkdtemp()
import pdblp
import blp

import pandas as pd
import numpy as np
import datetime as dt
import warnings
warnings.filterwarnings("ignore")

class Bloomberg(object):

    def __init__(self, tickers):
        self.tickers = tickers


    # get data from PDBLP efficiently
    def fetch_timeseries_data(self, fields:list, dates:list, col_names:list, reshape=True, dropna=True, resample=True):
        
        con = pdblp.BCon(debug=False, port=8194, timeout=5000)
        con.start()
        cacher = joblib.Memory(temp_dir)
        bdh = cacher.cache(con.bdh, ignore=['self'])
        
        df = bdh(self.tickers, fields, dates[0], dates[-1], longdata=True)
        if reshape is True:
            df = df[['date', 'ticker', 'value']].pivot(columns='ticker', index='date').droplevel(0, axis=1).reset_index().set_index("date")
            col_names = dict(zip(self.tickers, col_names))
            df.columns = df.columns.map(col_names)
            if resample is True:
                df = df.resample('D').asfreq().ffill()
            else: pass
                
            if dropna is True:
                df.dropna(inplace=True)
            else: pass
        else: pass
        
        return df
    
    def fetch_reference_data(self, fields:list):
        
        con = pdblp.BCon(debug=False, port=8194, timeout=5000)
        con.start()
        cacher = joblib.Memory(temp_dir)
        bdp = cacher.cache(con.ref, ignore=['self'])
        
        df = bdp(self.tickers, fields)
        
        return df
    
# working on function to use bql
    def fetch_bql_data(self, tickers:list, get_query_syntax:str, date_range:str,freq:str, reshape=True):
        bq = blp.BlpQuery().start().bql
        df = bq(expression=f"get(dropna(px_last(dates=range(-365d,0d),frq=m))) for(['{key}']) with(mode=cached)").dropna()[["security","secondary_value","value"]]
        df.columns = ["ticker","date", f"{key.split()[1]}"]
        df.date = pd.to_datetime(_df.date)
        df.date = [x.strftime("%Y-%m-%d") for x in df.date]
        if reshape is True:
            df = df.pivot(columns="ticker", index='date')
        else: pass

        return df
        
# def fetch_timeseries_data(tickers:list, fields:list, dates:list, col_names:list, reshape=True, dropna=True, resample=True):
        
#     con = pdblp.BCon(debug=False, port=8194, timeout=5000)
#     con.start()
#     cacher = joblib.Memory(temp_dir)
#     bdh = cacher.cache(con.bdh, ignore=['self'])
    
#     df = bdh(tickers, fields, dates[0], dates[-1], longdata=True)
#     if reshape is True:
#         df = df[['date', 'ticker', 'value']].pivot(columns='ticker', index='date').droplevel(0, axis=1).reset_index().set_index("date")
#         col_names = dict(zip(tickers, col_names))
#         df.columns = df.columns.map(col_names)        
#         if dropna is True:
#             df.dropna(inplace=True)
#         else: pass
            
#         if resample is True:
#             df = df.resample('D').asfreq().ffill()
#         else: pass
#     else: pass
        
#     return df
