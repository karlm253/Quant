# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 01:11:06 2024

@author: Heavens Base
"""
import yfinance as yf
from Fred_API import fred_get_series
from Edgar_Company_Facts import get_cik, get_financial_data
from prophet import Prophet
import pandas as pd
from pandas_datareader import data as pdr
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
yf.pdr_override()

#Variables
stock = 'CTAS'
end = dt.datetime.now() 
start = dt.datetime(2016,1,1)

#Get Data
def get_yf(stock):
    df = pdr.get_data_yahoo(stock,start,end)
    df = df.reindex(pd.date_range(start, end, freq='D')).dropna(axis=0, how = 'any')
    return df
def get_edgar(stock):
    cik = get_cik(stock)
    list_df_edgar = get_financial_data(cik)
    df_ltm = list_df_edgar[2].dropna(subset='Revenues')
    df_quarter = list_df_edgar[0].dropna(subset='Revenues')
    return df_ltm, df_quarter
def chart_prophet(df,sensitivity=0.05):
    #Prepare Data: columns must be ds and y
    series = df.rename(columns={'date': 'ds', 'Close': 'y'})
    train, val = series[:-4], series[-4:]
    
    #Create and Fit the Prophet Model:
    model = Prophet()
    model.fit(series)
    m = Prophet(changepoint_prior_scale=sensitivity) #Number is Trend Flexibility to Detect Breaks, Default is 0.05
    future = model.make_future_dataframe(periods=8, freq='Q')  # Adjust the number of periods as needed
    
    #Forecast
    forecast = model.predict(future)
    fig = model.plot(forecast)
    plt.suptitle(stock+title)
    plt.show()
    
    #Trend Changepoints
    from prophet.plot import add_changepoints_to_plot
    forecast = m.fit(series).predict(future)
    fig = m.plot(forecast)
    a = add_changepoints_to_plot(fig.gca(), m, forecast)
    plt.suptitle(stock+title)
    plt.show()
    
    #Components
    fig = model.plot_components(forecast)
    plt.suptitle(stock+title)
    plt.show()


df = get_yf(stock)
title = ' Price'
df['date'] = df.index
chart_prophet(df,0.05)

df_ltm,df_quarter = get_edgar(stock)
df = df_quarter
df['Close'] = df['Revenues']
df['date'] = df.index
title = ' Revenues'
chart_prophet(df,0.1)

df['Close'] = df['Revenues']-df['OperatingIncome']
df['date'] = df.index
title = ' Operating Expenses'
chart_prophet(df,0.1)
