# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 12:11:21 2024

@author: HB

V1: Added Page 1 Price, Page 2 Technicals, Page 3 Returns, Page 4 Correlations, Page 5 Options Chain

"""

import yfinance as yf
from Fred_API import fred_get_series
from Edgar_Company_Facts import get_cik, get_financial_data
import pandas as pd
from pandas_datareader import data as pdr
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.optimize import minimize
from scipy.signal import argrelextrema
from fpdf import FPDF
from PyPDF2 import PdfMerger


yf.pdr_override()

########Input Variables#########
end = dt.datetime.now() 
start = dt.datetime(2010,1,1)
look_forward=20
look_back=5
option_bin_size = None #Auto sets if none
highlight_filter = False

drawdown = -0.08
period_chg=1
threshold_high = -0.08
threshold_low =-0.50

edgar_data = True
stock = 'QCOM'
yf_1 = '^SPX' #Benchmark1, Can switch to Comp
yf_2 = '^RUT' #Benchmark2, Can switch to Comp
yf_3 = '^NDX' #Benchmark3, Can switch to Comp
yf_4 = '^VIX' #Equity Vol, Can use other Equity Vol indexes like ^VNX
yf_5 = '^VVIX'#VVIX
yf_6 = '^SKEW'#S&P Skew Index
yf_7 = '^MOVE'#Treasury Vol
yf_8 = '^TNX' #US10Y
fred_1 = 'GDPNOW' #'UNRATE' #CCSA continuing claims, UNRATE unemployment, GDPC1 Real GDP, M2SL M2, NFCI Chi Fed National Financial Conditions, WRMFNS Retail MM Funds
fred_2 = 'GDPC1' #'NFCI'
fred_3 = 'UNRATE' #'STLFSI4'
fred_4 = 'PCECC96' #Real Personal Consumption Expenditure
fred_5 = 'DPIC96' #Real Disposable Personal Income
fred_6 = 'PSAVE' #M2 Money SUpply'
fred_7 = None
fred_8 = None

sns.set_theme(rc={'figure.figsize':(8.5,11)})
x_axis_multiple = max(round((look_back+look_forward)/20),1)
pdf = PdfPages(stock+" Report as of "+dt.datetime.now().strftime("%Y-%m-%d")+'.pdf')

################################

#Creates a list of varibles where there are valid inputs i.e. non zero.
list_yf = [yf_1, yf_2, yf_3, yf_4, yf_5,yf_6,yf_7,yf_8]
list_yf = [var for var in list_yf if var]
list_fred = [fred_1, fred_2, fred_3, fred_4, fred_5,fred_6,fred_7,fred_8]
list_fred = [var for var in list_fred if var]
list_all_variables= [stock]+list_yf+list_fred

def get_yf_data (stock,start,end):
    df = pdr.get_data_yahoo(stock,start,end)
    df = df.reindex(pd.date_range(start, end, freq='D')).dropna(axis=0, how = 'any')
    return df
def get_fred_data(backtest_fred):
    df = fred_get_series(backtest_fred)
    df.index = df['date']
    df.index = pd.to_datetime(df.index)
    df['Close'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.reindex(pd.date_range(start, end, freq='D')).dropna(axis=0, how = 'any')
    return df
def get_options (stock):
    top = 3                         #top x open interest strike on each side
    list_options = []
    #Get Options Prices & Generate Graphs
    tk = yf.Ticker(stock)
    # Get Expiration dates
    exps = tk.options
    # Get options for each expiration
    list_wavg_oi = []
    list_wavg_vol = []
    list_top_oi = [] 
    list_top_iv = []
    list_bot_iv = []
    list_top_volume = []
    list_max_pain = []
    for e in exps:
        options = tk.option_chain(e)
        #Calculated Weighted Average Open Interest Strike
        df_call = options[0]
        df_call['Expiry'] = e
        df_call['PutCall'] = 'call'
        df_call = df_call[(df_call['ask']/df_call['strike'])>0.001] #10bps, filters out stale strikes
        df_call = df_call[(df_call['strike']/df_stock['Adj Close'][-1])<5] #filters stalke strikes from stock splits
        df_call = df_call[(df_call['impliedVolatility'])>0.05] #10bps, filters out stale strikes
        df_call['Breakeven'] = df_call['strike'] + df_call['lastPrice']
        df_call['Weight OI'] = df_call['openInterest']/df_call['openInterest'].sum()
        df_call['Weight Vol'] = df_call['volume']/df_call['volume'].sum()
        wavg_price_oi_call = (df_call['Breakeven']*df_call['Weight OI']).sum()
        wavg_price_vol_call = (df_call['Breakeven']*df_call['Weight Vol']).sum()
        oi_call = df_call['openInterest'].sum()
        vol_call = df_call['volume'].sum()
        
        df_put = options[1]
        df_put['Expiry'] = e
        df_put['PutCall'] = 'put'
        df_put = df_put[(df_put['ask']/df_put['strike'])>0.001] #10bps, filters out stale strikes
        df_put = df_put[(df_put['impliedVolatility'])>0.05] #10bps, filters out stale strikes
        df_put = df_put[(df_put['strike']/df_stock['Adj Close'][-1])<5] #filters stalke strikes from stock splits
        df_put['Breakeven'] = df_put['strike'] - df_put['lastPrice']
        df_put['Weight OI'] = df_put['openInterest']/df_put['openInterest'].sum()
        df_put['Weight Vol'] = df_put['volume']/df_put['volume'].sum()
        wavg_price_oi_put = (df_put['Breakeven']*df_put['Weight OI']).sum()
        wavg_price_vol_put = (df_put['Breakeven']*df_put['Weight Vol']).sum()
        oi_put = df_put['openInterest'].sum()
        vol_put = df_put['volume'].sum()
        
        #Adds all options to master list
        list_options.append(df_call) 
        list_options.append(df_put) 
        
        #
        wavg_price_oi_mat = (wavg_price_oi_call * oi_call + wavg_price_oi_put * oi_put)/(oi_call +oi_put)
        new_data = pd.DataFrame({'Expiry': e, 'WAVG Price':wavg_price_oi_mat,'WAVG Call Price': wavg_price_oi_call, 'WAVG Put Price':wavg_price_oi_put, 'OI Call':oi_call,'OI Put':oi_put}, index=[0])
        list_wavg_oi.append(new_data)
        #
        wavg_price_vol_mat = (wavg_price_vol_call * oi_call + wavg_price_vol_put * oi_put)/(oi_call +oi_put)
        new_data = pd.DataFrame({'Expiry': e, 'WAVG Price':wavg_price_vol_mat,'WAVG Call Price': wavg_price_vol_call, 'WAVG Put Price':wavg_price_vol_put, 'OI Call':oi_call,'OI Put':oi_put}, index=[0])
        list_wavg_vol.append(new_data)   
        #Gets the top x number of strikes by open interest
        df_call_top_oi = df_call.sort_values(by='openInterest',ascending = False)[:top]
        df_put_top_oi = df_put.sort_values(by='openInterest',ascending = False)[:top]
        list_top_oi.append(df_call_top_oi)
        list_top_oi.append(df_put_top_oi)      
        #Gets the top x number of strikes by IV, high IV = option buying some thesis whether long or short
        df_call_top_iv = df_call.sort_values(by='impliedVolatility',ascending = False)[:top]
        df_put_top_iv = df_put.sort_values(by='impliedVolatility',ascending = False)[:top]
        list_top_iv.append(df_call_top_iv)
        list_top_iv.append(df_put_top_iv)               
        #Gets the bot x number of strikes by IV, low IV = options selling or bounds of the price
        df_call_bot_iv = df_call.sort_values(by='impliedVolatility',ascending = True)[:top]
        df_put_bot_iv = df_put.sort_values(by='impliedVolatility',ascending = True)[:top]
        list_bot_iv.append(df_call_bot_iv)
        list_bot_iv.append(df_put_bot_iv)
        #Gets the top x number of strikes by volume
        df_call_top_volume = df_call.sort_values(by='volume',ascending = False)[:top]
        df_put_top_volume = df_put.sort_values(by='volume',ascending = False)[:top]
        list_top_volume.append(df_call_top_volume)
        list_top_volume.append(df_put_top_volume)
        
        def max_pain_func(price):
            return (np.maximum(price - df_call.strike,0) + np.maximum(df_put.strike-price,0)).sum()

        max_pain = minimize(max_pain_func,df_stock['Close'][-1])
        max_pain = max_pain.x
        df_max_pain = pd.DataFrame({'Expiry':df_call.Expiry.iloc[0],'Max Pain':max_pain})
        list_max_pain.append(df_max_pain)

    df_wavg_oi =pd.concat(list_wavg_oi)
    df_wavg_oi['Expiry'] = pd.to_datetime(df_wavg_oi['Expiry'])
    df_wavg_oi.set_index('Expiry',inplace=True)
    df_wavg_oi=df_wavg_oi.dropna(axis=0,how='any')
    
    df_top_oi = pd.concat(list_top_oi)
    df_top_oi['Expiry'] = pd.to_datetime(df_top_oi['Expiry'])
    df_top_oi.set_index('Expiry',inplace=True)
    
    df_bot_iv = pd.concat(list_bot_iv)
    df_bot_iv['Expiry'] = pd.to_datetime(df_bot_iv['Expiry'])
    df_bot_iv.set_index('Expiry',inplace=True)
    
    df_max_pain = pd.concat(list_max_pain)
    df_max_pain['Expiry'] = pd.to_datetime(   df_max_pain['Expiry'])
    df_max_pain.set_index('Expiry',inplace=True)
    
    df_options=pd.concat(list_options)
    df_options['Expiry'] = pd.to_datetime(df_options['Expiry'], format='%Y-%m-%d')
    df_options['Days'] = (df_options['Expiry'] - end).dt.days
    df_options['T'] = df_options['Days']/365         
    
    return df_wavg_oi,df_top_oi,df_bot_iv,df_max_pain,df_options
    
def date_drawdown_pct(df,threshold_high,threshold_low,period_chg=1): # Finds all the dates where the stock breached the threshold
    df['daily_pct_change'] = df.Close.pct_change(periods=1)
    df['period_pct_chg'] = df.Close.pct_change(periods=period_chg)
    threshold_list = df[(df['period_pct_chg']<threshold_high) & (df['period_pct_chg']>threshold_low)]
        
    list_dates = threshold_list.index.tolist()
    clean_dates = []
    clean_dates.append(list_dates[0])

    for i in range(0, len(list_dates)):
        if list_dates[i] > clean_dates[-1] + dt.timedelta(days=period_chg): #removes duplicates from returns i.e. -8% today will also screen in -5% tomorrow unless price moves up 3%
            clean_dates.append(list_dates[i])  
    for i in range(0,len(clean_dates)):
        clean_dates[i] = df_stock.index[df_stock.index.searchsorted(clean_dates[i])] #converts Fred Dates to Nearest Stock market dates
    return clean_dates
def date_drawdown_level(df,threshold_high,threshold_low,period_chg=1):
    threshold_list = df[(df['Close']<threshold_high) & (df['Close']>threshold_low)]
    list_dates = threshold_list.index.tolist()
    clean_dates = []
    clean_dates.append(list_dates[0])
    for i in range(0,len(clean_dates)):
        clean_dates[i] = list_df[stock].index[list_df[stock].index.searchsorted(clean_dates[i])] #converts Fred Dates to Nearest Stock market dates
    
    title = stock +' Below '+f'{threshold_high:+.1%}' 
    return clean_dates
def date_level_roc(df,threshold_low,rate_of_change,period_chg=1):
    df['daily_pct_change'] = df.Close.pct_change(periods=1)
    threshold_list = df[(df['Close']>threshold_low) & (df['daily_pct_change']>rate_of_change)]
    
    list_dates = threshold_list.index.tolist()
    clean_dates = []
    clean_dates.append(list_dates[0])

    for i in range(0, len(list_dates)):
        if list_dates[i] > clean_dates[-1] + dt.timedelta(days=period_chg):
            clean_dates.append(list_dates[i])
            
    title = stock + f'{rate_of_change:+.1%} Change From '+f'{threshold_low:.1f}'+' to ' +f'{threshold_high:.1f}'
    return clean_dates

def date_first_rate_cut():
    fedfunds = fred_get_series('FEDFUNDS')
    fedfunds.index = fedfunds['date']
    fedfunds.index = pd.to_datetime(fedfunds.index)
    fedfunds = fedfunds['value']
    fedfunds = pd.to_numeric(fedfunds)
    
    df_fedfunds = pd.DataFrame(fedfunds)
    df_rates = df_fedfunds.join(df_stock.Close).dropna(how='any')
    df_rates = df_rates.rename(columns={"value": "FedFunds", "Close": "US10Y"})
    
    
    threshold_list = df_rates[(df_rates.FedFunds.shift(1)-df_rates.FedFunds > 0.25)&(df_rates.FedFunds.rolling(12).max() - df_rates.FedFunds.shift(1)<0.25)] #First Rate Hike, Filters out rolling 12M noise 
    #threshold_list = df_rates[(df_rates.FedFunds.shift(1)-df_rates.FedFunds > 0.25)&(df_rates.FedFunds.rolling(12).max() - df_rates.FedFunds.shift(1)<0.25)&(df_rates.FedFunds > df_rates.US10Y)]  #First Cut with no changes in prior 12M and Fedfunds > US10Y
    #threshold_list = df_rates[(df_rates.FedFunds.shift(1)-df_rates.FedFunds > 0.25)] #any rate cut not just fisrt rate hike   
    list_dates = threshold_list.index.tolist()
    return list_dates
def calc_rsi(df,window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
def calc_ema(df, span):
    return df.ewm(span=span, adjust=False).mean()
def calc_macd(df):
    short_ema = calc_ema(df['Close'], 12)
    long_ema = calc_ema(df['Close'], 26)
    macd = short_ema - long_ema
    signal = calc_ema(macd, 9)
    histogram = macd - signal
    return macd, signal, histogram

#Get the data for the backtest variable. Change bbased on Data Source.
list_df = {}
list_df[stock] = get_yf_data(stock,start-dt.timedelta(days=365*2),end)
for x in list_yf:
    list_df[x] = get_yf_data(x,start-dt.timedelta(days=365*2),end)
for x in list_fred:
    list_df[x] = get_fred_data(x) 

#Transformations
def transform_data(df):
    df['daily_pct_change'] = df.Close.pct_change(periods=1)
    df['0.25Y Return Annualized'] = (1+df.Close.pct_change(periods=63))**(4)-1
    df['0.50Y Return Annualized'] = (1+df.Close.pct_change(periods=176))**(2)-1
    df['1Y Return CAGR'] = df.Close.pct_change(periods=253)
    df['2Y Return CAGR'] = (1+df.Close.pct_change(periods=253*2))**(1/2)-1
    df['3Y Return CAGR'] = (1+df.Close.pct_change(periods=253*3))**(1/3)-1
    df['YTD_Return'] = df['Close'] / df['Close'].groupby(df.index.year).transform('first') - 1
    df['Dollar_Volume'] = df['Close'] * df['Volume']
    df['VWAP 200D'] = df['Dollar_Volume'].rolling(window=200).sum()/df['Volume'].rolling(window=200).sum()
    df['LocalMax'] = df['Close'][argrelextrema(df['Close'].values, np.greater_equal, order=5)[0]]
    df['LocalMin'] = df['Close'][argrelextrema(df['Close'].values, np.less_equal, order=5)[0]]
    df['LocalDrawdown'] = df['Close'] / df['LocalMax'].ffill() - 1
    df['LocalDrawdownMin'] = df['LocalDrawdown'][argrelextrema(df['LocalDrawdown'].values, np.less_equal, order=5)[0]]
    df['LocalDrawdownMin'] = df['LocalDrawdownMin'].bfill()
    df['rolling_max'] = df['Close'].cummax()
    df['Drawdown'] = (df['Close']-df['rolling_max']) / df['rolling_max'] 
    df['Volatility'] = (df['Close'].rolling(window=20).std())
    df['Intraday_Range'] = df['High'] - df['Low']
    df['Intraday_Range_Pct'] = df['High']/df['Close'] - df['Low']/df['Close']
    df['Close_intraday_range'] = (df['Close'] - df['Low'])/df['Intraday_Range']*100
    df['Close_intraday_range_smooth'] = df['Close_intraday_range'].rolling(window=5).mean()
    df['5D Return'] = df['Close']/df['Close'].shift(5)-1
    df['20D Return'] = df['Close']/df['Close'].shift(20)-1
    df['50D Return'] = df['Close']/df['Close'].shift(50)-1
    df['200D Return'] = df['Close']/df['Close'].shift(200)-1
    df['20D SMA'] = df['Close'].rolling(window=20).mean()
    df['50D SMA'] = df['Close'].rolling(window=50).mean()
    df['200D SMA'] = df['Close'].rolling(window=200).mean()
    df['RSI 14D'] = calc_rsi(df,window=14)
    df['Momentum 10D'] = (df['Close']-df['Close'].shift(10))/df['Close'].shift(10)
    df['MACD'], df['MACD_Signal'], df['MACD_Histogram']  = calc_macd(df)
    df = df[df.index > start]
    return df
def get_drawdown_df(df):
    df=df_stock
    df_lcmax_dates=df[(df['LocalDrawdownMin']<drawdown) & (df['LocalMax']>0)]
    df_lcmin_dates=df[(df['LocalDrawdownMin']<drawdown) & (df['LocalMin']>0)]
    df_localdd=pd.concat([df_lcmax_dates,df_lcmin_dates])
    df_localdd=df_localdd.sort_index(axis=0,ascending=True)
    df_localdd['lmax'] = df_localdd['LocalMax'].notna() & df_localdd['LocalMax'].shift(1).isna()
    df_localdd['lmin'] = df_localdd['LocalMin'].notna() & df_localdd['LocalMin'].shift(1).isna()
    df_localdd=df_localdd[(df_localdd['lmax']==True)|(df_localdd['lmin']==True)]
    df_localdd['date']=df_localdd.index
    df_localdd['Days'] = (df_localdd['date'].shift(-1)-df_localdd['date']).dt.days
    df_localdd['Recovery Days'] = (df_localdd['date'].shift(-2)-df_localdd['date']).dt.days
    df_localdd['LocalDrawdown'] = df_localdd['Close']/df_localdd['Close'].shift(1)-1
    df_localdd_min = df_localdd[df_localdd['LocalMin']>0]
    df_localdd_max = df_localdd[df_localdd['LocalMax']>0]
    dates = df_localdd_max.index.tolist()
    return dates,df_localdd,df_localdd_min,df_localdd_max

df_stock = transform_data(list_df[stock])
df_benchmark1 = transform_data(list_df[yf_1])
df_benchmark2 = transform_data(list_df[yf_2])
df_benchmark3 = transform_data(list_df[yf_3])
dates,df_localdd,df_localdd_min,df_localdd_max = get_drawdown_df(df_stock)
          
#Get Dates
df_dd_start =df_stock[(df_stock['LocalDrawdownMin']<drawdown) & (df_stock['LocalMax']>0)]
#dates = df_dd_start.index.tolist()
#dates = date_drawdown_pct(df_stock,threshold_high=threshold_high,threshold_low=threshold_low,period_chg=period_chg)
dates = date_first_rate_cut()
transparency= max(0.8 - len(dates)/100,0.15)

#Page One Price Summary
var_count = 5
fig = plt.figure(figsize=(8.5, 11))
gs = gridspec.GridSpec(var_count, 1, height_ratios=[2] + [1] * (var_count- 1))  # First subplot is larger
axes = [fig.add_subplot(gs[i]) for i in range(var_count)]

axes[0].plot(df_stock['Close'],color='black',label='Price')
axes[0].plot(df_stock['50D SMA'],color='darkorange',label='SMA 50D')
axes[0].plot(df_stock['200D SMA'],color='cornflowerblue',label='SMA 200D')
axes[0].set_title(stock+' Price')
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[0].text(df_stock.index[-1],df_stock['Close'][-1],f'${df_stock["Close"][-1]:.0f}',color='black')
axes[0].text(df_stock.index[-1],df_stock['50D SMA'][-1],f'${df_stock["50D SMA"][-1]:.0f}',color='darkorange')
axes[0].text(df_stock.index[-1],df_stock['200D SMA'][-1],f'${df_stock["200D SMA"][-1]:.0f}',color='blue')
axes[0].legend()

axes[1].plot(df_stock['Drawdown'],color='cornflowerblue')
axes[1].set_title(stock +' Drawdown from Prior High')
axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[1].sharex(axes[0])
axes[1].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[1].text(df_stock.index[-1],df_stock['Drawdown'][-1],f'{df_stock["Drawdown"][-1]:+.1%}',color='blue')

axes[2].plot(df_stock['daily_pct_change'],color='cornflowerblue')
axes[2].set_title(stock +' 1D Percentage Change')
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[2].sharex(axes[0])
axes[2].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
#axes[2].text(df_stock.index[-1],df_stock['daily_pct_change'][-1],f'{df_stock["daily_pct_change"][-1]:+.1%}')

axes[3].plot(df_stock['Volume'],color='cornflowerblue')
axes[3].set_title(stock +' Volume')
axes[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[3].sharex(axes[0])
#axes[3].text(df_stock.index[-1],df_stock['Volatility'][-1],f'{df_stock["Volatility"][-1]:.1f}')

axes[4].plot(df_stock['Volatility'],color='cornflowerblue')
axes[4].set_title(stock +' Volatility 20D')
#axes[4].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[4].sharex(axes[0])

if highlight_filter == True:
    for y in range(0,len(dates)):
        for x in range(0,var_count):
            axes[x].axvline(dates[y],color='red', alpha=transparency)

fig.suptitle(stock+" Report as of "+dt.datetime.now().strftime("%Y-%m-%d"))
plt.tight_layout(h_pad=0.3)
pdf.savefig()
plt.show()
plt.close()

#Page Two Technicals Summary
var_count = 5
fig = plt.figure(figsize=(8.5, 11))
gs = gridspec.GridSpec(var_count, 1, height_ratios=[2] + [1] * (var_count- 1))  # First subplot is larger
axes = [fig.add_subplot(gs[i]) for i in range(var_count)]

axes[0].plot(df_stock['Close']/df_stock['20D SMA'],color='green',label='20D SMA')
axes[0].plot(df_stock['Close']/df_stock['50D SMA'],color='darkorange',label='50D SMA')
axes[0].plot(df_stock['Close']/df_stock['200D SMA'] ,color='cornflowerblue',label='200D SMA')
axes[0].axhline(1,color='black')
axes[0].set_title(stock+' Muliple of Simple Moving Average')
axes[0].set_xlim(start,end)
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[0].legend()

axes[1].plot(df_stock['MACD'],color='green')
axes[1].plot(df_stock['MACD_Signal'],color='red')
axes[1].bar(df_stock.index,df_stock['MACD_Histogram'],linewidth=0,label='Histogram',color='black')
axes[1].axhline(0,color='black')
axes[1].set_title(stock +' MACD')
axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[1].sharex(axes[0])

axes[2].plot(df_stock['Momentum 10D'],color='cornflowerblue')
axes[2].set_title(stock +' Momentum 10D')
axes[2].axhline(0,color='black')
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[2].sharex(axes[0])
axes[2].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))

axes[3].plot(df_stock['RSI 14D'],color='cornflowerblue')
axes[3].set_title(stock +' RSI')
axes[3].axhline(20, color='black')
axes[3].axhline(80,color='black')
axes[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[3].sharex(axes[0])

axes[4].plot(df_stock['Close_intraday_range_smooth'],color='cornflowerblue')
axes[4].set_title(stock +' Close Intraday Range 5D Average')
axes[4].axhline(50,color='black')
#axes[4].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[4].sharex(axes[0])

if highlight_filter == True:
    for y in range(0,len(dates)):
        for x in range(0,var_count):
            axes[x].axvline(dates[y],color='red', alpha=transparency)

plt.tight_layout(h_pad=0.3)
pdf.savefig()
plt.show()
plt.close()

#Page Three Returns
var_count = 5
fig = plt.figure(figsize=(8.5, 11))
gs = gridspec.GridSpec(var_count, 1, height_ratios=[2] + [1] * (var_count- 1))  # First subplot is larger
axes = [fig.add_subplot(gs[i]) for i in range(var_count)]

axes[0].plot(df_stock['YTD_Return'],label='YTD',color='black')
axes[0].plot(df_stock['1Y Return CAGR'],label='1Y',color='tab:green')
axes[0].plot(df_stock['2Y Return CAGR'],label='2Y',color='darkorange')
axes[0].plot(df_stock['3Y Return CAGR'],label='3Y',color='cornflowerblue')
axes[0].axhline(0,color='black')
axes[0].text(df_stock.index[-1],df_stock['YTD_Return'][-1],f'{df_stock["0.25Y Return Annualized"][-1]:+.1%}',color='black')
axes[0].text(df_stock.index[-1],df_stock['1Y Return CAGR'][-1],f'{df_stock["1Y Return CAGR"][-1]:+.1%}',color='green')
axes[0].text(df_stock.index[-1],df_stock['2Y Return CAGR'][-1],f'{df_stock["2Y Return CAGR"][-1]:+.1%}',color='darkorange')
axes[0].text(df_stock.index[-1],df_stock['3Y Return CAGR'][-1],f'{df_stock["3Y Return CAGR"][-1]:+.1%}',color='blue')
axes[0].set_title(stock+' Annualized Returns')
axes[0].legend()
axes[0].set_xlim(start,end)
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[0].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[0].legend()

axes[1].plot(df_stock['5D Return'],label='5D',color='black')
axes[1].plot(df_stock['20D Return'],label='20D',color='green')
axes[1].plot(df_stock['50D Return'],label='50D',color='darkorange')
axes[1].plot(df_stock['200D Return'],label='200D',color='cornflowerblue')
axes[1].axhline(0,color='black')
axes[1].text(df_stock.index[-1],df_stock['5D Return'][-1],f'{df_stock["5D Return"][-1]:+.1%}',color='black')
axes[1].text(df_stock.index[-1],df_stock['20D Return'][-1],f'{df_stock["20D Return"][-1]:+.1%}',color='green')
axes[1].text(df_stock.index[-1],df_stock['50D Return'][-1],f'{df_stock["50D Return"][-1]:+.1%}',color='darkorange')
axes[1].text(df_stock.index[-1],df_stock['200D Return'][-1],f'{df_stock["200D Return"][-1]:+.1%}',color='blue')
axes[1].set_title(stock+' Rolling Returns')
axes[1].legend()
axes[1].set_xlim(start,end)
axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[1].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[1].legend()

axes[2].plot(df_stock['20D Return']-df_benchmark1['20D Return'],label=yf_1,color='gray')
axes[2].plot(df_stock['20D Return']-df_benchmark2['20D Return'],label=yf_2,color='darkorange')
axes[2].plot(df_stock['20D Return']-df_benchmark3['20D Return'],label=yf_3,color='cornflowerblue')
axes[2].text(df_stock.index[-1],df_stock['20D Return'][-1]-df_benchmark1['20D Return'][-1],f'{df_stock["20D Return"][-1]-df_benchmark1["20D Return"][-1]:+.1%}',color='black')
axes[2].text(df_stock.index[-1],df_stock['20D Return'][-1]-df_benchmark2['20D Return'][-1],f'{df_stock["20D Return"][-1]-df_benchmark2["20D Return"][-1]:+.1%}',color='darkorange')
axes[2].text(df_stock.index[-1],df_stock['20D Return'][-1]-df_benchmark3['20D Return'][-1],f'{df_stock["20D Return"][-1]-df_benchmark3["20D Return"][-1]:+.1%}',color='blue')
axes[2].axhline(0,color='black')
axes[2].set_title(stock+' - 20D Relative Returns')
axes[2].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[2].legend()
axes[2].sharex(axes[0])
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[3].plot(df_stock['50D Return']-df_benchmark1['50D Return'],label=yf_1,color='gray')
axes[3].plot(df_stock['50D Return']-df_benchmark2['50D Return'],label=yf_2,color='darkorange')
axes[3].plot(df_stock['50D Return']-df_benchmark3['50D Return'],label=yf_3,color='cornflowerblue')
axes[3].text(df_stock.index[-1],df_stock['50D Return'][-1]-df_benchmark1['50D Return'][-1],f'{df_stock["50D Return"][-1]-df_benchmark1["50D Return"][-1]:+.1%}',color='black')
axes[3].text(df_stock.index[-1],df_stock['50D Return'][-1]-df_benchmark2['50D Return'][-1],f'{df_stock["50D Return"][-1]-df_benchmark2["50D Return"][-1]:+.1%}',color='darkorange')
axes[3].text(df_stock.index[-1],df_stock['50D Return'][-1]-df_benchmark3['50D Return'][-1],f'{df_stock["50D Return"][-1]-df_benchmark3["50D Return"][-1]:+.1%}',color='blue')
axes[3].axhline(0,color='black')
axes[3].set_title(stock+' - 50D Relative Returns')
axes[3].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[3].legend()
axes[3].sharex(axes[0])
axes[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[4].plot(df_stock['200D Return']-df_benchmark1['200D Return'],label=yf_1,color='gray')
axes[4].plot(df_stock['200D Return']-df_benchmark2['200D Return'],label=yf_2,color='darkorange')
axes[4].plot(df_stock['200D Return']-df_benchmark3['200D Return'],label=yf_3,color='cornflowerblue')
axes[4].text(df_stock.index[-1],df_stock['200D Return'][-1]-df_benchmark1['200D Return'][-1],f'{df_stock["200D Return"][-1]-df_benchmark1["200D Return"][-1]:+.1%}',color='black')
axes[4].text(df_stock.index[-1],df_stock['200D Return'][-1]-df_benchmark2['200D Return'][-1],f'{df_stock["200D Return"][-1]-df_benchmark2["200D Return"][-1]:+.1%}',color='darkorange')
axes[4].text(df_stock.index[-1],df_stock['200D Return'][-1]-df_benchmark3['200D Return'][-1],f'{df_stock["200D Return"][-1]-df_benchmark3["200D Return"][-1]:+.1%}',color='blue')
axes[4].axhline(0,color='black')
axes[4].set_title(stock+' - 200D Relative Returns')
axes[4].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[4].sharex(axes[0])
axes[4].legend()
#axes[4].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

if highlight_filter == True:
    for y in range(0,len(dates)):
        for x in range(0,var_count):
            axes[x].axvline(dates[y],color='red', alpha=transparency)

plt.tight_layout(h_pad=0.3)
pdf.savefig()
plt.show()
plt.close()

#Page Four Correlation
var_count = 6
fig = plt.figure(figsize=(8.5, 11))
gs = gridspec.GridSpec(var_count, 1, height_ratios=[1] + [1] * (var_count- 1))  # First subplot is larger
axes = [fig.add_subplot(gs[i]) for i in range(var_count)]

axes[0].plot(df_stock['Close'].pct_change(periods=1).rolling(10).corr(list_df[yf_1]['Close'].pct_change(periods=1)),color='cornflowerblue')
axes[0].set_title(stock+' Return Correlation with '+yf_1)
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[0].set_xlim(start,end)
axes[0].set_ylim(-1,1)

axes[1].plot(df_stock['Close'].pct_change(periods=1).rolling(10).corr(list_df[yf_2]['Close'].pct_change(periods=1)),color='cornflowerblue')
axes[1].set_title(stock+' Return Correlation with '+yf_2)
axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[1].set_ylim(-1,1)
axes[1].sharex(axes[0])

axes[2].plot(df_stock['Close'].pct_change(periods=1).rolling(10).corr(list_df[yf_3]['Close'].pct_change(periods=1)),color='cornflowerblue')
axes[2].set_title(stock+' Return Correlation with '+yf_3)
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[2].set_ylim(-1,1)
axes[2].sharex(axes[0])

axes[3].plot(df_stock['Close'].pct_change(periods=1).rolling(10).corr(list_df[yf_4]['Close'].pct_change(periods=1)),color='cornflowerblue')
axes[3].set_title(stock+' Return Correlation with '+yf_4)
axes[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[3].set_ylim(-1,1)
axes[3].sharex(axes[0])

axes[4].plot(df_stock['Close'].pct_change(periods=1).rolling(10).corr(list_df[yf_7]['Close'].pct_change(periods=1)),color='cornflowerblue')
axes[4].set_title(stock+' Return Correlation with '+yf_7)
axes[4].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[4].set_ylim(-1,1)
axes[4].sharex(axes[0])

axes[5].plot(df_stock['Close'].pct_change(periods=1).rolling(10).corr(list_df[yf_8]['Close'].pct_change(periods=1)),color='cornflowerblue')
axes[5].set_title(stock+' Return Correlation with '+yf_8)
axes[5].set_ylim(-1,1)
axes[5].sharex(axes[0])
#axes[5].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

if highlight_filter == True:
    for y in range(0,len(dates)):
        for x in range(0,var_count):
            axes[x].axvline(dates[y],color='red', alpha=transparency)

plt.tight_layout(h_pad=0.3)
pdf.savefig()
plt.show()
plt.close()

#Page Five Volatility
var_count = 6
fig = plt.figure(figsize=(8.5, 11))
gs = gridspec.GridSpec(var_count, 1, height_ratios=[1] + [1] * (var_count- 1))  # First subplot is larger
axes = [fig.add_subplot(gs[i]) for i in range(var_count)]

axes[0].plot(list_df[yf_4]['Close'][start:],color='cornflowerblue')
axes[0].set_title('S&P500 VIX')
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[0].set_xlim(start,end)
axes[0].sharex(axes[0])

axes[1].plot(list_df[yf_5]['Close'][start:],color='cornflowerblue')
axes[1].set_title('VVIX')
axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[1].sharex(axes[0])

axes[2].plot(list_df[yf_6]['Close'][start:],color='cornflowerblue')
axes[2].set_title('CBOE Skew Index')
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[2].sharex(axes[0])

axes[3].plot(list_df[yf_7]['Close'][start:],color='cornflowerblue')
axes[3].set_title('US Treasury Move Index')
axes[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[3].sharex(axes[0])

axes[4].plot(df_stock['Volatility'],color='cornflowerblue')
axes[4].set_title(stock+' Volatility')
axes[4].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[4].sharex(axes[0])

axes[5].plot(df_stock['Intraday_Range_Pct'],color='cornflowerblue')
axes[5].set_title(stock+' Intraday Range Percent')
axes[5].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[5].sharex(axes[0])
#axes[5].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

if highlight_filter == True:
    for y in range(0,len(dates)):
        for x in range(0,var_count):
            axes[x].axvline(dates[y],color='red', alpha=transparency)

plt.tight_layout(h_pad=0.5)
pdf.savefig()
plt.show()
plt.close()

#Page 6 Fundamental Data
if edgar_data == True:
    try:
        cik = get_cik(stock)
        list_df_edgar = get_financial_data(cik)
        df_edgar_ltm = list_df_edgar[2]
        df_edgar_ltm = pd.concat([df_stock['Close'],df_edgar_ltm],axis=0,join='outer')
        df_edgar_ltm=df_edgar_ltm.rename(columns={0: "Close"})
        df_edgar_ltm=df_edgar_ltm.sort_index()
        df_edgar_ltm=df_edgar_ltm.fillna(method='ffill')
        df_edgar_ltm=df_edgar_ltm.dropna(axis=0,subset=['Revenues'])
        df_edgar_ltm['Market_Cap'] = df_edgar_ltm['Close']*df_edgar_ltm['Adjusted DSO']
        df_edgar_ltm['Enterprise_Value'] = df_edgar_ltm['Market_Cap']+df_edgar_ltm['LongTermDebt']-df_edgar_ltm['Cash']
        df_edgar_ltm['EV/S'] = df_edgar_ltm['Enterprise_Value'] / df_edgar_ltm['Revenues']
        df_edgar_ltm['EV/EBITDA'] = df_edgar_ltm['Enterprise_Value'] / df_edgar_ltm['EBITDA']
        df_edgar_ltm['P/E'] = df_edgar_ltm['Market_Cap']/df_edgar_ltm['NetIncome']
        df_edgar_ltm['FCFE/Equity'] = df_edgar_ltm['FCFEexSBC']/df_edgar_ltm['Market_Cap']
        
        var_count = 5
        fig = plt.figure(figsize=(8.5, 11))
        gs = gridspec.GridSpec(var_count, 1, height_ratios=[1] + [1] * (var_count- 1))  # First subplot is larger
        axes = [fig.add_subplot(gs[i]) for i in range(var_count)]
        
        axes[0].plot(df_edgar_ltm['EV/S'])
        df_edgar_ltm['EV/S Avg'] = df_edgar_ltm['EV/S'].mean()
        df_edgar_ltm['EV/S SD'] = df_edgar_ltm['EV/S'].std()
        axes[0].plot(df_edgar_ltm['EV/S Avg']) 
        axes[0].plot(df_edgar_ltm['EV/S Avg']+2*df_edgar_ltm['EV/S SD']) 
        axes[0].plot(df_edgar_ltm['EV/S Avg']-2*df_edgar_ltm['EV/S SD']) 
        axes[0].set_title('LTM EV/S')
        axes[0].set_xlim(start,end)
        axes[0].sharex(axes[0])
        axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        
        axes[1].plot(df_edgar_ltm['EV/EBITDA'])
        df_edgar_ltm['EV/EBITDA Avg'] = df_edgar_ltm['EV/EBITDA'].mean()
        df_edgar_ltm['EV/EBITDA SD'] = df_edgar_ltm['EV/EBITDA'].std()
        axes[1].plot(df_edgar_ltm['EV/EBITDA Avg'])
        axes[1].plot(df_edgar_ltm['EV/EBITDA Avg']+2*df_edgar_ltm['EV/EBITDA SD']) 
        axes[1].plot(df_edgar_ltm['EV/EBITDA Avg']-2*df_edgar_ltm['EV/EBITDA SD']) 
        axes[1].set_title('LTM EV/EBITDA')
        axes[1].sharex(axes[0])
        axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        
        axes[2].plot(df_edgar_ltm['P/E'])
        df_edgar_ltm['P/E Avg'] = df_edgar_ltm['P/E'].mean()
        df_edgar_ltm['P/E SD'] = df_edgar_ltm['P/E'].std()
        axes[2].plot(df_edgar_ltm['P/E Avg'])
        axes[2].plot(df_edgar_ltm['P/E Avg']+2*df_edgar_ltm['P/E SD']) 
        axes[2].plot(df_edgar_ltm['P/E Avg']-2*df_edgar_ltm['P/E SD']) 
        axes[2].set_title('LTM P/E')
        axes[2].sharex(axes[0])
        axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        
        axes[3].plot(df_edgar_ltm['FCFE/Equity'],label='FCFE/Equity')
        df_edgar_ltm['FCFE/Equity Avg'] = df_edgar_ltm['FCFE/Equity'].mean()
        df_edgar_ltm['FCFE/Equity SD'] = df_edgar_ltm['FCFE/Equity'].std()
        axes[3].plot(df_edgar_ltm['FCFE/Equity Avg'],label='Average')
        axes[3].plot(df_edgar_ltm['FCFE/Equity Avg']+2*df_edgar_ltm['FCFE/Equity SD']) 
        axes[3].plot(df_edgar_ltm['FCFE/Equity Avg']-2*df_edgar_ltm['FCFE/Equity SD']) 
        axes[3].set_title('LTM FCFE/Equity')
        axes[3].sharex(axes[0])
        axes[3].yaxis.set_major_formatter(PercentFormatter(1,decimals=1))
        axes[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        

        df_edgar_ltm['Implied LTGR'] = (0.1 *df_edgar_ltm['P/E'] -1)/(1+df_edgar_ltm['P/E'])
        df_edgar_ltm['Implied COE'] = (1+0.04)/df_edgar_ltm['P/E']+0.04 #Gr = 0.04, Earnings = 1
        axes[4].plot(df_edgar_ltm['Implied LTGR'],label='LTGR @ 10% COE') #10% COE, No Reinvestment Costs
        axes[4].plot(df_edgar_ltm['Implied COE'],label='COE @ 4% LTGR') #4% LTGR
        axes[4].set_title('Implied Metrics')
        axes[4].legend()
        axes[4].sharex(axes[0])
        axes[4].yaxis.set_major_formatter(PercentFormatter(1,decimals=1))
        #axes[4].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tight_layout(h_pad=0.5)
        pdf.savefig()
        plt.show()
        plt.close()
        
        #Growth & Margins
        var_count = 3
        fig = plt.figure(figsize=(8.5, 11))
        gs = gridspec.GridSpec(var_count, 1, height_ratios=[1] + [1] * (var_count- 1))  # First subplot is larger
        axes = [fig.add_subplot(gs[i]) for i in range(var_count)]
        
        axes[0].plot((df_edgar_ltm['Revenues']/df_edgar_ltm['Revenues'].shift(1))**4-1,label='Revenues')
        axes[0].set_title('Growth')
        axes[0].legend()
        axes[0].set_xlim(start,end)
        axes[0].sharex(axes[0])
        axes[0].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        
        axes[1].plot(df_edgar_ltm['GrossProfit']/df_edgar_ltm['Revenues'],label='Gross Profit')
        axes[1].plot(df_edgar_ltm['EBITDA']/df_edgar_ltm['Revenues'],label='EBITDA')
        axes[1].plot(df_edgar_ltm['NetIncome']/df_edgar_ltm['Revenues'],label='Net Income')
        axes[1].plot(df_edgar_ltm['FCFE']/df_edgar_ltm['Revenues'],label='FCFE')
        axes[1].set_title('Margins')
        axes[1].legend()
        axes[1].sharex(axes[0])
        axes[1].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        
        axes[2].plot(df_edgar_ltm['ROE'],label='ROE')
        axes[2].plot(df_edgar_ltm['ROIC'],label='ROIC')
        axes[2].plot(df_edgar_ltm['ROCE'],label='ROCE')
        axes[2].set_title('Returns on Capital')
        axes[2].legend()
        axes[2].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        axes[2].sharex(axes[0])
        #axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        
        plt.tight_layout(h_pad=0.5)
        pdf.savefig()
        plt.show()
        plt.close()
        
        #Page Economic Data
        var_count = 6
        fig = plt.figure(figsize=(8.5, 11))
        gs = gridspec.GridSpec(var_count, 1, height_ratios=[2] + [1] * (var_count- 1))  # First subplot is larger
        axes = [fig.add_subplot(gs[i]) for i in range(var_count)]

        axes[0].plot((df_edgar_ltm['Revenues']/df_edgar_ltm['Revenues'].shift(4))**4-1,label='Revenues')
        axes[0].set_title('Company Earnings')
        axes[0].legend()
        #axes[0].sharex(axes[0])
        axes[0].set_xlim(start,end)
        axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        axes[1].plot(list_df[fred_2]['Close'][start:],color='cornflowerblue',label='GDP')
        axes[1].set_title('GDP')
        axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axes[1].sharex(axes[0])

        axes[2].plot(list_df[fred_3]['Close'][start:],color='cornflowerblue')
        axes[2].set_title('Unemployment Rate')
        axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axes[2].sharex(axes[0])

        axes[3].plot(list_df[fred_4]['Close'][start:],color='cornflowerblue')
        axes[3].set_title('Real Personal Consumption Expenditure')
        axes[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axes[3].sharex(axes[0])

        axes[4].plot(list_df[fred_5]['Close'][start:],color='cornflowerblue')
        axes[4].set_title('Real Personal Income')
        axes[4].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axes[4].sharex(axes[0])

        axes[5].plot(list_df[fred_6]['Close'][start:],color='cornflowerblue')
        axes[5].set_title('Personal Savings')
        #axes[5].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axes[5].sharex(axes[0])

        if highlight_filter == True:
            for y in range(0,len(dates)):
                for x in range(0,var_count):
                    axes[x].axvline(dates[y],color='red', alpha=transparency)

        plt.tight_layout(h_pad=0.5)
        pdf.savefig()
        plt.show()
        plt.close()
    except:
        #Page 6 Economic Data
        var_count = 6
        fig = plt.figure(figsize=(8.5, 11))
        gs = gridspec.GridSpec(var_count, 1, height_ratios=[1] + [1] * (var_count- 1))  # First subplot is larger
        axes = [fig.add_subplot(gs[i]) for i in range(var_count)]

        axes[0].plot(list_df[fred_1]['Close'][start:],color='cornflowerblue',label='GDP Now')
        axes[0].set_title('GDP Now')
        axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axes[0].set_xlim(start,end)
        axes[0].sharex(axes[0])

        axes[1].plot(list_df[fred_2]['Close'][start:],color='cornflowerblue',label='GDP')
        axes[1].set_title('GDP')
        axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axes[1].sharex(axes[0])

        axes[2].plot(list_df[fred_3]['Close'][start:],color='cornflowerblue')
        axes[2].set_title('Unemployment Rate')
        axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axes[2].sharex(axes[0])

        axes[3].plot(list_df[fred_4]['Close'][start:],color='cornflowerblue')
        axes[3].set_title('Real Personal Consumption Expenditure')
        axes[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axes[3].sharex(axes[0])

        axes[4].plot(list_df[fred_5]['Close'][start:],color='cornflowerblue')
        axes[4].set_title('Real Personal Income')
        axes[4].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axes[4].sharex(axes[0])

        axes[5].plot(list_df[fred_6]['Close'][start:],color='cornflowerblue')
        axes[5].set_title('Personal Savings')
        axes[5].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axes[5].sharex(axes[0])
        #axes[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        if highlight_filter == True:
            for y in range(0,len(dates)):
                for x in range(0,var_count):
                    axes[x].axvline(dates[y],color='red', alpha=transparency)

        plt.tight_layout(h_pad=0.5)
        pdf.savefig()
        plt.show()
        plt.close()
    
#Page 8 Options & Forecast
try: #Error handling for stocks without options on yf (e.g. non us stocks)
    df_wavg_oi,df_top_oi,df_bot_iv,df_max_pain,df_options = get_options(stock)
    
    var_count = 3
    fig = plt.figure(figsize=(8.5, 11))
    gs = gridspec.GridSpec(var_count, 1, height_ratios=[1] + [1] * (var_count- 1))  # First subplot is larger
    axes = [fig.add_subplot(gs[i]) for i in range(var_count)]
    
    axes[0].plot(df_max_pain['Max Pain'],color='darkorange',label='Max Pain')
    axes[0].plot(df_wavg_oi['WAVG Price'],color='black',label='WAVG OI Price')
    axes[0].plot(df_wavg_oi['WAVG Call Price'],color='Green',label='WAVG OI Call Price')
    axes[0].plot(df_wavg_oi['WAVG Put Price'],color='Red',label='WAVG OI Put Price')
    axes[0].legend()
    axes[0].set_title(stock+' Options Forecast')
    axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    axes[1].scatter(df_top_oi[df_top_oi['PutCall']=='call'].index, df_top_oi[df_top_oi['PutCall']=='call']['Breakeven'],color='green')
    axes[1].scatter(df_top_oi[df_top_oi['PutCall']=='put'].index, df_top_oi[df_top_oi['PutCall']=='put']['Breakeven'],color='red')
    axes[1].set_title(stock+' Options Highest Open Interest')
    axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    axes[2].scatter(df_bot_iv[df_bot_iv['PutCall']=='call'].index, df_bot_iv[df_bot_iv['PutCall']=='call']['Breakeven'],color='green')
    axes[2].scatter(df_bot_iv[df_bot_iv['PutCall']=='put'].index, df_bot_iv[df_bot_iv['PutCall']=='put']['Breakeven'],color='red')
    axes[2].set_title(stock+' Options Lowest Implied Volatility')
    #axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    plt.tight_layout(h_pad=0.5)
    pdf.savefig()
    plt.show()
    plt.close()
    
    ###Price Charts
    if option_bin_size == None:
        option_bin_size = round(round(((df_options['Breakeven'].max() - df_options['Breakeven'].min()))/40,0)/5,0)*5
        
    price_today = df_stock['Close'][-1]
    df_options_pc = df_options[['strike','openInterest','PutCall']]
    df_options_pc['strike'] = (df_options_pc['strike']/option_bin_size).round(decimals=0)*option_bin_size
    df_options_pc_expiry = df_options_pc[(df_options_pc['strike']>(price_today-25*option_bin_size)) & (df_options_pc['strike']<(price_today+25*option_bin_size)) ] #Optional but adjustes the y-axis range for the plot.barh
    df_options_pc_expiry = df_options_pc_expiry.groupby(['strike','PutCall']).sum()
    df_options_pc_expiry.unstack().plot.barh(stacked=True,figsize=(8.5,11),title=stock +' Open Interest at Strike as of '+end.strftime("%Y-%m-%d"))
    plt.tight_layout(h_pad=0.5)
    pdf.savefig()
    plt.show()
    plt.close()
    
    df_options_strike = df_options[['strike','openInterest','Expiry']]
    df_options_strike['strike'] = (df_options_strike['strike']/option_bin_size).round(decimals=0)*option_bin_size
    df_options_strike['Expiry'] =  pd.PeriodIndex(df_options_strike['Expiry'], freq='Y')
    df_options_strike_expiry = df_options_strike[(df_options_strike['strike']>(price_today-25*option_bin_size)) & (df_options_strike['strike']<(price_today+25*option_bin_size)) ] #Optional but adjustes the y-axis range for the plot.barh
    df_options_strike_expiry = df_options_strike_expiry.groupby(['strike','Expiry']).sum()
    df_options_strike_expiry.unstack().plot.barh(stacked=True,figsize=(8.5,11),title=stock +' Open Interest at Strike as of '+end.strftime("%Y-%m-%d"))
    plt.tight_layout(h_pad=0.5)
    pdf.savefig()
    plt.show()
    plt.close()
except:
    pass

pdf.close()                                                                             

#Page 0 Summary
#Returns
total_return = df_stock['Close'][-1]/df_stock['Close'][0]-1
print('Total Return: '+f'{total_return:+.1%}')
total_days = (df_stock.index[-1]-df_stock.index[0]).days
return_cagr = (total_return+1)**(365/total_days)-1
print('CAGR: '+f'{return_cagr:+.1%}')

#Number of Drawdowns
count_below_threshold =(df_localdd_min['LocalDrawdown']<drawdown).sum()
print('Number of Local Drawdowns Below '+f'{drawdown:+.1%}'+': '+str(count_below_threshold))
count_below_5 =(df_localdd_min['LocalDrawdown']<-0.05).sum()
print('Number of Local Drawdowns Below '+f'{-0.05:+.1%}'+': '+str(count_below_threshold))
count_below_10 =(df_localdd_min['LocalDrawdown']<-0.1).sum()
print('Number of Local Drawdowns Below '+f'{-0.1:+.1%}'+': '+str(count_below_threshold))
count_below_20 =(df_localdd_min['LocalDrawdown']<-0.2).sum()
print('Number of Local Drawdowns Below '+f'{-0.2:+.1%}'+': '+str(count_below_threshold))

#Average Days in Drawdown
dd_current_days = (end - df_localdd_max.index[-1]).days
print('Current Days in Local Drawdown: '+str(dd_current_days))
dd_min_days_avg = df_localdd_min['Days'].mean()
print('Average Days Until Local Minimum: '+str(dd_min_days_avg))
dd_recovery_days_avg = df_localdd_max['Days'].mean()
print('Average Days of Recovery after Minimum: '+str(dd_recovery_days_avg))
dd_total_days_avg = df_localdd_max['Recovery Days'].mean()
print('Average Days Total Drawdown Days :'+str(dd_total_days_avg))
dd_min_days_median = df_localdd_min['Days'].median()
print('Average Days Until Local Minimum: '+str(dd_min_days_median))
dd_recovery_days_median = df_localdd_max['Days'].median()
print('Average Days of Recovery after Minimum: '+str(dd_recovery_days_median))
dd_total_days_median = df_localdd_max['Recovery Days'].median()
print('Average Days Total Drawdown Days :'+str(dd_total_days_median))

#Drawdown %
drawdown_local_avg = df_localdd_min['Drawdown'].mean()
print('Average Local Drawdown Return:'+f'{drawdown_local_avg:+.1%}')
drawdown_local_median = df_localdd_min['Drawdown'].median()
print('Median Local Drawdown Return:'+f'{drawdown_local_median:+.1%}')
drawdown_local_min = df_localdd_min['Drawdown'].min()
print('Max Drawdown:'+f'{drawdown_local_min:+.1%}')
drawdown_local_max = df_localdd_min['Drawdown'].max()
print('Min Drawdown:'+f'{drawdown_local_max:+.1%}')

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, stock+" Summary Statistics", 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 5, body)

pdf = PDF()
pdf.add_page()

pdf.chapter_title('Date: '+dt.datetime.now().strftime("%Y-%m-%d"))
pdf.chapter_body('For the Period of '+start.strftime("%Y-%m-%d")+' to '+end.strftime("%Y-%m-%d"))
pdf.chapter_body('Total Days in Period: '+str(total_days))
pdf.chapter_body("")

pdf.chapter_title('Returns')
pdf.chapter_body('Price: '+f'${df_stock["Close"][-1]:.2f}')
pdf.chapter_body('High: '+f'${df_stock["Close"].max():.2f}')
pdf.chapter_body('Low: '+f'${df_stock["Close"].min():.2f}')
pdf.chapter_body("")
pdf.chapter_body('YTD: '+f'{df_stock["YTD_Return"][-1]:+.1%}')
pdf.chapter_body('CAGR: '+f'{return_cagr:+.1%}')
pdf.chapter_body('Total Period Return: '+f'{total_return:+.1%}')
pdf.chapter_body("")
pdf.chapter_body('Average Daily Return: '+f'{df_stock["daily_pct_change"].mean():+.2%}')
pdf.chapter_body('Median Daily Return: '+f'{df_stock["daily_pct_change"].median():+.2%}')
pdf.chapter_body("")

pdf.chapter_title('Drawdowns')
pdf.chapter_body('Current Drawdown: '+f'{df_stock["Drawdown"][-1]:+.1%}')
pdf.chapter_body('Max Drawdown: '+f'{drawdown_local_min:+.1%}')
pdf.chapter_body('Min Drawdown: '+f'{drawdown_local_max:+.1%}')
pdf.chapter_body('Average Local Drawdown: '+f'{drawdown_local_avg:+.1%}')
pdf.chapter_body('Median Local Drawdown: '+f'{drawdown_local_median:+.1%}')
pdf.chapter_body("")
pdf.chapter_body('Number of Local Drawdowns Below '+f'{drawdown:+.1%}'+': '+str(count_below_threshold))
pdf.chapter_body('Number of Local Drawdowns Below '+f'{-0.05:+.1%}'+': '+str(count_below_5))
pdf.chapter_body('Number of Local Drawdowns Below '+f'{-0.1:+.1%}'+': '+str(count_below_10))
pdf.chapter_body('Number of Local Drawdowns Below '+f'{-0.2:+.1%}'+': '+str(count_below_20))
pdf.chapter_body("")
pdf.chapter_body('Current Days in Local Drawdown: '+str(dd_current_days))
pdf.chapter_body('Average Days Until Local Max Drawdown: '+f'{dd_min_days_avg:.1f}')
pdf.chapter_body('Average Days of Recovery Local Max Drawdown: '+f'{dd_recovery_days_avg:.1f}')
pdf.chapter_body('Average Days Total Drawdown Days:'+f'{dd_total_days_avg:.1f}')
pdf.chapter_body('Median Days Until Local Max Drawdown: '+str(dd_min_days_median))
pdf.chapter_body('Median Days of Recovery after Local Max Drawdown: '+str(dd_recovery_days_median))
pdf.chapter_body('Median Days Total Drawdown Days: '+str(dd_total_days_median))

pdf.output('summary.pdf')

merger = PdfMerger()
merger.append('summary.pdf')
merger.append(stock+" Report as of "+dt.datetime.now().strftime("%Y-%m-%d")+'.pdf')
merger.write(stock+" Report as of "+dt.datetime.now().strftime("%Y-%m-%d")+'_Final.pdf')
merger.close()

