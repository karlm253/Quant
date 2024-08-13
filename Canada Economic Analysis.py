# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 12:11:21 2024

@author: HB

V1: Added Page 1 Price, Page 2 Technicals, Page 3 Returns, Page 4 Correlations, Page 5 Options Chain

"""

import yfinance as yf
from Fred_API import fred_get_series
from Edgar_Company_Facts import get_cik, get_financial_data
import time
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
country = 'CA'
yf_1 = '^GSPTSE' #TSX Index
yf_2 = None
yf_3 = None
yf_4 = None
yf_5 = None
yf_6 = None
yf_7 = None
yf_8 = None #10Y
fred_1 = 'NGDPSAXDCCAQ' #GDP
fred_2 = 'NGDPRSAXDCCAQ' #Real GDP Growth
fred_3 = 'CANRECDM' #OECD Recession Indicator
fred_4 = 'PCAGDPCAA646NWDB' #GDP Per Capita
fred_5 = 'POPTOTCAA647NWDB' #Population

fred_6 = 'CANPRCNTO01GYSAM' #Total Production
fred_7 = 'CANPRINTO01GYSAM' #Total Production Except Construction
fred_8 = 'NCPRSAXDCCAQ' #Real Private Sector Final Consumption Expenditure for Canada
fred_9 = 'NFIRSAXDCCAQ' #Real Gross Fixed Capital Formation for Canada
fred_10 = 'NCGGRSAXDCCAQ' #Real General Government Final Consumption Expenditure for Canada 
fred_11 = 'NINVRSAXDCCAQ' #Real Changes in Inventories for Canada
fred_12 = 'NCRXDCCAA' #Real Final Consumption Expenditure for Canada

fred_13 = 'LRUN64TTCAM156S' #Unemployment Rate 15-64
fred_14 = 'LRUN25TTCAA156S' #Unemployment Rate 25-54
fred_15 = 'LFAC64TTCAM647S' #Labor Force Total 1564
fred_16 = 'LFWA64TTCAM647S' #Working Age Population
fred_18 = 'CCRETT02CAQ661N' #Labor Unit Costs

fred_19 = 'CPALTT01CAM659N' #CPI
fred_20 = 'CANCPGRLH01GYM' #CPI of Services Less Housing
fred_21 = 'CANCPGRGO01GYM' #CPI of Goods
fred_24 = 'CANPIEAMP01GPM' #PPI Manufacturing
fred_25 = 'IRSTCB01CAM156N' #Central Bank Rate
fred_26 = 'IRLTLT01CAM156N' #10Y Rate
fred_27 = 'CCUSSP01CAM650N' #USDCAD
fred_28 = 'CANGGXWDGGDP' #Government Gross Debt
fred_29 = 'TOTDTECAQ163N' #Canada Debt to Equity
fred_30 = 'MANMM101CAM189S' #M1
fred_31 = 'MABMM301CAM189S' #M3

fred_32 = 'QCAR628BIS' #Real Residential Property Prices
fred_33 = 'QCAN628BIS' #Residential Property Prices
fred_34 = 'HDTGPDCAQ163N' #Household Debt to GDP
fred_35 = 'ODCNPI03CAA189S' #Orders: Construction: Permits Issued: Dwellings and Residential Buildings for Canada
fred_17 = 'WSCNDW01CAQ489S' #Work Started Residential
fred_22 = 'CANCP040200IXOBM' #CPI of Imputed Rentals for Housing
fred_23 = 'CANCPIHOUMINMEI' #CPI Housing

backtest = fred_1

sns.set_theme(rc={'figure.figsize':(8.5,11)})
x_axis_multiple = max(round((look_back+look_forward)/20),1)


################################

#Creates a list of varibles where there are valid inputs i.e. non zero.
list_yf = [yf_1, yf_2, yf_3, yf_4, yf_5,yf_6,yf_7,yf_8]
list_yf = [var for var in list_yf if var]
list_fred = [fred_1, fred_2, fred_3, fred_4, fred_5,fred_6,fred_7,fred_8,fred_8,fred_10,fred_11,fred_12,fred_13,fred_14,fred_15,fred_16,fred_17,fred_18,fred_19,fred_20,fred_21,fred_22,fred_23,fred_24,fred_25,fred_26,fred_27,fred_28,fred_29,fred_30,fred_31,fred_32,fred_33,fred_34,fred_35]
list_fred = [var for var in list_fred if var]
list_all_variables= list_yf+list_fred

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

def date_drawdown_level(df,threshold_high,threshold_low,period_chg=1):
    threshold_list = df[(df['Close']<threshold_high) & (df['Close']>threshold_low)]
    list_dates = threshold_list.index.tolist()
    clean_dates = []
    clean_dates.append(list_dates[0])
    return clean_dates
def date_first_rate_cut():
    fedfunds = fred_get_series('FEDFUNDS')
    fedfunds.index = fedfunds['date']
    fedfunds.index = pd.to_datetime(fedfunds.index)
    fedfunds = fedfunds['value']
    fedfunds = pd.to_numeric(fedfunds)
    
    df_fedfunds = pd.DataFrame(fedfunds)
    df_rates = df_fedfunds.join(df_backtest .Close).dropna(how='any')
    df_rates = df_rates.rename(columns={"value": "FedFunds", "Close": "US10Y"})
    
    
    threshold_list = df_rates[(df_rates.FedFunds.shift(1)-df_rates.FedFunds > 0.25)&(df_rates.FedFunds.rolling(12).max() - df_rates.FedFunds.shift(1)<0.25)] #First Rate Hike, Filters out rolling 12M noise 
    #threshold_list = df_rates[(df_rates.FedFunds.shift(1)-df_rates.FedFunds > 0.25)&(df_rates.FedFunds.rolling(12).max() - df_rates.FedFunds.shift(1)<0.25)&(df_rates.FedFunds > df_rates.US10Y)]  #First Cut with no changes in prior 12M and Fedfunds > US10Y
    #threshold_list = df_rates[(df_rates.FedFunds.shift(1)-df_rates.FedFunds > 0.25)] #any rate cut not just fisrt rate hike   
    list_dates = threshold_list.index.tolist()
    return list_dates

#Get the data for the backtest variable. Change bbased on Data Source###############
list_df = {}
for x in list_yf:
    list_df[x] = get_yf_data(x,start-dt.timedelta(days=365),end)
    time.sleep(1)
for x in list_fred:
    list_df[x] = get_fred_data(x) 

#Transformations
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

def transform_data(df):
    df['daily_pct_change'] = df.Close.pct_change(periods=1)
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
    df=df_backtest 
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

df_backtest = list_df[backtest]
df_benchmark1 = transform_data(list_df[yf_1])

          
#Get Dates#####################################################################
#dates = df_dd_start.index.tolist()
#dates = date_drawdown_pct(df_backtest ,threshold_high=threshold_high,threshold_low=threshold_low,period_chg=period_chg)
#transparency= max(0.8 - len(dates)/100,0.15)

pdf = PdfPages(country+" Report as of "+dt.datetime.now().strftime("%Y-%m-%d")+'.pdf')
#Page 1: Summary
var_count = 6
fig = plt.figure(figsize=(8.5, 11))
gs = gridspec.GridSpec(var_count, 1, height_ratios=[2] + [1] * (var_count- 1))  # First subplot is larger
axes = [fig.add_subplot(gs[i]) for i in range(var_count)]

axes[0].plot(list_df[fred_1]['Close'][start:],color='cornflowerblue',label='GDP')
axes[0].plot(list_df[fred_2]['Close'][start:],color='darkorange',label='Real GDP')
axes[0].set_title('GDP')
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[0].set_xlim(start,end)
axes[0].sharex(axes[0])

axes[1].plot(list_df[fred_4]['Close'][start:],color='cornflowerblue',label='GDP')
axes[1].set_title('GDP Per Capita')
axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[1].sharex(axes[0])

axes[2].plot(list_df[fred_13]['Close'][start:],color='cornflowerblue')
axes[2].set_title('Unemployment Rate')
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[2].sharex(axes[0])

axes[3].plot(list_df[fred_19]['Close'][start:],color='cornflowerblue')
axes[3].set_title('CPI')
axes[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[3].sharex(axes[0])

axes[4].plot(list_df[fred_25]['Close'][start:],color='cornflowerblue')
axes[4].set_title('Central Bank Rate')
axes[4].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[4].sharex(axes[0])

axes[5].plot(list_df[fred_27]['Close'][start:],color='cornflowerblue')
axes[5].set_title('USDCAD')
#axes[5].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[5].sharex(axes[0])

fig.suptitle(country+" Report as of "+dt.datetime.now().strftime("%Y-%m-%d"))

plt.tight_layout(h_pad=0.5)
pdf.savefig()
plt.show()
plt.close()

#Page 2: GDP
var_count = 4
fig = plt.figure(figsize=(8.5, 11))
gs = gridspec.GridSpec(var_count, 1, height_ratios=[2] + [1] * (var_count- 1))  # First subplot is larger
axes = [fig.add_subplot(gs[i]) for i in range(var_count)]

axes[0].plot(list_df[fred_1]['Close'][start:]/list_df[fred_1]['Close'][start:].shift(1)-1,color='cornflowerblue',label='GDP')
axes[0].plot(list_df[fred_2]['Close'][start:]/list_df[fred_2]['Close'][start:].shift(1)-1,color='darkorange',label='Real GDP')
axes[0].set_title('GDP Growth')
axes[0].legend()
axes[0].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[0].set_xlim(start,end)
axes[0].set_ylim(-0.03,+0.03)
axes[0].sharex(axes[0])
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[1].plot(list_df[fred_4]['Close'][start:]/100,color='cornflowerblue')
axes[1].set_title('GDP Per Capita Growth')
axes[1].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[1].sharex(axes[0])

axes[2].plot(list_df[fred_6]['Close'][start:]/100,color='cornflowerblue',label='Total Production')
axes[2].plot(list_df[fred_7]['Close'][start:]/100,color='darkorange',label='Production Ex Construction')
axes[2].set_title('Production Growth')
axes[2].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[2].legend()
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[2].sharex(axes[0])

axes[3].plot(list_df[fred_12]['Close'][start:]/list_df[fred_12]['Close'][start:].shift(1)-1,color='black',label='Total')
axes[3].plot(list_df[fred_8]['Close'][start:]/list_df[fred_8]['Close'][start:].shift(1)-1,color='green',label='Private')
axes[3].plot(list_df[fred_10]['Close'][start:]/list_df[fred_10]['Close'][start:].shift(1)-1,color='red',label ='Govt')
axes[3].legend()
axes[3].set_title('Real Consuption Expenditure')
axes[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[3].sharex(axes[0])
"""
axes[4].plot(list_df[fred_9]['Close'][start:],color='cornflowerblue')
axes[4].set_title('GDP Per Capita')
#axes[4].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[4].sharex(axes[0])
"""
plt.tight_layout(h_pad=0.5)
pdf.savefig()
plt.show()
plt.close()


#Page 3: Inflation & Debt
var_count = 5
fig = plt.figure(figsize=(8.5, 11))
gs = gridspec.GridSpec(var_count, 1, height_ratios=[2] + [1] * (var_count- 1))  # First subplot is larger
axes = [fig.add_subplot(gs[i]) for i in range(var_count)]

axes[0].plot(list_df[fred_19]['Close'][start:]/100,color='cornflowerblue',label='CPI')
axes[0].plot(list_df[fred_20]['Close'][start:]/100,color='red',label='CPI Services')
axes[0].plot(list_df[fred_21]['Close'][start:]/100,color='green',label='CPI Goods')
axes[0].axhline(0.02,color='black')
axes[0].set_title('CPI')
axes[0].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[0].set_xlim(start,end)
axes[0].sharex(axes[0])
axes[0].legend()
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[1].plot(list_df[fred_24]['Close'][start:]/100,color='cornflowerblue',label='GDP')
axes[1].axhline(0.02,color='black')
axes[1].set_title('PPI')
axes[1].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[1].sharex(axes[0])
axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[2].plot(list_df[fred_25]['Close'][start:]/100,color='cornflowerblue',label='Overnight Rate')
axes[2].plot(list_df[fred_26]['Close'][start:]/100,color='darkorange',label='10Y Rate')
axes[2].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[2].set_title('Rates')
axes[2].legend()
axes[2].sharex(axes[0])
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[3].plot(list_df[fred_30]['Close'][start:],color='cornflowerblue',label='M1')
axes[3].plot(list_df[fred_31]['Close'][start:],color='darkorange',label='M3')
axes[3].set_title('Money Supply')
axes[3].legend()
axes[3].sharex(axes[0])
axes[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[4].plot(list_df[fred_28]['Close'][start:]/100,color='cornflowerblue',label='Govt Debt to GDP')
axes[4].plot(list_df[fred_29]['Close'][start:]/100,color='darkorange',label='Govt Debt to Equity')
axes[4].axhline(1,color='black')
axes[4].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[3].legend()
axes[4].set_title('Government Debt')
axes[4].sharex(axes[0])
#axes[4].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

plt.tight_layout(h_pad=0.5)
pdf.savefig()
plt.show()
plt.close()

#Page 4: Unemployment Rate
var_count = 5
fig = plt.figure(figsize=(8.5, 11))
gs = gridspec.GridSpec(var_count, 1, height_ratios=[2] + [1] * (var_count- 1))  # First subplot is larger
axes = [fig.add_subplot(gs[i]) for i in range(var_count)]

axes[0].plot(list_df[fred_13]['Close'][start:],color='cornflowerblue',label='15-64')
axes[0].plot(list_df[fred_14]['Close'][start:],color='darkorange',label='24-54')
axes[0].legend()
axes[0].set_title('Unemployment Rate')
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[0].set_xlim(start,end)
axes[0].sharex(axes[0])

axes[1].plot(list_df[fred_15]['Close'][start:],color='cornflowerblue',label='Labor Force')
axes[1].plot(list_df[fred_16]['Close'][start:],color='black',label='Working Age Population')
axes[1].legend()
axes[1].set_title('Labor Force')
axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[1].sharex(axes[0])

axes[2].plot(list_df[fred_5]['Close'][start:],color='cornflowerblue',label='GDP')
axes[2].set_title('Population')
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[2].sharex(axes[0])

axes[3].plot(list_df[fred_4]['Close'][start:],color='cornflowerblue',label='GDP')
axes[3].set_title('GDP Per Capita')
axes[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[3].sharex(axes[0])

axes[4].plot(list_df[fred_18]['Close'][start:],color='cornflowerblue',label='GDP')
axes[4].set_title('Labor Unit Costs')
#axes[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[4].sharex(axes[0])

plt.tight_layout(h_pad=0.5)
pdf.savefig()
plt.show()
plt.close()

#Page 5: Real Estate
var_count = 5
fig = plt.figure(figsize=(8.5, 11))
gs = gridspec.GridSpec(var_count, 1, height_ratios=[2] + [1] * (var_count- 1))  # First subplot is larger
axes = [fig.add_subplot(gs[i]) for i in range(var_count)]

axes[0].plot(list_df[fred_32]['Close'][start:],color='darkorange',label='Residential Real Estate Prices')
axes[0].plot(list_df[fred_33]['Close'][start:],color='cornflowerblue',label='Real Residential Real Estate Prices')
axes[0].legend()
axes[0].set_title('Residential Real Estate Prices')
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[0].set_xlim(start,end)
axes[0].sharex(axes[0])

axes[1].plot(list_df[fred_22]['Close'][start:]/list_df[fred_22]['Close'][start:].shift(1)-1,color='cornflowerblue',label='CPI Imputed Rent Prices')
axes[1].set_title('CPI Imputed Rent Prices')
axes[1].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[1].sharex(axes[0])

axes[2].plot(list_df[fred_35]['Close'][start:],color='cornflowerblue',label='Real Residential Real Estate Prices')
axes[2].set_title('Permits Issued Dwellings and Residential Buildings')
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[2].sharex(axes[0])

axes[3].plot(list_df[fred_17]['Close'][start:],color='cornflowerblue')
axes[3].set_title('Work Started Residential')
axes[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[3].sharex(axes[0])

axes[4].plot(list_df[fred_34]['Close'][start:]/100,color='cornflowerblue')
axes[4].set_title('Household Debt to GDP')
axes[4].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[4].sharex(axes[0])
#axes[4].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

plt.tight_layout(h_pad=0.5)
pdf.savefig()
plt.show()
plt.close()

pdf.close()                    

                                                         
"""
#Page 0 Summary
#Returns
total_return = df_backtest ['Close'][-1]/df_backtest ['Close'][0]-1
print('Total Return: '+f'{total_return:+.1%}')
total_days = (df_backtest .index[-1]-df_backtest .index[0]).days
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
        self.cell(0, 10, backtest+" Summary Statistics", 0, 1, 'C')

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
pdf.chapter_body('Price: '+f'${df_backtest ["Close"][-1]:.2f}')
pdf.chapter_body('High: '+f'${df_backtest ["Close"].max():.2f}')
pdf.chapter_body('Low: '+f'${df_backtest ["Close"].min():.2f}')
pdf.chapter_body("")
pdf.chapter_body('YTD: '+f'{df_backtest ["YTD_Return"][-1]:+.1%}')
pdf.chapter_body('CAGR: '+f'{return_cagr:+.1%}')
pdf.chapter_body('Total Period Return: '+f'{total_return:+.1%}')
pdf.chapter_body("")
pdf.chapter_body('Average Daily Return: '+f'{df_backtest ["daily_pct_change"].mean():+.2%}')
pdf.chapter_body('Median Daily Return: '+f'{df_backtest ["daily_pct_change"].median():+.2%}')
pdf.chapter_body("")

pdf.chapter_title('Drawdowns')
pdf.chapter_body('Current Drawdown: '+f'{df_backtest ["Drawdown"][-1]:+.1%}')
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
merger.append(backtest+" Report as of "+dt.datetime.now().strftime("%Y-%m-%d")+'.pdf')
merger.write(backtest+" Report as of "+dt.datetime.now().strftime("%Y-%m-%d")+'_Final.pdf')
merger.close()
"""
