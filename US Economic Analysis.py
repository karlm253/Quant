# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 18:56:08 2024

@author: Heavens Base
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
country = 'US'
highlight_recession = True
highlight_filter = False
edgar_data = False
option_bin_size = None #Auto sets if none
pdf = PdfPages("US Economic Report as of "+dt.datetime.now().strftime("%Y-%m-%d")+'.pdf')

end = dt.datetime.now() 
start = dt.datetime(1980,1,1)
look_forward=20
look_back=5

yf_1 = '^SPX' #S&P 500 Cap Weight
yf_2 = '^SPXEW' #S&P Equal Weight
yf_3 = '^RUT' #Russell 2000
yf_4 = '^NDX' #Nasdaq
yf_5 = '^VIX' #CBOE Vix Index
yf_6 = 'V' #Visa, customer spending
yf_7 = '^DJT' #Dow Jones Transportation Average
yf_8 = '^BKX' #KBW Nasdaq Bank Index

fred_1 = 'GDP' #Nominal GDP
fred_2 = 'GDPC1' #Real GDP
fred_3 = 'GDPNOW' #GDPNOW
fred_4 = 'GNP' #Gross National Product
fred_5 = 'PCEPILFE' #Core PCE Index
fred_6 = 'CORESTICKM159SFRBATL' #CORE CPI
fred_7 = 'T10YIE' #10Y Breakeven Inflation Rate
fred_8 = 'DFF' #Fed Funds Rate
fred_9 = 'DGS2' # US2Y
fred_10 = 'DGS10' # US10Y

fred_11 = 'BAMLC0A1CAAAEY' #ICE AAA Effective Yield
fred_12 = 'BAMLH0A0HYM2EY' #ICE High Yield
fred_13 = 'BAMLH0A3HYC' #ICE CCC and Lower
fred_14 = 'BSCICP02USM460S' #Business Tendency Surveys (Manufacturing) National Indicator for United States
fred_15 = 'UMCSENT' #University of Michigan: Consumer Sentiment

fred_16 = 'UNRATE' #Unemployment Rate (U3)
fred_17 = 'UNRATENSA' #Unemployment Rate Non Seasonally Adjusted
fred_18 = 'ADPWNUSNERSA' # Total Nonfarm Private Payroll Employment
fred_19 = 'PAYEMS' #Non Farm Payrolls
fred_20 = 'ICSA' #Initial Claims
fred_21 = 'CCSA' #Continued Claims (Insured Unemployment)
fred_22 = 'UEMPMEAN' #Average Weeks Unemployed
fred_23 = 'UEMP27OV' #Number Unemployed for 27 Weeks & over 
fred_24 = 'CIVPART' #Labor Force Participation
fred_25 = 'ATLSBUEGEP' #Business Expectations: Employment Growth
fred_26 = 'CFSBCHIRINGEXP' #Chicago Fed Survey of Economic Conditions: Hiring Expectations in the next 12 Months in Federal Reserve District 7: Chicago
fred_27 = 'AWHAETP' #Average Weekly Hours of All Employees, Total Private
fred_28 = 'ECIWAG' #Employment Cost Index: Wages and Salaries: Private Industry Workers

fred_29 = 'LES1252881600Q' #Employed full time: Median usual weekly real earnings
fred_30 = 'DSPIC96' #Real Disposable Personal Income. No Real Signal.
fred_31 = 'LNS12026620' #Multiple Jobholders as a Percent of Employed
fred_32 = 'REVOLSLAR' #Percentage Change in Total Revolving Consumer Credit. Signal is if it is negative.
fred_33 = 'TDSP' #Household Debt Service Payments as a Percent of Disposable Personal Income. No clear signal but in theory increasing trend is bad.
fred_34 = 'DRCCLACBS' #Delinquency Rate on Credit Card Loans. Signal is probably Above 4.5%
fred_35 = 'PSAVERT' #Personal Savings Rate. Signal seems to be when the rate declines or is under 5%.
fred_36 = 'CSUSHPINSA' #Cash-Shiller US National Hope Price Index. For Wealth Effect.

fred_37 = 'A053RC1Q027SBEA' #National income: Corporate profits before tax (without IVA and CCAdj)
fred_38 = 'ATLSBUSRGEP' #Business Expectations: Sales Revenue Growth
fred_39 = 'CFSBCCAPXEXP' #Chicago Fed Survey of Economic Conditions: Capital Spending Expectations in the next 12 Months in Federal Reserve District 7: Chicago (CFSBCCAPXEXP) 
fred_40 = 'BUSFIXINVESTNOW' #Nowcast for Real Gross Private Domestic Investment: Fixed Investment: Business 
fred_41 = 'PRS85006091' #Nonfarm Business Sector: Labor Productivity
fred_42 = 'ISRATIO' #Total Business: Inventories to Sales Ratio
fred_43 = 'BUSLOANS' #Commercial and Industrial Loans, All Commercial Banks
fred_44 = 'DRBLACBS' #Delinquency Rate on Business Loans

fred_45 = None
fred_46 = None
fred_47 = None
fred_48 = None

sns.set_theme(rc={'figure.figsize':(8.5,11)})
x_axis_multiple = max(round((look_back+look_forward)/20),1)

#Creates a list of varibles where there are valid inputs i.e. non zero. ###############################
list_yf = [yf_1, yf_2, yf_3, yf_4, yf_5,yf_6,yf_7,yf_8]
list_yf = [var for var in list_yf if var]
list_fred = [fred_1, fred_2, fred_3, fred_4, fred_5,fred_6,fred_7,fred_8,fred_9,fred_10,fred_11,fred_12,fred_13,fred_14,fred_15,fred_16,fred_17,fred_18,fred_19,fred_20,fred_21,fred_22,fred_23,fred_24,fred_25,fred_26,fred_27,fred_28,fred_29,fred_30,fred_31,fred_32,fred_33,fred_34,fred_35,fred_36,fred_37,fred_38,fred_39,fred_40,fred_41,fred_42,fred_43,fred_44,fred_45,fred_46,fred_47,fred_48]
list_fred = [var for var in list_fred if var]
list_all_variables= list_yf+list_fred

def get_yf_data (stock,start,end):
    df = pdr.get_data_yahoo(stock,start,end)
    df = df.reindex(pd.date_range(start, end, freq='D')).dropna(axis=0, how = 'any')
    time.sleep(0.1)
    return df
def get_fred_data(backtest_fred):
    df = fred_get_series(backtest_fred)
    df.index = df['date']
    df.index = pd.to_datetime(df.index)
    df['Close'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.reindex(pd.date_range(start, end, freq='D')).dropna(axis=0, how = 'any')
    time.sleep(1)
    return df


def calc_ema(df, span):
    return df.ewm(span=span, adjust=False).mean()
def calc_macd(df):
    short_ema = calc_ema(df['Close'], 12)
    long_ema = calc_ema(df['Close'], 26)
    macd = short_ema - long_ema
    signal = calc_ema(macd, 9)
    histogram = macd - signal
    return macd, signal, histogram
def calc_rsi(df,window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
def transform_yf(df):
    df['daily_pct_change'] = df.Close.pct_change(periods=1)
    df['annual_pct_change'] = df.Close.pct_change(periods=253)
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
def transform_fred(df):
    #df=list_df['UNRATE']
    df['Date_DT'] = df.index
    df['Period_Days'] = (df['Date_DT']-df['Date_DT'].shift(1)).dt.days
    df['Period_Days'][3]
    df['PoP Change Value'] = df['Close']-df['Close'].shift(1)-1
    df['PoP Change Pct'] = df['Close']/df['Close'].shift(1)-1
    df['PoP Change Pct Annualized'] = (df['Close']/df['Close'].shift(1))**(365/df['Period_Days'])-1
    df['12M_Entries'] = round(365/df['Period_Days'].mean())
    df['Min_LTM'] = df['Close'].rolling(window=df['12M_Entries'][0]).min()
    df['Max_LTM'] = df['Close'].rolling(window=df['12M_Entries'][0]).max()
    return df

#Get the data for the backtest variable. Change bbased on Data Source###############
list_df = {}
for x in list_yf: #Gets Yahoo Finance Data
    list_df[x] = get_yf_data(x,start-dt.timedelta(days=365),end)
    list_df[x] = transform_yf(list_df[x])
for x in list_fred: #Gets Fred Data
    list_df[x] = get_fred_data(x)
    list_df[x] = transform_fred(list_df[x])

#Get Dates#####################################################################
#dates = df_dd_start.index.tolist()
#dates = date_drawdown_pct(df_backtest ,threshold_high=threshold_high,threshold_low=threshold_low,period_chg=period_chg)
#transparency= max(0.8 - len(dates)/100,0.15)

#pdf = PdfPages(country+" Report as of "+dt.datetime.now().strftime("%Y-%m-%d")+'.pdf')

#Page 1: Summary / Markets##############################################################
var_count = 5
fig = plt.figure(figsize=(8.5, 11))
gs = gridspec.GridSpec(var_count, 1, height_ratios=[1] + [1] * (var_count- 1))  # First subplot is larger
axes = [fig.add_subplot(gs[i]) for i in range(var_count)]

std_gdp = list_df[fred_1]['PoP Change Pct Annualized'][start:].std()
axes[0].plot(list_df[fred_1]['PoP Change Pct Annualized'][start:],label='GDP')
axes[0].plot(list_df[fred_2]['PoP Change Pct Annualized'][start:],label='Real GDP')
axes[0].plot(list_df[fred_3]['Close'][start:]/100,label='GDPNOW')
axes[0].plot(list_df[fred_4]['PoP Change Pct Annualized'][start:]/100,label='GNP')
axes[0].set_title('GDP Change PoP Annualized, Highlight Negative Real GDP')
axes[0].set_xlim(start,end)
axes[0].set_ylim(list_df[fred_1]['PoP Change Pct Annualized'][-1]-2*std_gdp,list_df[fred_1]['PoP Change Pct Annualized'][-1]+2*std_gdp)
axes[0].sharex(axes[0])
axes[0].yaxis.set_major_formatter(PercentFormatter(1,decimals=2))
axes[0].legend()
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

dates_realgdp = list_df[fred_2]
dates_realgdp = dates_realgdp[dates_realgdp['PoP Change Pct']<0]['Date_DT'] # Shades Dates where Real GDP is Negative
for x in range(0,len(dates_realgdp)):
    axes[0].axvline(dates_realgdp[x],color='black', alpha=0.7)

axes[1].plot(list_df[yf_1]['annual_pct_change'][start:],label='S&P500 Cap Weight')
axes[1].plot(list_df[yf_2]['annual_pct_change'][start:],label='S&P500 Equal Weight',alpha=0.5)
axes[1].plot(list_df[yf_3]['annual_pct_change'][start:],label='Russell 2000',alpha=0.5)
axes[1].plot(list_df[yf_4]['annual_pct_change'][start:],label='Nasdaq 100',alpha=0.5)
axes[1].set_title('Stock Market Rolling Annual Return, Highlight 20% Drawdown from Peak')
axes[1].sharex(axes[0])
axes[1].yaxis.set_major_formatter(PercentFormatter(1,decimals=2))
axes[1].legend()
axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

dates_stockcrash= list_df[yf_1]
dates_stockcrash = dates_stockcrash[dates_stockcrash['Drawdown']<-0.20].index #Shades Dates where SPX saw at least a 20% drawdown
for x in range(0,len(dates_stockcrash)):
    axes[1].axvline(dates_stockcrash[x],color='red', alpha=0.01)

axes[2].plot(list_df[yf_5]['Close'][start:],label='VIX')
axes[2].sharex(axes[0])
axes[2].set_title('VIX Index, Highlight Vix Above 30')
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

dates_vix = list_df[yf_5]
dates_vix = dates_vix[dates_vix['Close']>30].index #Shades Dates where Vix is Above 21
for x in range(0,len(dates_vix)):
    axes[2].axvline(dates_vix[x],color='red', alpha=0.02)

axes[3].plot(list_df[fred_5]['PoP Change Pct Annualized'][start:],label='Core PCE') 
axes[3].plot(list_df[fred_6]['Close'][start:]/100,label='Core CPI',alpha=0.5) #Percent not Index
axes[3].plot(list_df[fred_7]['Close'][start:]/100,label='10Y BE Inflation',alpha=0.5)
axes[3].set_title('Inflation PoP Annualized, Highlight Greater than 3% Core PCE')
axes[3].sharex(axes[0])
axes[3].yaxis.set_major_formatter(PercentFormatter(1,decimals=2))
axes[3].legend()
axes[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

dates_inflation3pct = list_df[fred_5]
dates_inflation3pct = dates_inflation3pct[dates_inflation3pct['PoP Change Pct Annualized']>0.03]['Date_DT'] # Shades Dates where Core PCE is above 3% (Upper Target Range)
for x in range(0,len(dates_inflation3pct)):
    axes[3].axvline(dates_inflation3pct[x],color='red', alpha=0.3)

axes[4].plot(list_df[fred_8]['Close'][start:]/100,label='Fed Funds Rate')
axes[4].plot(list_df[fred_9]['Close'][start:]/100,label='US02Y Rate',alpha=0.5)
axes[4].plot(list_df[fred_10]['Close'][start:]/100,label='US10Y Rate',alpha=0.5)
axes[4].set_title('US Interest Rates, Highlight 10Y < 02Y')
axes[4].sharex(axes[0])
axes[4].yaxis.set_major_formatter(PercentFormatter(1,decimals=2))
axes[4].legend()
#axes[4].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

dates_rateinvert = list_df[fred_10]['Close'] - list_df[fred_8]['Close']
dates_rateinvert = dates_rateinvert[dates_rateinvert<0].index # Shades Dates where the US02Y is below the FedFunds Rate
for x in range(0,len(dates_rateinvert)):
    axes[4].axvline(dates_rateinvert[x],color='red', alpha=0.01)

if highlight_recession == True:
    for x in range(0,len(dates_realgdp)):
        for ax in axes[1:]:
            ax.axvline(dates_realgdp[x],color='black', alpha=0.7)

fig.suptitle(country+" Economic Report as of "+dt.datetime.now().strftime("%Y-%m-%d"))

plt.tight_layout(h_pad=0.5)
pdf.savefig()
plt.show()
plt.close()

#Page 2: Market Based Leading Indicators ####################################################
var_count = 5
fig = plt.figure(figsize=(8.5, 11))
gs = gridspec.GridSpec(var_count, 1, height_ratios=[1] + [1] * (var_count- 1))  # First subplot is larger
axes = [fig.add_subplot(gs[i]) for i in range(var_count)]

#Companies that are leading indicators. Homebuilders. Transportation. Banks. Russell. Consumer Discretionary and Staples.
axes[0].plot((list_df[yf_3]['200D Return'][start:]+1)/(list_df[yf_1]['200D Return'][start:]+1),label=yf_3,linewidth=0.7,alpha=1)
#axes[0].plot((list_df[yf_7]['200D Return'][start:]+1)/(list_df[yf_1]['200D Return'][start:]+1),label=yf_7,linewidth=0.7,alpha=1)
#axes[0].plot((list_df[yf_8]['200D Return'][start:]+1)/(list_df[yf_1]['200D Return'][start:]+1),label=yf_8,linewidth=0.7,alpha=1)
axes[0].set_title('Small Cap Relative Returns vs S&P500')
axes[0].set_xlim(start,end)
axes[0].sharex(axes[0])
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

dates_stock1 = (list_df[yf_3]['200D Return'][start:]+1)/(list_df[yf_1]['200D Return'][start:]+1)
dates_stock1 = dates_stock1[dates_stock1<0.9].index
for x in range(0,len(dates_stock1)):
    axes[0].axvline(dates_stock1[x],color='red', alpha=0.02)

axes[1].plot((list_df[yf_7]['200D Return'][start:]+1)/(list_df[yf_1]['200D Return'][start:]+1),label=yf_7,linewidth=0.7,alpha=1)
#axes[1].plot((list_df[yf_8]['200D Return'][start:]+1)/(list_df[yf_1]['200D Return'][start:]+1),label=yf_8,linewidth=0.7,alpha=1)
axes[1].set_title('DJ Transport Relative Returns vs S&P500')
axes[1].set_xlim(start,end)
axes[1].sharex(axes[0])
axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

dates_stock2 = (list_df[yf_7]['200D Return'][start:]+1)/(list_df[yf_1]['200D Return'][start:]+1)
dates_stock2 = dates_stock2[dates_stock2<0.9].index
for x in range(0,len(dates_stock2)):
    axes[1].axvline(dates_stock2[x],color='red', alpha=0.02)
    
axes[2].plot((list_df[yf_8]['200D Return'][start:]+1)/(list_df[yf_1]['200D Return'][start:]+1),label=yf_7,linewidth=0.7,alpha=1)
#axes[1].plot((list_df[yf_8]['200D Return'][start:]+1)/(list_df[yf_1]['200D Return'][start:]+1),label=yf_8,linewidth=0.7,alpha=1)
axes[2].set_title('Bank Index Relative Returns vs S&P500')
axes[2].set_xlim(start,end)
axes[2].sharex(axes[0])
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

dates_stock3 = (list_df[yf_8]['200D Return'][start:]+1)/(list_df[yf_1]['200D Return'][start:]+1)
dates_stock3 = dates_stock3[dates_stock3<0.8].index
for x in range(0,len(dates_stock3)):
    axes[2].axvline(dates_stock3[x],color='red', alpha=0.02)

#Corporate Bond Spreads
axes[3].plot(list_df[fred_10]['Close'][start:]/100,label='US10Y')
axes[3].plot(list_df[fred_11]['Close'][start:]/100,label='AAA')
axes[3].plot(list_df[fred_12]['Close'][start:]/100,label='High Yield')
axes[3].plot(list_df[fred_13]['Close'][start:]/100,label='CCC & Lower')
axes[3].set_title('Credit Effective Yields')
axes[3].sharex(axes[0])
axes[3].yaxis.set_major_formatter(PercentFormatter(1,decimals=2))
axes[3].legend()
axes[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

dates_creditspread = list_df[fred_13]['Close'] - list_df[fred_11]['Close']
dates_creditspread = dates_creditspread[(dates_creditspread-dates_creditspread.rolling(window=20).min())>1].index
for x in range(0,len(dates_creditspread)):
    axes[3].axvline(dates_creditspread[x],color='red', alpha=0.02)

#Employment & Sahm Rule
axes[4].plot(list_df[fred_16]['Close'][start:]/100,label='Unemployment Rate SA')
axes[4].plot(list_df[fred_17]['Close'][start:]/100,label='Unemployment Rate NSA', alpha=0.5)
axes[4].set_title('Unemployment Rate, Highlight Sahm Rule')
axes[4].sharex(axes[0])
axes[4].yaxis.set_major_formatter(PercentFormatter(1,decimals=2))
axes[4].legend()
#axes[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

dates_sahm = list_df[fred_16]
dates_sahm['Sahm_Rule'] =  dates_sahm['Close'] - dates_sahm['Min_LTM']
dates_sahm = dates_sahm[dates_sahm['Sahm_Rule']>0.5]['Date_DT'] # Shades Dates where the Sahm Rule Triggers which is +50bps of unemployment from min on a LTM basis
for x in range(0,len(dates_sahm)):
    axes[4].axvline(dates_sahm[x],color='red', alpha=0.3)

if highlight_recession == True:
    for x in range(0,len(dates_realgdp)):
        for ax in axes:
            ax.axvline(dates_realgdp[x],color='black', alpha=0.7)

fig.suptitle("Market Based Leading Indicators")
plt.tight_layout(h_pad=0.5)
pdf.savefig()
plt.show()
plt.close()

"""
#Page 3: Market Based Leading Indicators ####################################################
#Business Confidence
axes[3].plot(list_df[fred_13]['Close'][start:]+100,label='Business Tendency Surveys (Manufacturing)')
axes[3].plot(list_df[fred_14]['Close'][start:],label='Michigan Consumer Sentiment')
axes[3].set_title('Sentiment')
axes[3].set_xlim(start,end)
axes[3].sharex(axes[0])
axes[3].legend()
axes[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

dates_businessconfidence = list_df[fred_13]
dates_businessconfidence = dates_businessconfidence[dates_businessconfidence['Close']<-10]['Date_DT'] # Shades Dates where the Sahm Rule Triggers which is +50bps of unemployment from min on a LTM basis
for x in range(0,len(dates_businessconfidence)):
    axes[3].axvline(dates_businessconfidence[x],color='red', alpha=0.3)
"""

#Page 4: Employment ####################################################################
var_count = 6
fig = plt.figure(figsize=(8.5, 11))
gs = gridspec.GridSpec(var_count, 1, height_ratios=[1] + [1] * (var_count- 1))  # First subplot is larger
axes = [fig.add_subplot(gs[i]) for i in range(var_count)]

axes[0].plot(list_df[fred_19]['Close'][start:],label='Non Farm Payrolls')
axes[0].set_xlim(start,end)
axes[0].set_title('NonFarm Payrolls')
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

std_initialclaims = list_df[fred_20]['Close'].std()
avg_initialclaims = list_df[fred_20]['Close'].mean()
axes[1].plot(list_df[fred_20]['Close'][start:])
axes[1].set_title('Initial Claims')
axes[1].sharex(axes[0])
axes[1].set_ylim(list_df[fred_20]['Close'].min(),avg_initialclaims+2*std_initialclaims)
axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[2].plot(list_df[fred_21]['Close'][start:])
axes[2].set_title('Continuing Claims')
axes[2].sharex(axes[0])
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[3].plot(list_df[fred_22]['Close'][start:])
axes[3].set_title('Average Weeks Unemployed')
axes[3].sharex(axes[0])
axes[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[4].plot(list_df[fred_23]['Close'][start:]/100,label='#Number Unemployed for 27 Weeks & over ', alpha=0.5)
axes[4].set_title('Number Unemployed for 27 Weeks & over ')
axes[4].sharex(axes[0])
axes[4].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[5].plot(list_df[fred_24]['Close'][start:]/100,label='Labor Force Participation')
axes[5].set_title('Labor Force Paricipation')
axes[5].set_xlim(start,end)
axes[5].sharex(axes[0])
axes[5].yaxis.set_major_formatter(PercentFormatter(1,decimals=2))
#axes[5].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

if highlight_recession == True:
    for x in range(0,len(dates_realgdp)):
        for ax in axes:
            ax.axvline(dates_realgdp[x],color='black', alpha=0.7)

fig.suptitle("Employment Data")        
plt.tight_layout(h_pad=0.5)
pdf.savefig()
plt.show()
plt.close()

#Page 5: Employment Part 2 ####################################################################
var_count = 4
fig = plt.figure(figsize=(8.5, 11))
gs = gridspec.GridSpec(var_count, 1, height_ratios=[1] + [1] * (var_count- 1))  # First subplot is larger
axes = [fig.add_subplot(gs[i]) for i in range(var_count)]

axes[0].plot(list_df[fred_25]['Close'][start:]/100,label='Business Expectations: Employment Growth')
axes[0].set_title('Business Expectations: Employment Growth')
axes[0].yaxis.set_major_formatter(PercentFormatter(1,decimals=2))
axes[0].set_xlim(start,end)
axes[0].sharex(axes[0])
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[1].plot(list_df[fred_26]['Close'][start:],label='Chicago Fed Hiring Expectations in the next 12M')
axes[1].set_title('Chicago Fed Hiring Expectations in the next 12M')
axes[1].sharex(axes[0])
axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[2].plot(list_df[fred_27]['Close'][start:])
axes[2].set_title('Average Weekly Hours of All Employees, Total Private')
axes[2].sharex(axes[0])
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[3].plot(list_df[fred_28]['PoP Change Pct Annualized'][start:])
axes[3].set_title('Employment Cost Index: Wages and Salaries: Private Industry Workers')
axes[3].sharex(axes[0])
axes[3].yaxis.set_major_formatter(PercentFormatter(1,decimals=2))
#axes[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

if highlight_recession == True:
    for x in range(0,len(dates_realgdp)):
        for ax in axes:
            ax.axvline(dates_realgdp[x],color='black', alpha=0.7)
            
fig.suptitle("Employment Data Page 2")    
plt.tight_layout(h_pad=0.5)
pdf.savefig()
plt.show()
plt.close()

#Page 6: Consumer
var_count = 8
fig = plt.figure(figsize=(8.5, 11))
gs = gridspec.GridSpec(var_count, 1, height_ratios=[1] + [1] * (var_count- 1))  # First subplot is larger
axes = [fig.add_subplot(gs[i]) for i in range(var_count)]

axes[0].plot(list_df[fred_29]['PoP Change Pct'][start:])
axes[0].set_title('Employed full time: Median usual weekly real earnings')
axes[0].set_xlim(start,end)
axes[0].sharex(axes[0])
axes[0].yaxis.set_major_formatter(PercentFormatter(1,decimals=2))
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

dates_consumer0 = list_df[fred_29]['PoP Change Pct']
dates_consumer0 = dates_consumer0[dates_consumer0<-0.01].index
for x in range(0,len(dates_consumer0)):
    axes[0].axvline(dates_consumer0[x],color='red', alpha=0.3)

std_consumer1 = list_df[fred_30]['PoP Change Pct'].std()
avg_consumer1 = list_df[fred_30]['PoP Change Pct'].mean()
axes[1].plot(list_df[fred_30]['PoP Change Pct'][start:])
axes[1].set_title('Real Disposable Personal Income')
axes[1].sharex(axes[0])
axes[1].set_ylim(avg_consumer1-2*std_consumer1,avg_consumer1+2*std_consumer1)
axes[1].yaxis.set_major_formatter(PercentFormatter(1,decimals=2))
axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

dates_consumer1 = list_df[fred_30]['PoP Change Pct']
dates_consumer1 = dates_consumer1[dates_consumer1<-0.005].index
for x in range(0,len(dates_consumer1)):
    axes[1].axvline(dates_consumer1[x],color='red', alpha=0.3)

axes[2].plot(list_df[fred_31]['Close'][start:]/100)
axes[2].set_title('Multiple Jobholders as a Percent of Employed')
axes[2].sharex(axes[0])
axes[2].yaxis.set_major_formatter(PercentFormatter(1,decimals=2))
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

std_consumer3 = list_df[fred_32]['Close'].std()/100
avg_consumer3 = list_df[fred_32]['Close'].mean()/100
axes[3].plot(list_df[fred_32]['Close'][start:]/100)
axes[3].set_title('Percentage Change in Total Revolving Consumer Credit')
axes[3].sharex(axes[0])
axes[3].set_ylim(avg_consumer3-2*std_consumer3,avg_consumer3+2*std_consumer3)
axes[3].yaxis.set_major_formatter(PercentFormatter(1,decimals=2))
axes[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

dates_consumer3 = list_df[fred_32]['Close']/100
dates_consumer3 = dates_consumer3[dates_consumer3<0].index
for x in range(0,len(dates_consumer3)):
    axes[3].axvline(dates_consumer3[x],color='red', alpha=0.3)

axes[4].plot(list_df[fred_33]['Close'][start:]/100)
axes[4].set_title('Household Debt Service Payments as a Percent of Disposable Personal Income')
axes[4].sharex(axes[0])
axes[4].yaxis.set_major_formatter(PercentFormatter(1,decimals=2))
axes[4].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[5].plot(list_df[fred_34]['Close'][start:]/100)
axes[5].set_title('Delinquency Rate on Credit Card Loans')
axes[5].sharex(axes[0])
axes[5].yaxis.set_major_formatter(PercentFormatter(1,decimals=2))
axes[5].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

dates_consumer5 = list_df[fred_34]['Close']/100
dates_consumer5 = dates_consumer5[(dates_consumer5 - dates_consumer5.rolling(window=4).min())>0.005].index
for x in range(0,len(dates_consumer5)):
    axes[5].axvline(dates_consumer5[x],color='red', alpha=0.3)

std_consumer6 = list_df[fred_35]['Close'].std()/100
avg_consumer6 = list_df[fred_35]['Close'].mean()/100
axes[6].plot(list_df[fred_35]['Close'][start:]/100)
axes[6].set_title('Personal Savings Rate')
axes[6].sharex(axes[0])
axes[6].set_ylim(avg_consumer6-2*std_consumer6,avg_consumer6+2*std_consumer6)
axes[6].yaxis.set_major_formatter(PercentFormatter(1,decimals=2))
axes[6].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

dates_consumer6 = list_df[fred_35]['Close']/100
dates_consumer6 = dates_consumer6[dates_consumer6<0.05].index
for x in range(0,len(dates_consumer6)):
    axes[6].axvline(dates_consumer6[x],color='red', alpha=0.3)

axes[7].plot(list_df[fred_36]['PoP Change Pct'][start:])
axes[7].set_title('Cash-Shiller US National Hope Price Index')
axes[7].sharex(axes[0])
axes[7].yaxis.set_major_formatter(PercentFormatter(1,decimals=2))
#axes[7].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

dates_consumer7 = list_df[fred_36]['PoP Change Pct']
dates_consumer7 = dates_consumer7[dates_consumer7<0].index
for x in range(0,len(dates_consumer7)):
    axes[7].axvline(dates_consumer7[x],color='red', alpha=0.3)

if highlight_recession == True:
    for x in range(0,len(dates_realgdp)):
        for ax in axes:
            ax.axvline(dates_realgdp[x],color='black', alpha=0.7)

fig.suptitle("Consumer Data")             
plt.tight_layout(h_pad=0.5)
pdf.savefig()
plt.show()
plt.close()

#Page 7: Private Sector
var_count = 8
fig = plt.figure(figsize=(8.5, 11))
gs = gridspec.GridSpec(var_count, 1, height_ratios=[1] + [1] * (var_count- 1))  # First subplot is larger
axes = [fig.add_subplot(gs[i]) for i in range(var_count)]

axes[0].plot(list_df[fred_37]['PoP Change Pct Annualized'][start:])
axes[0].set_title('National income Corporate profits before tax')
axes[0].set_xlim(start,end)
axes[0].sharex(axes[0])
axes[0].yaxis.set_major_formatter(PercentFormatter(1,decimals=2))
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[1].plot(list_df[fred_38]['Close'][start:]/100)
axes[1].set_title('Business Expectations: Sales Revenue Growth')
axes[1].set_xlim(start,end)
axes[1].sharex(axes[0])
axes[1].yaxis.set_major_formatter(PercentFormatter(1,decimals=2))
axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[2].plot(list_df[fred_39]['Close'][start:])
axes[2].set_title('Chicago Fed: Capital Spending Expectations')
axes[2].sharex(axes[0])
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[3].plot(list_df[fred_40]['Close'][start:])
axes[3].set_title('Nowcast for Real Gross Private Domestic Investment: Fixed Investment')
axes[3].sharex(axes[0])
axes[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[4].plot(list_df[fred_41]['Close'][start:])
axes[4].set_title('Nonfarm Business Sector: Labor Productivity')
axes[4].sharex(axes[0])
axes[4].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[5].plot(list_df[fred_42]['Close'][start:])
axes[5].set_title('Total Business: Inventories to Sales Ratio')
axes[5].sharex(axes[0])
axes[5].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

std_private5 = list_df[fred_43]['PoP Change Pct Annualized'].std()/100
avg_private5 = list_df[fred_43]['PoP Change Pct Annualized'].mean()/100
axes[6].plot(list_df[fred_43]['PoP Change Pct Annualized'][start:]/100)
axes[6].set_title('Commercial and Industrial Loans, All Commercial Banks')
axes[6].yaxis.set_major_formatter(PercentFormatter(1,decimals=2))
axes[6].set_ylim(avg_private5-2*std_private5,avg_private5+2*std_private5)
axes[6].sharex(axes[0])
axes[6].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[7].plot(list_df[fred_44]['Close'][start:]/100)
axes[7].set_title('Delinquency Rate on Business Loans')
axes[7].yaxis.set_major_formatter(PercentFormatter(1,decimals=2))
axes[7].sharex(axes[0])
#axes[7].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

if highlight_recession == True:
    for x in range(0,len(dates_realgdp)):
        for ax in axes:
            ax.axvline(dates_realgdp[x],color='black', alpha=0.7)

fig.suptitle("Private Sector Data")   
plt.tight_layout(h_pad=0.5)
pdf.savefig()
plt.show()
plt.close()

#Page 8: Government. Spending. Debt.
#fig.suptitle("Government Data")   

pdf.close()
