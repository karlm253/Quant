# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 20:06:25 2024

Definitions
Delta measures change price for a change in the underlying.
Gamma measures delta change for a change in delta.
Charm measures delta change for a given change in time.
Vanna measures delta change for a given change in implied volatility.

Objective: To measure the market Delta, Gamma, Charm, Vanna given open interest.

Steps
1. Calculate Deltas for Each Strike
2. Net the Deltas of Open Interest Calls and Open Interest Puts
3. Aggregate Net Deltas across strikes and maturities
4. Perform sensitivity with changes in Deltas from a Change in Price
5. Perform sensitivty with changes in Deltas from a Change in Time to Expiry


V1: Added Delta, Gamma, Charm, Vanna.
V2: Cleaned 
V3: Add Change in Net Value with a Put and Call Split
V4: Added y-lim ranges to option open interest charts
V5: Fixed Issue with Options Start Date filtering out current day
"""

import numpy as np
from scipy.stats import norm
import pandas as pd
from pandas_datareader import data as pdr
import datetime as dt
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.optimize import minimize
import yfinance as yf
yf.pdr_override()

def blackscholes_calc (r,S,K,T,IV,option):
    d1 = (np.log(S/K) + (r + IV**2/2)*T)/(IV*np.sqrt(T))
    d2 = d1 - IV*np.sqrt(T)
    if option == "call":
        price = S*norm.cdf(d1, 0, 1) - K*np.exp(-r*T)*norm.cdf(d2, 0, 1)
    elif option == "put":
        price = K*np.exp(-r*T)*norm.cdf(-d2, 0, 1) - S*norm.cdf(-d1, 0, 1)
    return price
def delta_calc (r,S,K,T,IV, option):
    d1 = (np.log(S/K) + (r + IV**2/2)*T)/(IV*np.sqrt(T))
    if option == "call":
        delta = norm.cdf(d1, 0, 1)
    elif option == "put":
        delta = -norm.cdf(-d1, 0, 1)
    return delta
def gamma_calc (r, S, K, T, IV, option):
    d1 = (np.log(S/K) + (r + IV**2/2)*T)/(IV*np.sqrt(T))
    d2 = d1 - IV*np.sqrt(T)
    gamma_calc = norm.pdf(d1, 0, 1)/(S*IV*np.sqrt(T))
    return gamma_calc
def vega_calc(r, S, K, T, IV, option):
    "Calculate BS price of call/put"
    d1 = (np.log(S/K) + (r + IV**2/2)*T)/(IV*np.sqrt(T))
    d2 = d1 - IV*np.sqrt(T)
    vega = S*norm.pdf(d1, 0, 1)*np.sqrt(T)
    return vega*0.01
def theta_calc(r, S, K, T, IV, option):
    "Calculate BS price of call/put"
    d1 = (np.log(S/K) + (r + IV**2/2)*T)/(IV*np.sqrt(T))
    d2 = d1 - IV*np.sqrt(T)
    if option == "c":
        theta_calc = -S*norm.pdf(d1, 0, 1)*IV/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2, 0, 1)
    elif option == "p":
        theta_calc = -S*norm.pdf(d1, 0, 1)*IV/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2, 0, 1)
    return theta_calc/365

### Define variables ###################################################
r = 0.0525      #Risk Free Rate
stocks = ['META']
start = dt.datetime(2023,1,1)   #how far back you want to see price
end = dt.datetime.now()         #default is today
price_interval = '1d'          #intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
options_start = dt.datetime.now()  
options_end = dt.datetime(2027,8,20)
sensitivity_price_interval = 1
sensitivity_price = 20
sensitivity_days = 20
sensitivity_days_interval = 1
sensitivity_IV = 50
sensitivity_IV_interval = 5
stock_bin_size = None
option_bin_size = 1
stock_segment_period = 'Y'        #Q=Quarterly, Y = Yearly
options_segment_period ='Y'       #Q=Quarterly, Y = Yearly

pdf = PdfPages(stocks[0]+" Options Structure Report as of "+dt.datetime.now().strftime("%Y-%m-%d")+'.pdf')

#Get Common Share Price #################################################
df = pdr.get_data_yahoo(stocks[0],start,end, interval=price_interval)    
price_today = round(df['Close'][-1],2)
sensitivity_high = int(price_today+sensitivity_price)
sensitivity_low =int(max(price_today-sensitivity_price,0))
if stock_bin_size == None:
    stock_bin_size = round(round(((df['Close'].max() - df['Close'].min()))/20,0)/5,0)*5

#Get Options Prices & Generate Graphs ##################################
tk = yf.Ticker(stocks[0])
# Get Expiration dates
exps = pd.Series(tk.options)
exps = exps[(exps>=str(options_start-timedelta(days=1))) & (exps<=str(options_end))] #Filters Options Expiries for selected dates

# Get options for each expiration
list_options = []

for e in exps:
    options = tk.option_chain(e)
    
    df_call = options[0]
    df_call['PutCall'] = 'call'
    df_call['Expiry'] = e
    df_call['Expiry'] = pd.to_datetime(df_call['Expiry'], format='%Y-%m-%d')
    df_call['T'] = (df_call['Expiry'] - end)/365    
    
    df_call = df_call[(df_call['ask']/df_call['strike'])>0.001] #10bps, filters out stale strikes
    df_call = df_call[(df_call['impliedVolatility'])>0.05] #10bps, filters out stale strikes
    df_call = df_call[(df_call['impliedVolatility'])>0.05] #10bps, filters out stale strikes
    df_call['Breakeven'] = df_call['strike'] + df_call['lastPrice']
    df_call['Weight OI'] = df_call['openInterest']/df_call['openInterest'].sum()
    df_call['Weight Vol'] = df_call['volume']/df_call['volume'].sum()
    wavg_price_oi_call = (df_call['Breakeven']*df_call['Weight OI']).sum()
    wavg_price_vol_call = (df_call['Breakeven']*df_call['Weight Vol']).sum()
    oi_call = df_call['openInterest'].sum()
    vol_call = df_call['volume'].sum()
    
    df_put = options[1]
    df_put['PutCall'] = 'put'
    df_put['Expiry'] = e
    df_put = df_put[(df_put['ask']/df_put['strike'])>0.001] #10bps, filters out stale strikes
    df_put = df_put[(df_put['impliedVolatility'])>0.05] #10bps, filters out stale strikes
    df_put = df_put[(df_put['strike']/df['Adj Close'][-1])<5] #filters stalke strikes from stock splits
    df_put['Breakeven'] = df_put['strike'] - df_put['lastPrice']
    df_put['Weight OI'] = df_put['openInterest']/df_put['openInterest'].sum()
    df_put['Weight Vol'] = df_put['volume']/df_put['volume'].sum()
    wavg_price_oi_put = (df_put['Breakeven']*df_put['Weight OI']).sum()
    wavg_price_vol_put = (df_put['Breakeven']*df_put['Weight Vol']).sum()
    oi_put = df_put['openInterest'].sum()
    vol_put = df_put['volume'].sum()
    
    if df_call.empty | df_put.empty:
        continue #restarts the loops when the df is empty
        
    list_options.append(df_call) 
    list_options.append(df_put) 

df_options=pd.concat(list_options)
df_options['Expiry'] = pd.to_datetime(df_options['Expiry'], format='%Y-%m-%d')
df_options['Days'] = (df_options['Expiry'] - end).dt.days
df_options['T'] = df_options['Days']/365                   

###Price Charts########
df_bins = df[['Adj Close','Volume']].reset_index()
df_bins['Adj Close'] = (df_bins['Adj Close']/stock_bin_size).round(decimals=0)*stock_bin_size
df_bins['Date'] =  pd.PeriodIndex(df_bins['Date'], freq='Y')
df_volume_at_price = df_bins.groupby(['Adj Close','Date']).sum()
df_volume_at_price.unstack().plot.barh(stacked=True,figsize=(8.5,11), title=stocks[0] +' Volume at Price '+start.strftime("%Y-%m-%d")+" to "+end.strftime("%Y-%m-%d"))
plt.tight_layout()
pdf.savefig()
plt.show()

df_options_pc = df_options[['strike','openInterest','PutCall']]
df_options_pc['strike'] = (df_options_pc['strike']/option_bin_size).round(decimals=0)*option_bin_size
df_options_pc_expiry = df_options_pc[(df_options_pc['strike']>(price_today-25*option_bin_size)) & (df_options_pc['strike']<(price_today+25*option_bin_size)) ] #Optional but adjustes the y-axis range for the plot.barh
df_options_pc_expiry = df_options_pc_expiry.groupby(['strike','PutCall']).sum()
df_options_pc_expiry.unstack().plot.barh(stacked=True,figsize=(8.5,11),title=stocks[0] +' Open Interest at Strike as of '+end.strftime("%Y-%m-%d"))
plt.tight_layout()
pdf.savefig()
plt.show()

df_options_strike = df_options[['strike','openInterest','Expiry']]
df_options_strike['strike'] = (df_options_strike['strike']/option_bin_size).round(decimals=0)*option_bin_size
df_options_strike['Expiry'] =  pd.PeriodIndex(df_options_strike['Expiry'], freq=options_segment_period)
df_options_strike_expiry = df_options_strike[(df_options_strike['strike']>(price_today-25*option_bin_size)) & (df_options_strike['strike']<(price_today+25*option_bin_size)) ] #Optional but adjustes the y-axis range for the plot.barh
df_options_strike_expiry = df_options_strike_expiry.groupby(['strike','Expiry']).sum()
df_options_strike_expiry.unstack().plot.barh(stacked=True,figsize=(8.5,11),title=stocks[0] +' Open Interest at Strike as of '+end.strftime("%Y-%m-%d"))
plt.tight_layout()
pdf.savefig() 
plt.show()


### Change in Price - P&L, Value, Delta, Gamma #####################################################

#Calculate Options Value Across All Maturities
df_options['Value'] = np.where(df_options['PutCall'] == 'put', blackscholes_calc(r,price_today,df_options['strike'],df_options['T'],df_options['impliedVolatility'],'put'), blackscholes_calc(r,price_today,df_options['strike'],df_options['T'],df_options['impliedVolatility'],'call'))
df_options['Total Value'] = df_options['Value'] * df_options['openInterest'] 
total_value = df_options['Total Value'].sum()

list_options_mm_value = []
list_options_mm_value.append([price_today,total_value])
for price in range(sensitivity_low,sensitivity_high,sensitivity_price_interval):
    df_options['Value'] = np.where(df_options['PutCall'] == 'put', blackscholes_calc (r,price,df_options['strike'],df_options['T'],df_options['impliedVolatility'],'put'), blackscholes_calc (r,price,df_options['strike'],df_options['T'],df_options['impliedVolatility'],'call'))
    df_options['Total Value'] = df_options['Value'] * df_options['openInterest'] 
    total_value = df_options['Total Value'].sum()
    list_options_mm_value.append([price,total_value])
    
df_options_mm_value = pd.DataFrame(list_options_mm_value, columns = ['Price', 'Total Value'])
df_options_mm_value = df_options_mm_value.set_index('Price')
df_options_mm_value['Change Total Value'] = df_options_mm_value['Total Value'] - df_options_mm_value['Total Value'].iloc[0] #Change in profit vs current price
df_options_mm_value = df_options_mm_value.sort_index()

df_options_mm_value['Total Value'].plot.bar(figsize=(11,8.5),title=stocks[0] +' Open Interest Net Value of All Options vs Underlying Price '+end.strftime("%Y-%m-%d"))
plt.tight_layout()
pdf.savefig() 
plt.show()

#Calculate Delta
df_options['Delta'] = np.where(df_options['PutCall'] == 'put', delta_calc(r,price_today,df_options['strike'],df_options['T'],df_options['impliedVolatility'],'put'), delta_calc(r,price_today,df_options['strike'],df_options['T'],df_options['impliedVolatility'],'call'))
df_options['Total Delta'] = df_options['Delta'] * df_options['openInterest']
total_delta = df_options['Total Delta'].sum()

list_options_mm_delta = []
list_options_mm_delta.append([price_today,total_delta])
for price in range(sensitivity_low,sensitivity_high,sensitivity_price_interval):
    df_options['Delta'] = np.where(df_options['PutCall'] == 'put', delta_calc(r,price,df_options['strike'],df_options['T'],df_options['impliedVolatility'],'put'), delta_calc(r,price,df_options['strike'],df_options['T'],df_options['impliedVolatility'],'call'))
    df_options['Total Delta'] = df_options['Delta'] * df_options['openInterest']
    total_delta = df_options['Total Delta'].sum()
    list_options_mm_delta.append([price,total_delta])
    
df_options_mm_delta = pd.DataFrame(list_options_mm_delta, columns = ['Price', 'Total Delta'])
df_options_mm_delta = df_options_mm_delta.set_index('Price')
df_options_mm_delta['Total Change in Net Delta'] = df_options_mm_delta['Total Delta'] - df_options_mm_delta['Total Delta'].iloc[0]
df_options_mm_delta = df_options_mm_delta.sort_index()
df_options_mm_delta['Total Delta'].plot.bar(figsize=(11,8.5),title=stocks[0] +' Open Interest Net Delta of All Options vs Underlying Price '+end.strftime("%Y-%m-%d"))
plt.tight_layout()
pdf.savefig() 
plt.show()

#Total change in Net Delta at $ vs Price Today.
df_options_mm_delta['Total Change in Net Delta'].plot.bar(figsize=(11,8.5),title=stocks[0] +' Open Interest Change in Total Net Delta of All Options vs Underlying Price '+end.strftime("%Y-%m-%d"))
pdf.savefig() 
plt.show()

#Calculate Gamma #############################################################
df_options_mm_delta['Gamma'] = df_options_mm_delta['Total Delta'] - df_options_mm_delta['Total Delta'].shift(1) #Change in Net Delta per Incremntal $ Move (not vs price today)
df_options_mm_delta['Gamma'].plot.bar(figsize=(11,8.5),title=stocks[0] +' Open Interest Gamma as of '+end.strftime("%Y-%m-%d"))
plt.tight_layout()
pdf.savefig() 
plt.show()
#df_options_mm_delta['MM Gamma'].plot.bar(figsize=(11,8.5),title=stocks[0] +' Market Maker Net Gamma  as of '+end.strftime("%Y-%m-%d"))
#plt.tight_layout()
#pdf.savefig() 
#plt.show()


### Change in Volatility: Vega, Vanna, Vomma ##################################
df_options_v = df_options

#Calculate Options Value Across All Maturities
list_options_mm_v_value = []

df_options_v['Value'] = np.where(df_options_v['PutCall'] == 'put', blackscholes_calc (r,price_today,df_options_v['strike'],df_options_v['T'],df_options_v['impliedVolatility'],'put'), blackscholes_calc (r,price_today,df_options_v['strike'],df_options_v['T'],df_options_v['impliedVolatility'],'call'))
df_options_v['Total Value'] = df_options_v['Value'] * df_options_v['openInterest'] 
total_value = df_options_v['Total Value'].sum()
list_options_mm_v_value.append([df_options['impliedVolatility'].median(),total_value])

iv_median = int(round(df_options_v['impliedVolatility'].median()*100,0))

for iv in range( max(iv_median - sensitivity_IV,sensitivity_IV_interval) , iv_median + sensitivity_IV,sensitivity_IV_interval):
    iv = iv/100
    df_options_v['Value'] = np.where(df_options_v['PutCall'] == 'put', blackscholes_calc (r,price_today,df_options_v['strike'],df_options_v['T'],iv,'put'), blackscholes_calc (r,price_today,df_options_v['strike'],df_options_v['T'],iv,'call'))
    df_options_v['Total Value'] = df_options_v['Value'] * df_options_v['openInterest'] 
    total_value = df_options_v['Total Value'].sum()
    list_options_mm_v_value.append([iv,total_value])
    
df_options_v_mm_value = pd.DataFrame(list_options_mm_v_value, columns = ['IV', 'Total Value'])
df_options_v_mm_value['IV'] = round(df_options_v_mm_value['IV'] *100,0)
df_options_v_mm_value = df_options_v_mm_value.set_index('IV')
df_options_v_mm_value['Change Total Value'] = df_options_v_mm_value['Total Value'] - df_options_v_mm_value['Total Value'].iloc[0] #Change in profit vs current price
df_options_v_mm_value = df_options_v_mm_value.sort_index()
df_options_v_mm_value['Incremental Change'] = df_options_v_mm_value['Total Value'] - df_options_v_mm_value['Total Value'].shift(1) #Change in profit vs current price

#df_options_v_mm_value['Total Value'].plot.bar(figsize=(11,8.5),title=stocks[0] +' Open Interest Net Value vs Implied Volatility as of '+end.strftime("%Y-%m-%d"))
#pdf.savefig() 
#plt.show()
#df_options_v_mm_value['Incremental Change'].plot.bar(figsize=(11,8.5),title=stocks[0] +' Open Interest Vega as of '+end.strftime("%Y-%m-%d"))
#pdf.savefig() 
#plt.show()

#Calculate Options Delta Across All Maturities
list_options_mm_v_delta = []

df_options_v['Delta'] = np.where(df_options_v['PutCall'] == 'put', delta_calc(r,price_today,df_options_v['strike'],df_options_v['T'],df_options_v['impliedVolatility'],'put'), delta_calc(r,price_today,df_options_v['strike'],df_options_v['T'],df_options_v['impliedVolatility'],'call'))
df_options_v['Total Delta'] = df_options_v['Delta'] * df_options_v['openInterest']
total_delta = df_options_v['Total Delta'].sum()
list_options_mm_v_delta.append([df_options['impliedVolatility'].median(),total_delta])

for iv in range( max(iv_median - sensitivity_IV,sensitivity_IV_interval) , iv_median + sensitivity_IV,sensitivity_IV_interval):
    iv = iv/100
    df_options_v['Delta'] = np.where(df_options_v['PutCall'] == 'put', delta_calc(r,price_today,df_options_v['strike'],df_options_v['T'],iv,'put'), delta_calc(r,price_today,df_options_v['strike'],df_options_v['T'],iv,'call'))
    df_options_v['Total Delta'] = df_options_v['Delta'] * df_options_v['openInterest']
    total_delta = df_options_v['Total Delta'].sum()
    list_options_mm_v_delta.append([iv,total_delta])
    
df_options_v_mm_delta = pd.DataFrame(list_options_mm_v_delta, columns = ['IV', 'Total Delta'])
df_options_v_mm_delta['IV'] = round(df_options_v_mm_delta['IV'] *100,0)
df_options_v_mm_delta = df_options_v_mm_delta.set_index('IV')
df_options_v_mm_delta['Change Total Delta'] = df_options_v_mm_delta['Total Delta'] - df_options_v_mm_delta['Total Delta'].iloc[0] #Change in profit vs current price
df_options_v_mm_delta = df_options_v_mm_delta.sort_index()

#df_options_v_mm_delta['Total Delta'].plot.bar(figsize=(11,8.5),title=stocks[0] +' Open Interest Net Delta given IV as of '+end.strftime("%Y-%m-%d"))
#plt.show()

df_options_v_mm_delta['Change Total Delta'].plot.bar(figsize=(11,8.5),title=stocks[0] +' Open Interest Change in Total Net Delta given IV as of '+end.strftime("%Y-%m-%d"))
plt.tight_layout()
pdf.savefig() 
plt.show()

#Calculate Vanna #############################################################
df_options_v_mm_delta['Vanna'] = df_options_v_mm_delta['Total Delta'] - df_options_v_mm_delta['Total Delta'].shift(1)
df_options_v_mm_delta['Vanna'].plot.bar(figsize=(11,8.5),title=stocks[0] +' Open Interest Vanna as of '+end.strftime("%Y-%m-%d"))
plt.tight_layout()
pdf.savefig() 
plt.show()


### Change in Time: Theta, Charm, Veta (not sure this can be calculated) ###############################################################
df_options_t = df_options

#Calculate Options Value Across All Maturities
list_options_mm_t_value = []

df_options_t['Days'] = (df_options_t['Expiry'] - end).dt.days
df_options_t['T'] = np.maximum(df_options_t['Days']/365,0)
df_options_t['Value'] = np.where(df_options_t['PutCall'] == 'put', blackscholes_calc (r,price_today,df_options_t['strike'],df_options_t['T'],df_options_t['impliedVolatility'],'put'), blackscholes_calc (r,price_today,df_options_t['strike'],df_options_t['T'],df_options_t['impliedVolatility'],'call'))
df_options_t['Total Value'] = df_options_t['Value'] * df_options_t['openInterest'] 
total_value = df_options_t['Total Value'].sum()
list_options_mm_t_value.append([0,total_value])

for day in range(sensitivity_days_interval, sensitivity_days+sensitivity_days_interval, sensitivity_days_interval):
    end_t = end + dt.timedelta(days=day)
    df_options_t['Days'] = (df_options_t['Expiry'] - end_t).dt.days
    df_options_t['T'] = np.maximum(df_options_t['Days']/365,0)
    df_options_t['Value'] = np.where(df_options_t['PutCall'] == 'put', blackscholes_calc (r,price_today,df_options_t['strike'],df_options_t['T'],df_options_t['impliedVolatility'],'put'), blackscholes_calc (r,price_today,df_options_t['strike'],df_options_t['T'],df_options_t['impliedVolatility'],'call'))
    df_options_t['Total Value'] = df_options_t['Value'] * df_options_t['openInterest'] 
    total_value = df_options_t['Total Value'].sum()
    list_options_mm_t_value.append([day,total_value])
    
df_options_t_mm_value = pd.DataFrame(list_options_mm_t_value, columns = ['Days', 'Total Value'])
df_options_t_mm_value = df_options_t_mm_value.set_index('Days')
df_options_t_mm_value['Change Total Value'] = df_options_t_mm_value['Total Value'] - df_options_t_mm_value['Total Value'].iloc[0] #Change in profit vs current price
df_options_t_mm_value = df_options_t_mm_value.sort_index()
df_options_t_mm_value['Theta'] = df_options_t_mm_value['Total Value'] - df_options_t_mm_value['Total Value'].shift(1)

#df_options_t_mm_value['Total Value'].plot.bar(figsize=(11,8.5),title=stocks[0] +' Open Interest Net Value Over Time Assuming no Rolls as of '+end.strftime("%Y-%m-%d"))
#plt.show()
#df_options_t_mm_value['Theta'].plot.bar(figsize=(11,8.5),title=stocks[0] +' Open Interest Change in Net Value Over Time Assuming no Rolls as of '+end.strftime("%Y-%m-%d"))
#pdf.savefig(stocks[0]+' Open Interest Incremental Change in Net Value Over Time - ' + end.strftime("%Y-%m-%d")) 
#plt.show()

#Calculate Options Delta Across All Maturities
list_options_mm_t_delta = []

df_options_t['Days'] = (df_options_t['Expiry'] - end).dt.days
df_options_t['T'] = np.maximum(df_options_t['Days']/365,0)
df_options_t['Delta'] = np.where(df_options_t['PutCall'] == 'put', delta_calc(r,price_today,df_options_t['strike'],df_options_t['T'],df_options_t['impliedVolatility'],'put'), delta_calc(r,price_today,df_options_t['strike'],df_options_t['T'],df_options_t['impliedVolatility'],'call'))
df_options_t['Total Delta'] = df_options_t['Delta'] * df_options_t['openInterest']
total_delta = df_options_t['Total Delta'].sum()
list_options_mm_t_delta.append([0,total_delta])

for day in range(sensitivity_days_interval, sensitivity_days+sensitivity_days_interval, sensitivity_days_interval):
    end_t = end + dt.timedelta(days=day)
    df_options_t['Days'] = (df_options_t['Expiry'] - end_t).dt.days
    df_options_t['T'] = np.maximum(df_options_t['Days']/365,0)
    df_options_t['Delta'] = np.where(df_options_t['PutCall'] == 'put', delta_calc(r,price_today,df_options_t['strike'],df_options_t['T'],df_options_t['impliedVolatility'],'put'), delta_calc(r,price_today,df_options_t['strike'],df_options_t['T'],df_options_t['impliedVolatility'],'call'))
    df_options_t['Total Delta'] = df_options_t['Delta'] * df_options_t['openInterest']
    total_delta = df_options_t['Total Delta'].sum()
    list_options_mm_t_delta.append([day,total_delta])
    
df_options_t_mm_delta = pd.DataFrame(list_options_mm_t_delta, columns = ['Days', 'Total Delta'])
df_options_t_mm_delta = df_options_t_mm_delta.set_index('Days')
df_options_t_mm_delta['Change Total Delta'] = df_options_t_mm_delta['Total Delta'] - df_options_t_mm_delta['Total Delta'].iloc[0] #Change in profit vs current price
df_options_t_mm_delta = df_options_t_mm_delta.sort_index()


#df_options_t_mm_delta['Total Delta'].plot.bar(figsize=(11,8.5),title=stocks[0] +' Open Interest Net Delta Over Time Assuming no Rolls as of '+end.strftime("%Y-%m-%d"))
#plt.show()

#Calculate Charm
df_options_t_mm_delta['Charm'] = df_options_t_mm_delta['Total Delta'] - df_options_t_mm_delta['Total Delta'].shift(1)
df_options_t_mm_delta['Charm'].plot.bar(figsize=(11,8.5),title=stocks[0] +' Open Interest Change in Total Net Delta Over Time as of '+end.strftime("%Y-%m-%d"))
plt.tight_layout()
pdf.savefig() 
plt.show()

pdf.close()
