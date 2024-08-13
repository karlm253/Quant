# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 10:48:12 2024

@author: CX

This is a simple tool to visualize options by open interest.
The flaw in this tool is that you cannot tell which direction the bet is with open interest. 
i.e. It could be long puts or short puts such as the bottom of a spread trade.

V1 Added Highest Open Interest Dots & WAVG OI Price
V2 Added Highest Implied Volatility Dots4 & WVG OI Price
V3 Added Max Pain
V4 Added Day's Volume 
V5 Added a continue for when the options expiry df is empty
V6 Added Put Call Partiy Implied Stock Price
V7 Added open interest by strike across all maturities -> Gamma % Short Squeezes)
V8 Added Highest Implied VOl to Lowest Implied Vol (most probable)
V9 Added Volume at Price and OI at Breakeven

"""

import pandas as pd
from pandas_datareader import data as pdr
import datetime as dt
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.optimize import minimize
import yfinance as yf
yf.pdr_override()

def intensify_cmap(cmap, factor=1.5):
    colors = cmap(np.arange(cmap.N))
    hsv_colors = plt.cm.colors.rgb_to_hsv(colors[:, :3])
    hsv_colors[:, 1] = np.clip(hsv_colors[:, 1] * factor, 0, 1)
    new_colors = plt.cm.colors.hsv_to_rgb(hsv_colors)
    return LinearSegmentedColormap.from_list(cmap.name + "_intense", new_colors, cmap.N)    
def price_formatter(x, pos):
    return f"${x:.2f}"
def returns_formatter(x, pos):
    return f"{x*100:.0f}%"

#Custom Variables
stock = 'META'
top = 3                         #top x open interest strike on each side
start = dt.datetime(2022,1,1)   #how far back you want to see price
end = dt.datetime.now()         #default is today
price_interval = '1d'           #intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
stock_bin_size = None           #None = Auto
option_bin_size = None          #None = Auto
stock_segment_period = 'Y'      #Q=Quarterly, Y = Yearly
options_segment_period ='Y'     #Q=Quarterly, Y = Yearly

pdf = PdfPages(stock+" Options Forecast Report as of "+dt.datetime.now().strftime("%Y-%m-%d")+'.pdf')

#Get Common Share Price
df = pdr.get_data_yahoo(stock,start,end, interval=price_interval)   
price_today = round(df['Close'][-1],2)

if stock_bin_size == None:
    stock_bin_size = round(round(((df['Close'].max() - df['Close'].min()))/20,0)/5,0)*5

#Creates Stock Volume at Price Chart
df_bins = df[['Adj Close','Volume']].reset_index()
df_bins['Adj Close'] = (df_bins['Adj Close']/stock_bin_size).round(decimals=0)*stock_bin_size
df_bins['Date'] =  pd.PeriodIndex(df_bins['Date'], freq=stock_segment_period)
df_volume_at_price = df_bins.groupby(['Adj Close','Date']).sum()
df_volume_at_price.unstack().plot.barh(stacked=True,figsize=(8.5,11), title=stock +' Volume at Price '+start.strftime("%Y-%m-%d")+" to "+end.strftime("%Y-%m-%d"))
plt.tight_layout()
pdf.savefig()
plt.show()

#Get Options Prices & Generate Graphs

list_options = []
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
    df_call = df_call[(df_call['strike']/df['Adj Close'][-1])<5] #filters stalke strikes from stock splits
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
    #Get Max Pain
   
    def max_pain_func(price):
        return (np.maximum(price - df_call.strike,0) + np.maximum(df_put.strike-price,0)).sum()

    max_pain = minimize(max_pain_func,df['Close'][-1])
    max_pain = max_pain.x
    df_max_pain = pd.DataFrame({'Expiry':df_call.Expiry.iloc[0],'Max Pain':max_pain})
    list_max_pain.append(df_max_pain)

df_wavg_oi =pd.concat(list_wavg_oi)
df_wavg_oi['Expiry'] = pd.to_datetime(df_wavg_oi['Expiry'])
df_wavg_oi.set_index('Expiry',inplace=True)

df_wavg_vol =pd.concat(list_wavg_vol)
df_wavg_vol['Expiry'] = pd.to_datetime(df_wavg_vol['Expiry'])
df_wavg_vol.set_index('Expiry',inplace=True)

df_top_oi = pd.concat(list_top_oi)
df_top_oi['Expiry'] = pd.to_datetime(df_top_oi['Expiry'])
df_top_oi.set_index('Expiry',inplace=True)

df_top_iv = pd.concat(list_top_iv)
df_top_iv['Expiry'] = pd.to_datetime(df_top_iv['Expiry'])
df_top_iv.set_index('Expiry',inplace=True)

df_bot_iv = pd.concat(list_bot_iv)
df_bot_iv['Expiry'] = pd.to_datetime(df_bot_iv['Expiry'])
df_bot_iv.set_index('Expiry',inplace=True)

df_top_volume = pd.concat(list_top_volume)
df_top_volume['Expiry'] = pd.to_datetime(df_top_volume['Expiry'])
df_top_volume.set_index('Expiry',inplace=True)

df_max_pain = pd.concat(list_max_pain)
df_max_pain['Expiry'] = pd.to_datetime(   df_max_pain['Expiry'])
df_max_pain.set_index('Expiry',inplace=True)


cmap = plt.get_cmap('Blues')


#Create Chart for Volume at Strike Options Price
df_options=pd.concat(list_options)

if option_bin_size == None:
    option_bin_size = round(round(((df_options['Breakeven'].max() - df_options['Breakeven'].min()))/40,0)/5,0)*5

df_options_pc = df_options[['strike','openInterest','PutCall']]
df_options_pc['strike'] = (df_options_pc['strike']/option_bin_size).round(decimals=0)*option_bin_size
df_options_pc_expiry = df_options_pc.groupby(['strike','PutCall']).sum()
df_options_pc_expiry.unstack().plot.barh(stacked=True,figsize=(8.5,11),title=stock +' Open Interest at Strike as of '+end.strftime("%Y-%m-%d"))
plt.savefig(stock+' Options Put Call Open Interest at Strike Price - ' + end.strftime("%Y-%m-%d")+".png") 

df_options_strike = df_options[['strike','openInterest','Expiry']]
df_options_strike['strike'] = (df_options_strike['strike']/option_bin_size).round(decimals=0)*option_bin_size
df_options_strike['Expiry'] =  pd.PeriodIndex(df_options_strike['Expiry'], freq=options_segment_period)
df_options_strike_expiry = df_options_strike.groupby(['strike','Expiry']).sum()
df_options_strike_expiry.unstack().plot.barh(stacked=True,figsize=(8.5,11),title=stock +' Open Interest at Strike as of '+end.strftime("%Y-%m-%d"))
plt.savefig(stock+' Options Open Interest at Strike Price - ' + end.strftime("%Y-%m-%d")+".png") 

df_options_strike_0 = df_options_strike[df_options_strike['Expiry']==str(end.year)].groupby(['strike','Expiry']).sum()
df_options_strike_0.unstack().plot.barh(stacked=True,figsize=(8.5,11),title=stock +' '+str(end.year)+' Open Interest at Strike as of '+end.strftime("%Y-%m-%d"))
df_options_strike_1 = df_options_strike[df_options_strike['Expiry']==str(end.year+1)].groupby(['strike','Expiry']).sum()
df_options_strike_1.unstack().plot.barh(stacked=True,figsize=(8.5,11),title=stock +' '+str(end.year+1)+' Open Interest at Strike as of '+end.strftime("%Y-%m-%d"))
df_options_strike_2 = df_options_strike[df_options_strike['Expiry']==str(end.year+2)].groupby(['strike','Expiry']).sum()
df_options_strike_2.unstack().plot.barh(stacked=True,figsize=(8.5,11),title=stock +' '+str(end.year+2)+' Open Interest at Strike as of '+end.strftime("%Y-%m-%d"))

# Plot Options Data Only
intense_cmap = intensify_cmap(cmap,factor =6)
fig, ax = plt.subplots(figsize=(11, 8.5), dpi=400)
ax1 = plt.plot(df.index[-1],df[('Close')][-1])
ax2 = plt.plot(df_wavg_oi.index,df_wavg_oi.iloc[:,0:3])
ax3 = plt.plot(df_max_pain.index,df_max_pain)
ax4 = plt.scatter(df_top_oi.index,df_top_oi['strike'],c=df_top_oi['openInterest'], cmap=intense_cmap)
legend_handles = ['Historical Price',df_wavg_oi.columns[0], df_wavg_oi.columns[1],df_wavg_oi.columns[2],df_max_pain.columns[0]]
ax.legend(legend_handles)
cbar = plt.colorbar(label='Open Interest')
num_ticks = len(plt.yticks()[0]) * 2
plt.yticks(np.linspace(plt.ylim()[0], plt.ylim()[1], num_ticks))
plt.grid(True,linewidth=0.5, color='gray', alpha=0.5)
plt.title(stock+' Highest Open Interest - '+ end.strftime("%Y-%m-%d"), fontsize=16)                               
plt.tight_layout()
pdf.savefig()
plt.show()
    
intense_cmap = intensify_cmap(cmap,factor =6)
fig, ax = plt.subplots(figsize=(11, 8.5), dpi=400)
ax1 = plt.plot(df.index[-1],df[('Close')][-1])
ax2 = plt.plot(df_wavg_vol.index,df_wavg_vol.iloc[:,0:3])
ax3 = plt.plot(df_max_pain.index,df_max_pain)
ax4 = plt.scatter(df_top_volume.index,df_top_volume['Breakeven'],c=df_top_volume['volume'], cmap=intense_cmap)
legend_handles = ['Historical Price',df_wavg_vol.columns[0], df_wavg_vol.columns[1],df_wavg_vol.columns[2],df_max_pain.columns[0]]
ax.legend(legend_handles)
cbar = plt.colorbar(label='Volume')
num_ticks = len(plt.yticks()[0]) * 2
plt.yticks(np.linspace(plt.ylim()[0], plt.ylim()[1], num_ticks))
plt.grid(True,linewidth=0.5, color='gray', alpha=0.5)
plt.title(stock+' Highest Volume - '+ end.strftime("%Y-%m-%d"), fontsize=16)                               
plt.tight_layout()
pdf.savefig()
plt.show()

intense_cmap = intensify_cmap(cmap,factor =4)
fig, ax = plt.subplots(figsize=(11, 8.5), dpi=400)
ax1 = plt.plot(df.index[-1],df[('Close')][-1])
ax2 = plt.plot(df_wavg_oi.index,df_wavg_oi.iloc[:,0:3])
ax3 = plt.plot(df_max_pain.index,df_max_pain)
ax4 = plt.scatter(df_bot_iv.index,df_bot_iv['Breakeven'],c=df_bot_iv['impliedVolatility'], cmap=intense_cmap)
legend_handles = ['Historical Price',df_wavg_oi.columns[0], df_wavg_oi.columns[1],df_wavg_oi.columns[2],df_max_pain.columns[0]]
ax.legend(legend_handles)
cbar = plt.colorbar(label='impliedVolatility')
num_ticks = len(plt.yticks()[0]) * 2
plt.yticks(np.linspace(plt.ylim()[0], plt.ylim()[1], num_ticks))
plt.grid(True,linewidth=0.5, color='gray', alpha=0.5)
plt.title(stock+' Lowest Implied Volatility - '+ end.strftime("%Y-%m-%d"), fontsize=16)                               
plt.tight_layout()
pdf.savefig()
plt.show()

intense_cmap = intensify_cmap(cmap,factor =4)
fig, ax = plt.subplots(figsize=(11, 8.5), dpi=400)
ax1 = plt.plot(df.index[-1],df[('Close')][-1])
ax2 = plt.plot(df_wavg_oi.index,df_wavg_oi.iloc[:,0:3])
ax3 = plt.plot(df_max_pain.index,df_max_pain)
ax4 = plt.scatter(df_top_iv.index,df_top_iv['Breakeven'],c=df_top_iv['impliedVolatility'], cmap=intense_cmap)
legend_handles = ['Historical Price',df_wavg_oi.columns[0], df_wavg_oi.columns[1],df_wavg_oi.columns[2],df_max_pain.columns[0]]
ax.legend(legend_handles)
cbar = plt.colorbar(label='impliedVolatility')
num_ticks = len(plt.yticks()[0]) * 2
plt.yticks(np.linspace(plt.ylim()[0], plt.ylim()[1], num_ticks))
plt.grid(True,linewidth=0.5, color='gray', alpha=0.5)
plt.title(stock+' Highest Implied Volatility - '+ end.strftime("%Y-%m-%d"), fontsize=16)                               
plt.tight_layout()
pdf.savefig()
plt.show()

# Plot Price History + Options Data
intense_cmap = intensify_cmap(cmap,factor =6)
fig, ax = plt.subplots(figsize=(11, 8.5), dpi=400)
ax1 = plt.plot(df[('Close')])
ax2 = plt.plot(df_wavg_oi.index,df_wavg_oi.iloc[:,0:3])
ax3 = plt.plot(df_max_pain.index,df_max_pain)
ax4 = plt.scatter(df_top_oi.index,df_top_oi['Breakeven'],c=df_top_oi['openInterest'], cmap=intense_cmap)
return_axis = plt.ylim()[0]/df[('Close')][-1]-1, plt.ylim()[1]/df[('Close')][-1]-1
ax4 = ax.twinx()
ax4.set_ylim(return_axis)
ax.legend(legend_handles)
num_ticks = len(plt.yticks()[0]) * 2
plt.yticks(np.linspace(plt.ylim()[0], plt.ylim()[1], num_ticks))
plt.grid(True,linewidth=0.5, color='gray', alpha=0.5)
plt.title(stock+' Highest Open Interest - ' + end.strftime("%Y-%m-%d") , fontsize=16)  
ax.yaxis.set_major_formatter(FuncFormatter(price_formatter))
ax4.yaxis.set_major_formatter(FuncFormatter(returns_formatter))
plt.tight_layout()
pdf.savefig()
plt.show()

intense_cmap = intensify_cmap(cmap,factor =6)
fig, ax = plt.subplots(figsize=(11, 8.5), dpi=400)
ax1 = plt.plot(df[('Close')])
ax2 = plt.plot(df_wavg_vol.index,df_wavg_vol.iloc[:,0:3])
ax3 = plt.plot(df_max_pain.index,df_max_pain)
ax4 = plt.scatter(df_top_volume.index,df_top_volume['Breakeven'],c=df_top_volume['volume'], cmap=intense_cmap)
return_axis = plt.ylim()[0]/df[('Close')][-1]-1, plt.ylim()[1]/df[('Close')][-1]-1
ax4 = ax.twinx()
ax4.set_ylim(return_axis)
ax.legend(legend_handles)
num_ticks = len(plt.yticks()[0]) * 2
plt.yticks(np.linspace(plt.ylim()[0], plt.ylim()[1], num_ticks))
plt.grid(True,linewidth=0.5, color='gray', alpha=0.5)
plt.title(stock+' Highest Volume - ' + end.strftime("%Y-%m-%d") , fontsize=16)  
ax.yaxis.set_major_formatter(FuncFormatter(price_formatter))
ax4.yaxis.set_major_formatter(FuncFormatter(returns_formatter))
plt.tight_layout()
pdf.savefig()
plt.show()

intense_cmap = intensify_cmap(cmap,factor =4)
fig, ax = plt.subplots(figsize=(11, 8.5), dpi=400)
ax1 = plt.plot(df[('Close')])
ax2 = plt.plot(df_wavg_oi.index,df_wavg_oi.iloc[:,0:3])
ax4 = plt.plot(df_max_pain.index,df_max_pain)
ax4 = plt.scatter(df_bot_iv.index,df_bot_iv['Breakeven'],c=df_bot_iv['impliedVolatility'], cmap=intense_cmap)
return_axis = plt.ylim()[0]/df[('Close')][-1]-1, plt.ylim()[1]/df[('Close')][-1]-1
ax4 = ax.twinx()
ax4.set_ylim(return_axis)
ax.legend(legend_handles)
num_ticks = len(plt.yticks()[0]) * 2
plt.yticks(np.linspace(plt.ylim()[0], plt.ylim()[1], num_ticks))
plt.grid(True,linewidth=0.5, color='gray', alpha=0.5)
plt.title(stock+' Lowest Implied Volaility - ' + end.strftime("%Y-%m-%d") , fontsize=16)  
ax.yaxis.set_major_formatter(FuncFormatter(price_formatter))
ax4.yaxis.set_major_formatter(FuncFormatter(returns_formatter))
plt.tight_layout()
pdf.savefig()
plt.show()

intense_cmap = intensify_cmap(cmap,factor =4)
fig, ax = plt.subplots(figsize=(11, 8.5), dpi=400)
ax1 = plt.plot(df[('Close')])
ax2 = plt.plot(df_wavg_oi.index,df_wavg_oi.iloc[:,0:3])
ax4 = plt.plot(df_max_pain.index,df_max_pain)
ax4 = plt.scatter(df_top_iv.index,df_top_iv['Breakeven'],c=df_top_iv['impliedVolatility'], cmap=intense_cmap)
return_axis = plt.ylim()[0]/df[('Close')][-1]-1, plt.ylim()[1]/df[('Close')][-1]-1
ax4 = ax.twinx()
ax4.set_ylim(return_axis)
ax.legend(legend_handles)
num_ticks = len(plt.yticks()[0]) * 2
plt.yticks(np.linspace(plt.ylim()[0], plt.ylim()[1], num_ticks))
plt.grid(True,linewidth=0.5, color='gray', alpha=0.5)
plt.title(stock+' Highest Implied Volaility - ' + end.strftime("%Y-%m-%d") , fontsize=16)  
ax.yaxis.set_major_formatter(FuncFormatter(price_formatter))
ax4.yaxis.set_major_formatter(FuncFormatter(returns_formatter))
plt.tight_layout()
pdf.savefig()
plt.show()

pdf.close()