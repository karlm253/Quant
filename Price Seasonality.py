# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 20:59:03 2023

@author: HB

V1 - Monthly Seasonality
V2 - Added Price Interval Feature to toggle 1M, 3M, 1W etc.

"""

#Compare Annual For Performance
import math
import yfinance as yf
import pandas as pd
from pandas_datareader import data as pdr
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
yf.pdr_override()


stocks = ['^NDX']
end = dt.datetime.now()
start = dt.datetime(1987,12,31)
price_interval = '1wk' #1d, 1wk, 1mo, 3mo
every_x_years = 1 #if you want to track the presidential cycle for instance
sns.set_theme(rc={'figure.figsize':(20,8)}) #Chart Settings

if price_interval == '1mo':
    look_forward = 13
if price_interval == '3mo':
    look_forward = 5
if price_interval == '1wk':
    look_forward = 53
if price_interval == '1d':
    look_forward = 253

xaxis = np.arange(1,look_forward)

df = pdr.get_data_yahoo(stocks[0],start,end, interval=price_interval)
df_returns = df/df.shift(1)-1

years =  df.groupby(pd.DatetimeIndex(df.index).to_period('Y')).nth(-1)
years = years.iloc[::every_x_years, :].index.tolist()

vintage_table_list = []
for i in years:
    date = df_returns.index.get_loc(i)
    series = df_returns.iloc[date: date + look_forward, 4].rename(i).reset_index(drop=True)
    vintage_table_list.append(series)
df_vintage_period_chg = pd.concat(vintage_table_list, axis = 1)
df_vintage_period_chg.columns = map(str, df_vintage_period_chg.columns)
df_vintage_period_chg = df_vintage_period_chg.set_axis(range(0, len(df_vintage_period_chg)))
df_vintage = df_vintage_period_chg.iloc[0:-1]

df_vintage_columns = len(df_vintage.columns)
df_vintage['average'] = df_vintage.iloc[:].mean(axis=1)
df_vintage['max'] = df_vintage.iloc[:,:df_vintage_columns].max(axis=1)
df_vintage['min'] = df_vintage.iloc[:,:df_vintage_columns].min(axis=1)
df_vintage['median'] = df_vintage.iloc[:,:df_vintage_columns].median(axis=1)
df_vintage['sd'] = df_vintage.iloc[:,:df_vintage_columns].std(axis=1)

#Line Chart
#sns.lineplot(data=df_vintage_period_chg)
#plt.xticks(xaxis)
#plt.title(stocks[0] + ' PoP Return Chart')
#plt.show()

#Subplot form
#axes = df_vintage_period_chg.plot(subplots=True, layout=(df_vintage_columns,1),figsize=(12, 3*df_vintage_columns ))
#plt.xticks(xaxis)
#plt.suptitle(stocks[0] + ' Price Chart')
#plt.tight_layout()
#plt.show()

#AVG, StD MAX and MIN Returns
avg_line = sns.lineplot(x=df_vintage.index, y= 'average',data=df_vintage, linewidth=3)
plt.xticks(xaxis)
plt.fill_between(df_vintage.index,df_vintage['min'] , df_vintage['max'], alpha=0.25)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.title(stocks[0] + ' Average, Max and Min Returns')
plt.show()

avg_line = sns.lineplot(x=df_vintage.index, y= 'average',data=df_vintage, linewidth=3)
plt.xticks(xaxis)
plt.fill_between(df_vintage.index,df_vintage['average'] + df_vintage['sd'], df_vintage['average'] - df_vintage['sd'], alpha=0.25)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.title(stocks[0] + ' Average, STD Returns')
plt.show()

#Boxplot
df_boxplot = df_vintage
df_boxplot.index = np.arange(1, len(df_boxplot) + 1)
df_boxplot = df_boxplot.transpose()

period_avg = df_boxplot.iloc[-5,:].reset_index(drop=True)
period_std = df_boxplot.iloc[-1,:].reset_index(drop=True)
period_median = df_boxplot.iloc[-2,:].reset_index(drop=True)

boxplot = sns.boxplot(df_boxplot,showmeans=True)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
for x in boxplot.get_xticks():
    boxplot.text(x,period_median[x],f'{period_avg[x]:.1%}',horizontalalignment='center',size='large',color='black',weight='bold')
plt.suptitle(stocks[0]+' '+price_interval+' Return and Median')
plt.show()

#Swarm Plot
swarmplot = sns.swarmplot(df_boxplot)
plt.suptitle(stocks[0]+' '+price_interval+' Return')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
for x in swarmplot.get_xticks():
    swarmplot.text(x,period_median[x],f'{period_avg[x]:.1%}',horizontalalignment='center',size='large',color='black',weight='bold',backgroundcolor='white')
plt.suptitle(stocks[0]+' '+price_interval+' Pop Returns% and Median')
plt.show()
plt.show()

#Histogram Returns Median
layout_x = int(math.floor(look_forward**0.5))
layout_y = int(round(look_forward**0.5,0)+1)
df_hist = df_boxplot.iloc[:-6,:]
axes_2 = df_hist.plot.hist(subplots=True, bins=50,legend=True, layout=(layout_y,layout_x),color = "tab:blue", figsize=(layout_x*4,layout_y*3))
for ax, x in zip(axes_2.flatten(), period_median):
    if x > 0:
        ax.axvline(x,color='green')
        ax.text(x-.01, 0.5, f'{x:.1%}', weight='bold',fontsize=14,color='green', backgroundcolor='white',horizontalalignment='center')
    else:
        ax.axvline(x,color='red')
        ax.text(x-.01, 0.5, f'{x:.1%}', weight='bold',fontsize=14,color='red', backgroundcolor='white',horizontalalignment='center')
for ax, x in zip(axes_2.flatten(), df_hist.iloc[-1].transpose()):
    try:
        ax.axvline(x,color='black')
        ax.text(x-.01, 2, f'{x:.1%}', weight='bold',fontsize=14,color='black', backgroundcolor='white',horizontalalignment='center')
    except:
        pass
plt.gca().xaxis.set_major_formatter(PercentFormatter(1))
plt.suptitle(stocks[0]+' Histogram PoP Return. Median Return and '+ str(end.year)+' Year Return')
plt.tight_layout()
plt.show()

#Histogram Returns Average
layout_x = int(math.floor(look_forward**0.5))
layout_y = int(round(look_forward**0.5,0)+1)
df_hist = df_boxplot.iloc[:-6,:]
axes_2 = df_hist.plot.hist(subplots=True, bins=50,legend=True, layout=(layout_y,layout_x),color = "tab:blue", figsize=(layout_x*4,layout_y*3))
for ax, x in zip(axes_2.flatten(), period_avg):
    if x > 0:
        ax.axvline(x,color='green')
        ax.text(x-.01, 0.5, f'{x:.1%}', weight='bold',fontsize=14,color='green', backgroundcolor='white',horizontalalignment='center')
    else:
        ax.axvline(x,color='red')
        ax.text(x-.01, 0.5, f'{x:.1%}', weight='bold',fontsize=14,color='red', backgroundcolor='white',horizontalalignment='center')
for ax, x in zip(axes_2.flatten(), df_hist.iloc[-1].transpose()):
    try:
        ax.axvline(x,color='black')
        ax.text(x-.01, 1.5, f'{x:.1%}', weight='bold',fontsize=14,color='black', backgroundcolor='white',horizontalalignment='center')
    except:
        pass
plt.gca().xaxis.set_major_formatter(PercentFormatter(1))
plt.suptitle(stocks[0]+' Histogram PoP Return. Average Return and '+ str(end.year)+' Year Return')
plt.tight_layout()
plt.show()
