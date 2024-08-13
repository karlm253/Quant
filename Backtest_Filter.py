# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:56:48 2024

@author: Heavens Base

V1: Boxplot, Lineplot, Histogram for Price Change Filter.
V2: Added backtest variable seperate from stock returns. You can now see stock return vs backtest   filter.
V3: Added Line Chart
V4: Added fixed level function drawdown level
V5: Added Fred Data for Backtest. Line105 Issue with index dates on FRED vs Stock Dates. Will need to merge and backfill probably or try to go forward from the fred date on stock df.
V6: Title changes with date function.
V7: Added Vix Date for Vix Level and Rate of Change
V8: Added 2nd Step Change Percentage
V9: Added 2nd Backtest Filter
"""

import yfinance as yf
import pandas as pd
from pandas_datareader import data as pdr
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.backends.backend_pdf import PdfPages
from Fred_API import fred_get_series

yf.pdr_override()

#Input Variables
end = dt.datetime.now()
start = dt.datetime(1985,1,1)
stock = 'SOXX'
backtest_stock = stock #'^VIX' #use backtest=stock if you just want to test vs own historical price #^VIX, ^VXN, ^TNX, ^MOVE, ^SKEW
backtest_fred = None #'UNRATE' #UNRATE, CCSM
threshold_high = -0.20
threshold_low = -0.30
rate_of_change = 0.30
period_chg=20
look_forward=20
look_back=5

sns.set_theme(rc={'figure.figsize':(11.5,8)})
x_axis_multiple = max(round((look_back+look_forward)/20),1)

if backtest_stock != None:
    backtest = backtest_stock
elif backtest_fred != None:
    backtest = backtest_fred


#Get the data for the stock data.
if stock != None:
    df_stock = pdr.get_data_yahoo(stock,start,end)
    df_stock = df_stock.reindex(pd.date_range(start, end, freq='D')).dropna(axis=0, how = 'any')
    df_returns = df_stock/df_stock.shift(1)-1

if backtest_stock != None:
    #Get the data for the backtest variable from Yfinance
    df_backtest = pdr.get_data_yahoo(backtest,start,end)
    start = df_backtest.index[0]
    df_backtest = df_backtest.reindex(pd.date_range(start, end, freq='D')).dropna(axis=0, how = 'any')

if backtest_fred != None:
    #Get the data for the backtest variable from FRED
    df_backtest = fred_get_series(backtest)
    df_backtest.index = df_backtest['date']
    df_backtest.index = pd.to_datetime(df_backtest.index)
    df_backtest['Close'] = pd.to_numeric(df_backtest['value'],errors='coerce')
    df_backtest = df_backtest.reindex(pd.date_range(start, end, freq='D')).dropna(axis=0, how = 'any')

# Finds all the dates where the stock breached the threshold
def date_drawdown_pct(df,threshold_high,threshold_low,period_chg=1):
    df['daily_pct_change'] = df.Close.pct_change(periods=1)
    df['period_pct_chg'] = df.Close.pct_change(periods=period_chg)
    threshold_list = df[(df['period_pct_chg']<threshold_high) & (df['period_pct_chg']>threshold_low)]
        
    list_dates = threshold_list.index.tolist()
    clean_dates = []
    clean_dates.append(list_dates[0])

    for i in range(0, len(list_dates)):
        if list_dates[i] > clean_dates[-1] + dt.timedelta(days=period_chg):
            clean_dates.append(list_dates[i])
            
    title = ' - After '+f'{threshold_high:+.1%} Change in ' +backtest+' Over '+str(period_chg)+ ' Days'
    return clean_dates,title
def date_drawdown_level(df,threshold_high,threshold_low,period_chg=1):
    threshold_list = df[(df['Close']<threshold_high) & (df['Close']>threshold_low)]
    list_dates = threshold_list.index.tolist()
    clean_dates = []
    clean_dates.append(list_dates[0])

    for i in range(0, len(list_dates)):
        if list_dates[i] > clean_dates[-1] + dt.timedelta(days=period_chg):
            clean_dates.append(list_dates[i])
            
    title = backtest +' - From '+f'{threshold_low:+.1f}'+' to ' +f'{threshold_high:+.1f}' 
    return clean_dates,title
def date_level_roc(df,threshold_low,rate_of_change,period_chg=1):
    df['daily_pct_change'] = df.Close.pct_change(periods=1)
    threshold_list = df[(df['Close']>threshold_low) & (df['daily_pct_change']>rate_of_change)]
    
    list_dates = threshold_list.index.tolist()
    clean_dates = []
    clean_dates.append(list_dates[0])

    for i in range(0, len(list_dates)):
        if list_dates[i] > clean_dates[-1] + dt.timedelta(days=period_chg):
            clean_dates.append(list_dates[i])
            
    title = backtest+' - ' + f'{rate_of_change:+.1%} Change From '+f'{threshold_low:.1f}'+' to ' +f'{threshold_high:.1f}'
    return clean_dates,title
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
    title = ' -  After First Rate Cut'
    return list_dates, title

#Get Dates
dates,title = date_drawdown_pct(df_backtest,threshold_high=threshold_high,threshold_low=threshold_low,period_chg=period_chg)
#dates,title = date_drawdown_level(df_backtest,threshold_high=threshold_high,threshold_low=threshold_low,period_chg=period_chg)
#dates,title = date_first_rate_cut()
#dates,title = date_level_roc(df_backtest,threshold_low,rate_of_change)
for x in range(0,len(dates)):
    dates[x] = df_stock.index[df_stock.index.searchsorted(dates[x])] #converts Fred Dates to Nearest Stock market dates
#Append Timeseries Since Date
vintage_table_list = []
date = df_stock.index.get_loc(dates[0])
series = df_stock.iloc[date-period_chg: date + look_forward, 4].rename(dates[0]).reset_index(drop=True)

for i in dates:
    date = df_stock.index.get_loc(i)
    series = df_stock.iloc[date-look_back-1: date + look_forward+1, 4].rename(i).reset_index(drop=True)
    vintage_table_list.append(series)
df_price = pd.concat(vintage_table_list, axis = 1)
df_price.columns = map(str, df_price.columns)
df_price = df_price.set_axis(range(-look_back-1, len(df_price)-look_back-1))
df_columns = len(df_price.columns)
                 
df_return = df_price/df_price.shift(1)-1
df_return = df_return.loc[-look_back:look_forward]
df_return['average'] = df_return.iloc[:].mean(axis=1)
df_return['max'] = df_return.iloc[:,:df_columns].max(axis=1)
df_return['min'] = df_return.iloc[:,:df_columns].min(axis=1)
df_return['median'] = df_return.iloc[:,:df_columns].median(axis=1)
df_return['sd'] = df_return.iloc[:,:df_columns].std(axis=1)

df_index = df_price/df_price.loc[0]-1
df_index = df_index.loc[-look_back:look_forward]
df_index['average'] = df_index.iloc[:].mean(axis=1)
df_index['max'] = df_index.iloc[:,:df_columns].max(axis=1)
df_index['min'] = df_index.iloc[:,:df_columns].min(axis=1)
df_index['median'] = df_index.iloc[:,:df_columns].median(axis=1)
df_index['sd'] = df_index.iloc[:,:df_columns].std(axis=1)

df_return_boxplot = df_return.transpose()
return_period_avg = df_return_boxplot.iloc[-5,:].reset_index(drop=True)
return_period_std = df_return_boxplot.iloc[-1,:].reset_index(drop=True)
return_period_median = df_return_boxplot.iloc[-2,:].reset_index(drop=True)

df_index_boxplot = df_index.iloc[::x_axis_multiple].transpose()
index_period_avg = df_index_boxplot.iloc[-5,:].reset_index(drop=True)
index_period_std = df_index_boxplot.iloc[-1,:].reset_index(drop=True)
index_period_median = df_index_boxplot.iloc[-2,:].reset_index(drop=True)

#Create PDF
pdf = PdfPages(stock + title+' Backtest From ' +str(start.year)+ " to "+ str(end.year)+" as of "+dt.datetime.now().strftime("%Y-%m-%d")+'.pdf')

#Line Chart
transparency= max(1.01 - len(dates)/100,0.15)
if stock != backtest_stock:
    fig, axes = plt.subplots(nrows=2, ncols=1,sharex=True)
    axes[0].plot(df_stock['Close'],color='black')
    axes[1].plot(df_backtest['Close'],color='black')
    axes[0].set_title(stock)
    axes[1].set_title(backtest)
    for x in range(0,len(dates)):
        axes[0].axvline(dates[x],color='red', alpha=transparency)
        axes[1].axvline(dates[x],color='red', alpha=transparency)
    plt.subplots_adjust(hspace=0.15)  # Decrease the vertical spacing
    plt.suptitle(stock +' '+ title)
elif stock!= backtest_fred:
    fig, axes = plt.subplots(nrows=2, ncols=1,sharex=True)
    axes[0].plot(df_stock['Close'],color='black')
    axes[1].plot(df_backtest['Close'],color='black')
    axes[0].set_title(stock)
    axes[1].set_title(backtest)
    for x in range(0,len(dates)):
        axes[0].axvline(dates[x],color='red', alpha=transparency)
        axes[1].axvline(dates[x],color='red', alpha=transparency)
    plt.subplots_adjust(hspace=0.15)  # Decrease the vertical spacing
    plt.suptitle(stock +' '+ title)
else:
    df_stock['Close'].plot(color='black',title=stock)
    for x in range(0,len(dates)):
        plt.axvline(dates[x],color='red', alpha=transparency)
    plt.title(stock +' '+ title)

plt.tight_layout()
pdf.savefig() 
plt.show()

#BoxPlot
if (look_back+look_forward) < 31:
    boxplot = sns.boxplot(df_return_boxplot)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    for x in boxplot.get_xticks():
        if return_period_avg[x] > 0:
            boxplot.text(x,return_period_avg[x],f'{return_period_avg[x]:.1%}',horizontalalignment='center',size='medium',color='green',weight='bold',backgroundcolor='white')
        else:
            boxplot.text(x,return_period_avg[x],f'{return_period_avg[x]:.1%}',horizontalalignment='center',size='medium',color='red',weight='bold',backgroundcolor='white')
            plt.suptitle(stock +' - Average Return '+ title +' - '+ str(start.year)+ " to "+ str(end.year))
    plt.show()

boxplot = sns.boxplot(df_index_boxplot)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
for x in boxplot.get_xticks():
    if index_period_avg[x] > 0:
        boxplot.text(x,index_period_avg[x],f'{index_period_avg[x]:.1%}',horizontalalignment='center',size='medium',color='green',weight='bold',backgroundcolor='white')
    else:
        boxplot.text(x,index_period_avg[x],f'{index_period_avg[x]:.1%}',horizontalalignment='center',size='medium',color='red',weight='bold',backgroundcolor='white')
plt.suptitle(stock +' - Average Cumulative Return '+ title +' - '+ str(start.year)+ " to "+ str(end.year))
plt.tight_layout()
pdf.savefig() 
plt.show()

#Line Plot
avg_line = sns.lineplot(x=df_index.index, y= df_index['average'],data=df_index, linewidth=3)
plt.fill_between(df_index.index, df_index['average']-df_index['sd']*2, df_index['average']+df_index['sd']*2, alpha=0.25) #2 Standard Deviation
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(x_axis_multiple))
plt.suptitle(stock +' - Average Cumulative Return '+ title +' - '+ str(start.year)+ " to "+ str(end.year))
plt.tight_layout()
pdf.savefig() 
plt.show()

#Histogram Returns after Look Forward Period
df_index_last = df_index.iloc[-1]
df_index_last[:-5].plot.hist(bins=50, alpha=0.75)
plt.axvline(df_index_last['average'], color='black', linestyle='dashed', linewidth=3)
plt.text(df_index_last['average'],1,f'{df_index_last["average"]:.1%}',horizontalalignment='center',size='large',color='black',weight='bold',backgroundcolor='white')
plt.gca().xaxis.set_major_formatter(PercentFormatter(1))
plt.suptitle(stock +' - Average '+str(look_forward)+' Day Cumulative Return '+ title +' - '+ str(start.year)+ " to "+ str(end.year))
plt.tight_layout()
pdf.savefig() 
plt.show()

pdf.close()
