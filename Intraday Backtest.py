# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 16:09:24 2024

@author: 14165
"""

import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.backends.backend_pdf import PdfPages
from fpdf import FPDF
from PyPDF2 import PdfMerger
import numpy as np

end = dt.datetime.now()
start = dt.datetime(2022,12,31)
timeframe = 'minute' #minute or hour
minute_interval = 30
title = 'SPY'

pdf = PdfPages(title+" Intraday Statistics "+dt.datetime.now().strftime("%Y-%m-%d")+'.pdf')
sns.set_theme(rc={'figure.figsize':(11.5,8)}) #Chart Settings
#df = pd.read_csv('QQQ.USUSD_Candlestick_1_M_BID_01.01.2022-10.08.2024.csv') #Load file name
#df = pd.read_csv('QQQ.USUSD_Candlestick_1_Hour_BID_26.01.2017-10.08.2024.csv') #Load file name

df = pd.read_csv('SPY.USUSD_Candlestick_1_M_BID_31.12.2022-10.08.2024.csv') #Load file name

#Convert format to datetime
df['Local time'] = df['Local time'].str[:-16]
df['Local time'] = pd.to_datetime(df['Local time'], format='mixed') #Converts to Datetime variable.
df = df[(df['Local time']>start) & (df['Local time']<end)] #Date Filter, Start End.
df['Day'] = df['Local time'].dt.date
if timeframe=='hour':  
    df['Hour'] = df['Local time'].dt.hour
if timeframe=='minute':
    df = df[df['Local time'].dt.minute % minute_interval == 0]
    df['Hour_Minute'] = df['Local time'].dt.strftime('%H:%M')
df.index = df['Local time']

#Price Change
if timeframe=='minute':
    df_close = df.pivot_table(index='Hour_Minute',columns='Day',values='Close')
if timeframe=='hour':  
    df_close = df.pivot_table(index='Hour',columns='Day',values='Close')
df_close_return = (df_close/df_close.shift(1) -1)
df_close_return = df_close_return[df_close_return!=0] #If value is 0 sets as nan
df_close_return = df_close_return.dropna(how='all', axis=1) #Removes holidays
df_close_return = df_close_return.dropna(how='all', axis=0) #Removes Hours without trades
df_close_return = df_close_return.transpose()

if df_close_return.shape[0] < 300:
    sns.swarmplot(df_close_return)
    plt.title(title+' Intraday Hourly Price Change')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xticks(rotation=45)
    plt.tight_layout(h_pad=0.3)
    pdf.savefig()
    plt.show()

sns.boxplot(df_close_return, showmeans = True)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.title(title+' Intraday Hourly Price Change')
plt.xticks(rotation=45)
plt.tight_layout(h_pad=0.3)
pdf.savefig()
plt.show()


df_close.index
df_close_creturn = df_close.div(df_close.loc['09:00'])-1
df_close_creturn = df_close_return[df_close_return!=0] #If value is 0 sets as nan
df_close_creturn = df_close_return.dropna(how='all', axis=1) #Removes holidays
df_close_creturn = df_close_return.dropna(how='all', axis=0) #Removes Hours without trades
df_close_creturn = df_close_return.transpose()

sns.boxplot(df_close_creturn.transpose(), showmeans = True)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.title(title+' Intraday Cumulative Daily Price Change')
plt.xticks(rotation=45)
plt.tight_layout(h_pad=0.3)
pdf.savefig()
plt.show()

df_day_max = df_close_creturn.apply(lambda x: np.where(x == x.max(), 1, 0))
df_day_min = df_close_creturn.apply(lambda x: np.where(x == x.min(), 1, 0))
df_day_max['Count'] = df_day_max.transpose().sum()
df_day_min['Count'] = df_day_min.transpose().sum()

df_day_max['Count'].plot.bar()
plt.title(title+' Intraday High of the Day')
plt.tight_layout(h_pad=0.3)
pdf.savefig()
plt.show()

df_day_min['Count'].plot.bar()
plt.title(title+' Intraday Low of the Day')
plt.tight_layout(h_pad=0.3)
pdf.savefig()
plt.show()

#Max Change
if timeframe=='minute':
    df_low = df.pivot_table(index='Hour_Minute',columns='Day',values='Low')
    df_high = df.pivot_table(index='Hour_Minute',columns='Day',values='High')
if timeframe=='hour':  
    df_low = df.pivot_table(index='Hour',columns='Day',values='Low')
    df_high = df.pivot_table(index='Hour',columns='Day',values='High')
df_pos_return = (df_high/df_low.shift(1) -1)
df_neg_return = abs(df_low/df_high.shift(1) -1)
df_max_return = df_pos_return.where(df_pos_return.ge(df_neg_return), df_neg_return)
df_max_return = df_max_return[df_max_return!=0] #If value is 0 sets as nan
df_max_return = df_max_return.dropna(how='all', axis=1) #Removes holidays
df_max_return = df_max_return.dropna(how='all', axis=0) #Removes Hours without trades
df_max_return = df_max_return.transpose()

if df_max_return.shape[0] < 300:
    sns.swarmplot(df_max_return)
    plt.title(title+' Intraday Max Absolute Price Change')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xticks(rotation=45)
    plt.tight_layout(h_pad=0.3)
    pdf.savefig()
    plt.show()

sns.boxplot(df_max_return, showmeans = True)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.title(title+' Intraday Max Price Absolute Change')
plt.xticks(rotation=45)
plt.tight_layout(h_pad=0.3)
pdf.savefig()
plt.show()

absolute_swing = 0.01
df_50p = df_max_return[df_max_return > absolute_swing]
df_50p = df_50p.applymap(lambda x: 1 if not pd.isna(x) else 0).transpose()
df_50p['Count']= df_50p.transpose().sum()
df_50p['Count'].plot.bar()
plt.title(title+' Number of '+ f'{absolute_swing:+.2%}'+' Max Absolute Value Swings')
plt.tight_layout(h_pad=0.3)
pdf.savefig()
plt.show()


#Volume
if timeframe=='minute':
    df_volume = df.pivot_table(index='Hour_Minute',columns='Day',values='Volume')
if timeframe=='hour':  
    df_volume = df.pivot_table(index='Hour',columns='Day',values='Volume')
df_volume = df_volume[df_volume!=0] #If value is 0 sets as nan
df_volume = df_volume.dropna(how='all', axis=1) #Removes holidays
df_volume = df_volume.dropna(how='all', axis=0) #Removes Hours without trades
df_volume = df_volume.transpose()

if df_volume.shape[0] < 300:
    sns.swarmplot(df_volume)
    plt.title(title+' Intraday Volume')
    plt.xticks(rotation=45)
    plt.tight_layout(h_pad=0.3)
    pdf.savefig()
    plt.show()

sns.boxplot(df_volume, showmeans = True)
plt.title(title+' Intraday Volume')
plt.xticks(rotation=45)
plt.tight_layout(h_pad=0.3)
pdf.savefig()
plt.show()
pdf.close()

