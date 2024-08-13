# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 02:34:44 2024

@author: Heavens Base
"""

import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
import time
import datetime as dt
import os
from Edgar_Company_Facts import get_cik, get_financial_data

#auto to prevent botting
headers = {'User-Agent': "karl.maple.beans@gmail.com"}

def price_formatter(x, pos):
    return f"${x:.2f}"
def returns_formatter(x, pos):
    return f"{x*100:.0f}%"
def millions_formatter(x, pos):
    return f"${x:1.1} M" % (x*1e-6)
def convert_df_group(df,yaxis_text = 'Period Over Period Change',columns=0):
    list_groups=[]
    for i in range(len(df.columns)+columns):
        df_group = pd.DataFrame({'Date': df.index,
                       yaxis_text: df.iloc[:,i],
                       'Group': df.columns[i]})
        list_groups.append(df_group)
    df_groups = pd.concat(list_groups).sort_index()
    df_groups.dropna(how = 'any', inplace=True)
    return df_groups
def plot_clustered_bar (df,graph_format='',tick_num = 11):
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(11, 8.5), dpi=400)
    ax = sns.barplot(x='Date', y=df.columns[1], hue='Group', data=df)
    ax.set(xlabel=None,ylabel=None)
    if graph_format != '':
        ax.yaxis.set_major_formatter(FuncFormatter(graph_format))
    if graph_format == returns_formatter:
        multiple = max(round((ax.get_ylim()[1]-ax.get_ylim()[0])/(tick_num),1),round((ax.get_ylim()[1]-ax.get_ylim()[0])/(tick_num),2))
        ax.yaxis.set_major_locator(MultipleLocator(base=multiple))
    else:  
        plt.yticks(np.linspace(plt.ylim()[0], plt.ylim()[1], (tick_num)))
    plt.grid(visible=True,which='major',axis='y',color='black',linestyle='-',alpha=0.3)
    ax.set_title(df.columns[1])
    #plt.xticks(rotation=90)
    plt.show()
def plot_line(df,title, graph_format = '',tick_num = 11, ylim_bot='', ylim_top =''):
    ax = df.plot(figsize=(11, 8.5))
    if graph_format != '':
        ax.yaxis.set_major_formatter(FuncFormatter(graph_format))
    if ylim_bot !='':
        plt.ylim(ylim_bot,ylim_top)        
    if graph_format == returns_formatter:
        multiple = max(round((ax.get_ylim()[1]-ax.get_ylim()[0])/(tick_num),1),round((ax.get_ylim()[1]-ax.get_ylim()[0])/(tick_num),2))
        ax.yaxis.set_major_locator(MultipleLocator(base=multiple))
    else:
        plt.yticks(np.linspace(plt.ylim()[0], plt.ylim()[1], (tick_num)))
    plt.grid(visible=True,which='major',axis='y',color='black',linestyle='-',alpha=0.3)
    ax.set_title(title)
    plt.show()

###Input
start = dt.datetime(2019,1,1)
end = dt.datetime.now()
stocks =['DIS','WBD','PARA']

#Get Data
ciklist = [get_cik(i) for i in stocks]
list_quarterly_data = []
list_annual_data = []
for stock in ciklist:
    #get a company's financial line item in a dataframe
    list_df = get_financial_data(stock)
    list_quarterly_data.append(list_df[0])
    list_annual_data.append(list_df[1])
    time.sleep(0.2)
    
#Quarterly
##Concats with 2 column indexes stock and financial data
df_quarterly= pd.concat(list_quarterly_data, axis=1,  keys = stocks, names = ['Stock','Data'])
df_quarterly = df_quarterly[(df_quarterly.index > start) & (df_quarterly.index < end)]
df_quarterly.dropna(how='all',inplace=True)

##Adds CY Quarter as Rows for Barchart Later
quarterly_dates = pd.date_range(start=start, end=end, freq='Q')
df_dates = pd.DataFrame(index=quarterly_dates)
df_quarterly = pd.concat([df_quarterly,df_dates])
df_quarterly = df_quarterly[~df_quarterly.duplicated(keep='first')]
df_quarterly.sort_index(inplace=True)
df_quarterly = df_quarterly.ffill()
df_quarterly_dropna = df_quarterly.dropna()
df_cal_quarterly = df_quarterly[df_quarterly.index.is_quarter_end]
df_cal_quarterly.index = df_cal_quarterly.index.date


"""Industry Sales"""
##Get sales from multiplot
df_quarterly_sales = df_quarterly.xs('Revenues', level=1, axis=1)

##Industry Sum of Companies
df_quarterly_sales['Industry'] = df_quarterly_sales.sum(axis=1, numeric_only=True)
df_quarterly_sales_dropna = df_quarterly_sales.dropna()
df_cal_quarterly_sales = df_quarterly_sales[df_quarterly_sales.index.is_quarter_end]
df_cal_quarterly_sales.index = df_cal_quarterly_sales.index.date

##Revenue Growth Index / CAGR
df_quarterly_sales_index = df_quarterly_sales_dropna / df_quarterly_sales_dropna.iloc[0,:]
num_years = (df_quarterly_sales_index.index - df_quarterly_sales_index.index[0]).days / 365.25
df_quarterly_sales_index_cagr = df_quarterly_sales_index.pow(1/num_years,axis=0) -1

#Annual
df_ltm_sales = df_quarterly_sales.rolling(4).sum()
df_ltm_sales = df_quarterly_sales.dropna(axis=0,how='all')
df_ltm_sales_dropna = df_ltm_sales.dropna()

df_annual_sales = df_ltm_sales[df_ltm_sales.index.is_year_end]
df_annual_sales_dropna = df_annual_sales.dropna()

df_ltm_sales_index = df_ltm_sales_dropna / df_ltm_sales_dropna.iloc[0,:]
num_years = (df_ltm_sales_index.index - df_ltm_sales_index.index[0]).days / 365.25
df_ltm_sales_index_cagr = df_ltm_sales_index.pow(1/num_years,axis=0) -1

df_ltm_sales_index_industry = pd.DataFrame((df_ltm_sales['Industry'] / df_ltm_sales['Industry'][0]))
num_years = (df_ltm_sales_index_industry.index - df_ltm_sales_index_industry.index[0]).days / 365.25
df_ltm_sales_index_industry['YoY Growth'] = df_ltm_sales_index_industry['Industry']/df_ltm_sales_index_industry['Industry'].shift(4)-1
df_ltm_sales_index_industry['CAGR']= df_ltm_sales_index_industry['Industry'].pow(1/num_years,axis=0) -1
df_ltm_sales_yoy_all = df_ltm_sales / df_ltm_sales.shift(4)-1
df_ltm_sales_index_industry['Median Company Growth'] = df_ltm_sales_yoy_all.iloc[:,:-1].median(axis=1)


df_annual_sales.index = df_annual_sales.index.date
df_ltm_sales_index.index = df_ltm_sales_index.index.date
df_ltm_sales_index_cagr.index = df_ltm_sales_index_cagr.index.date
df_ltm_sales_index_industry.index = df_ltm_sales_index_industry.index.date

plot_line(df_ltm_sales,"Industry LTM Revenue")
plot_line(df_ltm_sales_index_industry.iloc[:,1:] ,"Industry LTM Revenue Growth",returns_formatter,tick_num=20,ylim_bot=-0.3,ylim_top=0.3)
#plot_line(df_ltm_sales_index,"Industry LTM Revenue Index",returns_formatter)
plot_line(df_ltm_sales_index_cagr,"Industry LTM Revenue CAGR",returns_formatter)

"""Industry Margins"""
# Industry Margins
df_quarterly_total = df_cal_quarterly.groupby(level=1, axis=1).sum()
df_quarterly_total_margin = df_quarterly_total.div(df_quarterly_total['Revenues'],axis=0)
plot_line(df_quarterly_total_margin.loc[:,['GrossProfit', 'NetIncome', 'OperatingIncome']], 'Industry Quarter Margins',returns_formatter)

df_annual_total = df_quarterly_total.rolling(4).sum()
#df_annual_total = df_annual_total[df_annual_total.index.is_year_end]
df_annual_total_margin = df_annual_total.div(df_annual_total['Revenues'],axis=0)
plot_line(df_annual_total_margin.loc[:,['GrossProfit','NetIncome', 'OperatingIncome']], 'Industry LTM Margins',returns_formatter)

"""Sales by Companies"""
#Plot Revenue Growth Index --------------------------------------
plot_line(df_quarterly_sales_index.iloc[:,:len(stocks)+1], 'Quarterly Revenue Growth Index',returns_formatter)
plot_line(df_quarterly_sales_index_cagr.iloc[:,:len(stocks)+1], 'Quarterly Revenue Growth CAGR %',returns_formatter)

plot_line(df_ltm_sales_index.iloc[:,:len(stocks)], 'LTM Revenue Growth Index',returns_formatter)
plot_line(df_ltm_sales_index_cagr.iloc[:,:len(stocks)], 'LTM Revenue Growth CAGR %',returns_formatter)

##Revenue Growth Percentage
df_cal_quarterly_sales_gr_pct = df_cal_quarterly_sales / df_cal_quarterly_sales.shift(1) -1
df_cal_quarterly_groups_pct = convert_df_group(df_cal_quarterly_sales_gr_pct.iloc[-9:,:],'Quarterly Growth %',-1)
plot_clustered_bar(df_cal_quarterly_groups_pct, returns_formatter)

df_annual_sales_gr_pct = df_annual_sales / df_annual_sales.shift(1) -1
df_annual_groups_pct = convert_df_group(df_annual_sales_gr_pct,'Annual Growth %',-1)
plot_clustered_bar(df_annual_groups_pct, returns_formatter)

##Revenue Growth Dollar
df_cal_quarterly_sales_gr_dol = df_cal_quarterly_sales - df_cal_quarterly_sales.shift(1)
df_cal_quarterly_groups_dol = convert_df_group(df_cal_quarterly_sales_gr_dol,'Annual Growth $',-1)
plot_clustered_bar(df_cal_quarterly_groups_dol)

df_annual_sales_gr_dol = df_annual_sales - df_annual_sales.shift(1)
df_annual_groups_dol = convert_df_group(df_annual_sales_gr_dol,'Annual Growth $',-1)
plot_clustered_bar(df_annual_groups_dol)


#Plot Market Share Stacked Area Chart----------------------------
df_quarterly_percentage = df_quarterly_sales.iloc[:,0:len(stocks)].div(df_quarterly_sales.iloc[:,0:len(stocks)].sum(axis=1), axis=0) * 100
#fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(11, 8.5), dpi=400)
ax = df_quarterly_percentage.plot(kind='area', stacked=True,xlabel='', figsize=(11, 8.5))
ax.set_title('Market Share Quarterly')
plt.yticks(np.linspace(0, 100, 21))
plt.grid(visible=True,which='major',axis='y',color='black',linestyle='-',alpha=0.3)
plt.show()

df_annual_percentage = df_annual_sales.iloc[:,0:len(stocks)].div(df_annual_sales.iloc[:,0:len(stocks)].sum(axis=1), axis=0) * 100
#fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(11, 8.5), dpi=400)
ax = df_annual_percentage.plot(kind='area', stacked=True,xlabel='', figsize=(11, 8.5))
ax.set_title('Market Share LTM')
plt.yticks(np.linspace(0, 100, 21))
plt.grid(visible=True,which='major',axis='y',color='black',linestyle='-',alpha=0.3)
plt.show()

#Market Share Change BPS
df_cal_quarterly_percentage = df_cal_quarterly_sales.iloc[:,0:len(stocks)].div(df_cal_quarterly_sales.iloc[:,0:len(stocks)].sum(axis=1), axis=0) * 100
df_mktshr_quarterly_pct_bps = (df_cal_quarterly_percentage - df_cal_quarterly_percentage.shift(1))*100
df_quarterly_groups_bps = convert_df_group(df_mktshr_quarterly_pct_bps.iloc[-9:,:],'Change in Quarterly Market Share BPS')
plot_clustered_bar(df_quarterly_groups_bps)

df_mktshr_annual_pct_bps = (df_annual_percentage - df_annual_percentage.shift(1))*100
df_annual_groups_bps = convert_df_group(df_mktshr_annual_pct_bps,'Change in LTM Market Share BPS')
plot_clustered_bar(df_annual_groups_bps)

"""Margins by Companies"""

df_quarterly_gm = df_quarterly.xs('GrossProfit', level=1, axis=1) / df_quarterly.xs('Revenues', level=1, axis=1)
df_quarterly_om = df_quarterly.xs('OperatingIncome', level=1, axis=1) / df_quarterly.xs('Revenues', level=1, axis=1)
df_quarterly_nm = df_quarterly.xs('NetIncome', level=1, axis=1) / df_quarterly.xs('Revenues', level=1, axis=1)

plot_line(df_quarterly_gm, 'Industry Gross Margin',returns_formatter)
plot_line(df_quarterly_om, 'Industry Operating Margin',returns_formatter)
plot_line(df_quarterly_nm, 'Industry Net Income Margin',returns_formatter)

