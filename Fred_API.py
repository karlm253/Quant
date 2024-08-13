# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 23:48:13 2024

@author: CHX
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FRED_API_KEY ='6f4a70779a055e9866adbcaa44392472'

def fred_search_series(search_words):
    response = requests.get('https://api.stlouisfed.org/fred/series/search?search_text='+search_words+'&api_key='+FRED_API_KEY+'&file_type=json')
    data = response.json()
    df = pd.DataFrame(data['seriess'])
    return df

def fred_get_series(series):
    response = requests.get('https://api.stlouisfed.org/fred/series/observations?series_id='+series+'&api_key='+FRED_API_KEY+'&file_type=json')
    data = response.json()
    df = pd.DataFrame(data['observations'])
    return df

#search_words = 'money+supply' #should use + to seperate words, you can also look at the fred website
#series = fred_get_series('GNPCA') 


fredseries_dict = {"GDP" : "GDP", 
                   "RealGDP": "GDPC1", 
                   
                   "CPI ALL Items" : "CPALTT01USM657N",
                   "Sticky Price Consumer Price Index":"STICKCPIM157SFRBATL",
                   "10Y BE Inflation": "T10YIE",
                   "10Y TIPS Market Yield": "DFII10",
                   "30Y BE Inflation": "310YIE",
                   "30Y TIPS Market Yield": "DFII30",
                   "PCE Ex Food & Energy":"DPCCRV1Q225SBEA",
                   "Trimmed Mean PCE Inflation Rate":"PCETRIM12M159SFRBDAL",
                   
                   "Unemployment Rate": "UNRATE",
                   "Unemployment Level": "UNEMPLOY",
                   "Continued Claims (Insured Unemployment)": "CCSA",
                   "Average Weeks Unemployed":"UEMPMEAN",
                   "Employment Level":"CE16OV",
                   "All Employees, Total Nonfarm":"PAYEMS",
                   
                   "University of Michigan: Consumer Sentiment":"UMCSENT",
                   "Leading Index for the United States": "USSLIND",
                   "St. Louis Fed Financial Stress Index":"STLFSI4",
                   "Chicago Fed National Financial Conditions Index":"NFCI",
                   "Chicago Fed National Financial Conditions Leverage Subindex":"NFCILEVERAGE",
                   
                   "Monetary Base Total" : "BOGMBASE",
                   "M2" : "M2SL",
                   "Retail Money Market Funds":"WRMFNS",
                   "Fed Total Assets":"WALCL",
                   
                   "ONFFR": "FEDFUNDS",
                   "ICE BofA US High Yield Index": "BAMLHYH0A0HYM2TRIV",
                   "ICE BofA US Corporate Index": "BAMLCC0A0CMTRIV",
                   "ICE BofA CCC & Lower US High Yield":"BAMLHYH0A3CMTRIV",
                   "ICE BofA Single-B US High Yield":"BAMLHYH0A2BTRIV",
                   "ICE BofA BB US High Yield":"BAMLHYH0A1BBTRIV",
                   "ICE BofA BBB US Corporate":"BAMLCC0A4BBBTRIV",
                   "ICE BofA AAA US Corporate":"BAMLCC0A1AAATRIV",
                   
                   "Quarterly Financial Report: U.S. Corporations Total Cash":"QFRTCASHINFUSNO",
                   
                   "Dates of U.S. recessions":"JHDUSRGDPBR"
                   
                   }
          

