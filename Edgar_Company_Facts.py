# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:21:20 2024

@author: 14165
V2 S+GA, D+A, fillna for interest and intangible (for EBITDA)
V3 Added Finance and Operating Lease (for ROIC and Debt Analysis)
"""

import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
import time
import datetime as dt
import os

#auto to prevent botting
headers = {'User-Agent': "karl.maple.beans@gmail.com"}

stocks =['AMZN']

def get_cik(stock):
    #get SEC tickers
    tickers_cik = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers)
    tickers_cik = pd.json_normalize(pd.json_normalize(tickers_cik.json(), max_level=0).values[0])
    tickers_cik["cik_str"] = tickers_cik["cik_str"].astype(str).str.zfill(10)
    tickers_cik.set_index("ticker",inplace=True)
    cik= tickers_cik.loc[stock][0]
    return cik


def get_financial_data(cik):
    response = requests.get("https://data.sec.gov/api/xbrl/companyfacts/CIK"+cik+".json", headers=headers)
    #Checks for a successful API Connection
    if response.status_code == 200:
        data = response.json()
    
    df = pd.DataFrame(data['facts']['us-gaap'])
    
    list_annual=[]
    list_quarterly=[]
    
    for line_item in df.columns:
        key = df[line_item][2]
        first_key=next(iter(key))
    
        #Combined Data ------------------------
        df_line_item = pd.DataFrame(df[line_item][2][first_key])
        df_line_item.rename(columns={"val": line_item}, inplace=True)
        df_line_item['end'] = pd.to_datetime(df_line_item['end'])
        
        try: #For Line Items with a Start Date i.e. Non-Balance Sheet
            df_line_item = df_line_item[~df_line_item.duplicated(subset=['end','start'], keep='last')] 
    
            ##Adding Days in Period
            df_line_item['start'] = pd.to_datetime(df_line_item['start'])
            df_line_item['days'] = df_line_item['end']-df_line_item['start']
            df_line_item['days'] = df_line_item['days'].dt.days
            df_line_item = df_line_item.sort_values(by=['end','start','filed'])
        
            #Annual Data -----------------------------
            df_annual = df_line_item[df_line_item['days'] > 300]
            df_annual = df_annual[~df_annual.duplicated(subset=['end'], keep='last')]
            df_annual.set_index('end', inplace=True)
            
            #Quarterly Data---------------------------
            df_quarterly = df_line_item.sort_values(by=['end','start','filed'])
            df_quarterly.reset_index(drop=True,inplace=True)
            
            ##Calculates Quarter/Quarter for YTD Figures
            df_quarterly_360 = df_quarterly.loc[(df_quarterly['days'] > 300) & (df_quarterly['days'].shift(1) < 300) & (df_quarterly['days'].shift(1) >250),line_item] - df_quarterly[line_item].shift(1)
            df_quarterly_270 = df_quarterly.loc[(df_quarterly['days'] < 300) & (df_quarterly['days'] > 250) & (df_quarterly['days'].shift(1) <250)&(df_quarterly['days'].shift(1) >150),line_item] - df_quarterly[line_item].shift(1)
            df_quarterly_180 = df_quarterly.loc[(df_quarterly['days'] < 250) & (df_quarterly['days'] > 150) & (df_quarterly['days'].shift(1) <150),line_item] - df_quarterly[line_item].shift(1)
            df_quarterly_90 = df_quarterly.loc[df_quarterly['days'] <150,line_item]
            ##Data Cleaning & Merging
            df_quarterly_var = pd.concat([df_quarterly_90,df_quarterly_180,df_quarterly_270,df_quarterly_360],axis=1)
            df_quarterly_var.iloc[:,0] = df_quarterly_var.iloc[:,0].fillna(df_quarterly_var.iloc[:,1])
            df_quarterly_var.iloc[:,0] = df_quarterly_var.iloc[:,0].fillna(df_quarterly_var.iloc[:,2])
            df_quarterly_var.iloc[:,0] = df_quarterly_var.iloc[:,0].fillna(df_quarterly_var.iloc[:,3])
            df_quarterly_var = df_quarterly_var.iloc[:,0]        
            ##Add Clean Data Back to DF then For all valid values in data_q then 'days' 
            df_quarterly[line_item+'_Q'] = df_quarterly_var    
            df_quarterly.loc[~df_quarterly[line_item+'_Q'].isna(),'days'] = 90
            ##filters for annual or quarterly only then fills with last 3 q
            df_quarterly = df_quarterly.loc[(df_quarterly['days'] > 300)|(df_quarterly['days'] <150)]
            ##FillNA for Missing Annual Figures with the past 3 rows (180 & 270 were captured in YTD)
            df_quarterly[line_item+'_Q'] = df_quarterly[line_item+'_Q'].fillna(df_quarterly[line_item] - df_quarterly[line_item+'_Q'].shift(1) - df_quarterly[line_item+'_Q'].shift(2)-df_quarterly[line_item+'_Q'].shift(3))
            df_quarterly[line_item] = df_quarterly[line_item+'_Q']
            df_quarterly.sort_index(inplace=True)
            df_quarterly = df_quarterly[~df_quarterly.duplicated(subset=['end'], keep='last')]
            df_quarterly.set_index('end', inplace=True) 
            df_quarterly.sort_values(by=['end','start','filed'])     
            #Append
            list_annual.append(df_annual[line_item])
            list_quarterly.append(df_quarterly[line_item])
            
        except: #For Balance Sheet
            df_line_item = df_line_item[~df_line_item.duplicated(subset=['end'], keep='last')]
            df_line_item.set_index('end', inplace=True)
            df_annual = df_line_item[df_line_item['fp'] == 'FY']
            list_quarterly.append(df_line_item[line_item])
            list_annual.append(df_annual[line_item])
            
    df_annual = pd.DataFrame(list_annual).T
    df_quarterly = pd.DataFrame(list_quarterly).T
    
    #Adjusts Shares Outstanding ----------------------------------------------------------------
    shares_list = ['CommonStockSharesOutstanding', 'WeightedAverageNumberOfDilutedSharesOutstanding','StockholdersEquityNoteStockSplitConversionRatio1']
    list_check = [item in df_quarterly.columns for item in shares_list]
    
    ##Cleans the shares outstanding that was affected by previous transformations
    if list_check[0] == True:
        bso = df_annual['CommonStockSharesOutstanding'].combine_first(df_quarterly['CommonStockSharesOutstanding'])
        df_quarterly['CommonStockSharesOutstanding'] = bso
        df_annual.loc[df_annual['CommonStockSharesOutstanding'].notnull(), 'CommonStockSharesOutstanding'] = bso
    
    if list_check[1] ==  True:
        dso = df_annual['WeightedAverageNumberOfDilutedSharesOutstanding'].combine_first(df_quarterly['WeightedAverageNumberOfDilutedSharesOutstanding'])
        df_quarterly['WeightedAverageNumberOfDilutedSharesOutstanding'] = dso
        df_annual.loc[df_annual['WeightedAverageNumberOfDilutedSharesOutstanding'].notnull(), 'WeightedAverageNumberOfDilutedSharesOutstanding'] = dso
    
    if list_check[2] == True:
        ## Find the stock split multiple for adjustment 
        stock_split=df_quarterly['StockholdersEquityNoteStockSplitConversionRatio1']
        stock_split[-1] = 1
        stock_split=stock_split.dropna().sort_index(ascending=False)
        stock_split=stock_split.cumprod()
        df_quarterly['StockholdersEquityNoteStockSplitConversionRatio1'] = stock_split
        df_quarterly['StockholdersEquityNoteStockSplitConversionRatio1'].fillna(method='bfill', inplace=True)
        ## Creates the Adjusted DSO Column
        df_quarterly.dropna(subset=['WeightedAverageNumberOfDilutedSharesOutstanding'], axis=0,inplace=True)  
        df_quarterly['AdjustedDSO'] = None
        ## Ads the first value to compare subsequent values off of
        df_quarterly['AdjustedDSO'][0] = df_quarterly['WeightedAverageNumberOfDilutedSharesOutstanding'][1] *stock_split[-1]
        ## Chooses the min difference between WAVG and Stock Split Multiplier. Some dates and multiplers do not match.
        for i in range(1,len(df_quarterly.index)):      
            df_quarterly['AdjustedDSO'][i] = min((df_quarterly['WeightedAverageNumberOfDilutedSharesOutstanding'][i] * stock_split), key=lambda num: abs(num - df_quarterly['AdjustedDSO'][i-1]))
    elif list_check[1] == True:
        df_quarterly['AdjustedDSO'] = df_quarterly[shares_list[1]]
    elif list_check[0] == True:
        df_quarterly['AdjustedDSO'] = df_quarterly[shares_list[0]]
    
    df_annual['AdjustedDSO'] = df_quarterly['AdjustedDSO']
    
    #Combine and Clean Key Line Items ------------------------------------
    ## Create Empty DataFrame
    
    def merge_clean_line_item(df_data,df_append,item_list, item_name):
        #Checks to see if the items exist for the stock
        list_check = [item in df_data.columns for item in item_list]
        #Filters for only the items that exist
        list_filtered = [item for item, is_present in zip(item_list,list_check) if is_present]
        #If any items exist otherwise there will be an error. Fill with first item in list then fillna with other items.
        df_append[item_name] = None
        if len(list_filtered) >0:
            df_append[item_name] = df_data[list_filtered[0]]
            for i in list_filtered[1:]:
                df_append[item_name].fillna(df_data[i], inplace=True)
    
    #Income Related Mappings
    revenue_list = ['RevenueFromContractWithCustomerExcludingAssessedTax','SalesRevenueNet','Revenues','SalesRevenueGoodsNet']
    grossprofit_list = ['GrossProfit']
    operatingincome_list = ['OperatingIncomeLoss','IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest']
    netincome_list = ['NetIncomeLoss']
    cogs_list = ['COGS_implied','CostOfGoodsAndServicesSold','CostOfGoodsSold']
    rd_list = ['ResearchAndDevelopmentExpense','ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost','ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost']
    sm_list = ['SellingAndMarketingExpense','MarketingExpense']
    ga_list = ['GeneralAndAdministrativeExpense']
    sga_list = ['SellingGeneralAndAdministrativeExpense','S+G&A']
    interest_list = ['InterestExpense','InterestIncomeExpenseNet']
    tax_list = ['IncomeTaxExpenseBenefit']
    cfo_list = ['NetCashProvidedByUsedInOperatingActivities','NetCashProvidedByUsedInOperatingActivitiesContinuingOperations']
    capex_list = ['PaymentsToAcquirePropertyPlantAndEquipment','PaymentsToAcquireProductiveAssets']
    depreciation_list = ['Depreciation']
    amort_list = ['AmortizationOfIntangibleAssets']
    depamort_list = ['DepreciationDepletionAndAmortization','DepreciationAmortizationAndAccretionNet','D+A']
    sbc_list = ['ShareBasedCompensation']
    
    #Debt Related Mappings
    cash_list = ['CashAndCashEquivalentsAtCarryingValue']
    ltdebt_list = ['LongTermDebt','LongTermDebtFairValue','LongTermDebtNoncurrent']
    ltdebtm1_list=['LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths']
    ltdebtm2_list=['LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo']
    ltdebtm3_list=['LongTermDebtMaturitiesRepaymentsOfPrincipalInYearThree']
    ltdebtm4_list=['LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFour']
    ltdebtm5_list=['LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFive']
    fleasem1_list=['FinanceLeaseLiabilityPaymentsDueNextTwelveMonths']
    fleasem2_list=['FinanceLeaseLiabilityPaymentsDueYearTwo']
    fleasem3_list=['FinanceLeaseLiabilityPaymentsDueYearThree']
    fleasem4_list=['FinanceLeaseLiabilityPaymentsDueYearFour']
    fleasem5_list=['FinanceLeaseLiabilityPaymentsDueYearFive']
    oleasem1_list=['LesseeOperatingLeaseLiabilityPaymentsDueNextTwelveMonths']
    oleasem2_list=['LesseeOperatingLeaseLiabilityPaymentsDueYearTwo']
    oleasem3_list=['LesseeOperatingLeaseLiabilityPaymentsDueYearThree']
    oleasem4_list=['LesseeOperatingLeaseLiabilityPaymentsDueYearFour']
    oleasem5_list=['LesseeOperatingLeaseLiabilityPaymentsDueAfterYearFive']
    asset_list=['Assets']
    liability_list=['Liabilities']
    equity_list=['StockholdersEquity']
    
    #ROIC
    ppe_list = ['PropertyPlantAndEquipmentNet','PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization']
    rouassets_list = ['OperatingLeaseRightOfUseAsset']
    inta_list = ['FiniteLivedIntangibleAssetsNet','IntangibleAssetsNetExcludingGoodwill']
    ca_list = ['AssetsCurrent']
    cl_list = ['LiabilitiesCurrent']
    
    df_annual_clean = pd.DataFrame()
    merge_clean_line_item(df_annual, df_annual_clean, revenue_list, 'Revenues')
    merge_clean_line_item(df_annual, df_annual_clean, grossprofit_list, 'GrossProfit')
    merge_clean_line_item(df_annual, df_annual_clean, operatingincome_list, 'OperatingIncome')
    merge_clean_line_item(df_annual, df_annual_clean, netincome_list, 'NetIncome')
    merge_clean_line_item(df_annual, df_annual_clean, cfo_list, 'CashFromOperations')
    merge_clean_line_item(df_annual, df_annual_clean, capex_list, 'Capex')
    merge_clean_line_item(df_annual, df_annual_clean, depreciation_list, 'Depreciation')
    merge_clean_line_item(df_annual, df_annual_clean, amort_list, 'Amortization')
    df_annual['D+A'] =  df_annual_clean['Depreciation'] + df_annual_clean['Amortization'] #Fills with R&D + S&M
    merge_clean_line_item(df_annual, df_annual_clean, depamort_list, 'D&A')
    df_annual['COGS_implied'] =  df_annual_clean['Revenues'] - df_annual_clean['GrossProfit']
    merge_clean_line_item(df_annual, df_annual_clean, cogs_list, 'COGS')
    merge_clean_line_item(df_annual, df_annual_clean, rd_list, 'R&DExpense')
    merge_clean_line_item(df_annual, df_annual_clean, sm_list, 'S&MExpense')
    merge_clean_line_item(df_annual, df_annual_clean, ga_list, 'G&AExpense')
    df_annual['S+G&A'] =  df_annual_clean['S&MExpense'] + df_annual_clean['G&AExpense'] #Fills with R&D + S&M
    merge_clean_line_item(df_annual, df_annual_clean, sga_list, 'SG&AExpense')                      
    merge_clean_line_item(df_annual, df_annual_clean, interest_list, 'Interest')
    merge_clean_line_item(df_annual, df_annual_clean, tax_list, 'Tax')    
    merge_clean_line_item(df_annual, df_annual_clean, sbc_list, 'SBC')
    merge_clean_line_item(df_annual, df_annual_clean, ltdebt_list, 'LongTermDebt')
    merge_clean_line_item(df_annual, df_annual_clean, cash_list, 'Cash')      
    merge_clean_line_item(df_annual, df_annual_clean, ltdebtm1_list, 'LTDebtDueNTM')
    merge_clean_line_item(df_annual, df_annual_clean, ltdebtm2_list, 'LTDebtDue2Y')
    merge_clean_line_item(df_annual, df_annual_clean, ltdebtm3_list, 'LTDebtDue3Y')
    merge_clean_line_item(df_annual, df_annual_clean, ltdebtm4_list, 'LTDebtDue4Y')
    merge_clean_line_item(df_annual, df_annual_clean, ltdebtm5_list, 'LTDebtDue5Y+')
    merge_clean_line_item(df_annual, df_annual_clean, fleasem1_list, 'FinLeaseDueNTM')
    merge_clean_line_item(df_annual, df_annual_clean, fleasem2_list, 'FinLeaseDue2Y')
    merge_clean_line_item(df_annual, df_annual_clean, fleasem3_list, 'FinLeaseDue3Y')
    merge_clean_line_item(df_annual, df_annual_clean, fleasem4_list, 'FinLeaseDue4Y')
    merge_clean_line_item(df_annual, df_annual_clean, fleasem5_list, 'FinLeaseDue5Y+')
    merge_clean_line_item(df_annual, df_annual_clean, oleasem1_list, 'OpLeaseDueNTM')
    merge_clean_line_item(df_annual, df_annual_clean, oleasem2_list, 'OpLeaseDue2Y')
    merge_clean_line_item(df_annual, df_annual_clean, oleasem3_list, 'OpLeaseDue3Y')
    merge_clean_line_item(df_annual, df_annual_clean, oleasem4_list, 'OpLeaseDue4Y')
    merge_clean_line_item(df_annual, df_annual_clean, oleasem5_list, 'OpLeaseDue5Y+')
    merge_clean_line_item(df_annual, df_annual_clean, ppe_list, 'PPE')
    merge_clean_line_item(df_annual, df_annual_clean, rouassets_list, 'ROUAssets')
    merge_clean_line_item(df_annual, df_annual_clean, inta_list, 'IntangibleAssets')
    merge_clean_line_item(df_annual, df_annual_clean, ca_list, 'CurrentAssets')
    merge_clean_line_item(df_annual, df_annual_clean, cl_list, 'CurrentLiabilities')
    merge_clean_line_item(df_annual, df_annual_clean, asset_list, 'TotalAssets')
    merge_clean_line_item(df_annual, df_annual_clean, liability_list, 'TotalLiabilities')
    merge_clean_line_item(df_annual, df_annual_clean, equity_list, 'TotalStockholderEquity')
    df_annual_clean.dropna(how='all',inplace=True)

    df_quarterly_clean = pd.DataFrame()
    merge_clean_line_item(df_quarterly, df_quarterly_clean, revenue_list, 'Revenues')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, grossprofit_list, 'GrossProfit')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, operatingincome_list, 'OperatingIncome')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, netincome_list, 'NetIncome')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, cfo_list, 'CashFromOperations')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, capex_list, 'Capex')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, depreciation_list, 'Depreciation')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, amort_list, 'Amortization')
    df_quarterly['D+A'] =  df_quarterly_clean['Depreciation'] + df_quarterly_clean['Amortization']
    merge_clean_line_item(df_quarterly, df_quarterly_clean, depamort_list, 'D&A')
    df_quarterly['COGS_implied'] =  df_quarterly_clean['Revenues'] - df_quarterly_clean['GrossProfit']
    merge_clean_line_item(df_quarterly, df_quarterly_clean, cogs_list, 'COGS')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, rd_list, 'R&DExpense')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, sm_list, 'S&MExpense')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, ga_list, 'G&AExpense')
    df_quarterly['S+G&A'] =  df_quarterly_clean['S&MExpense'] + df_quarterly_clean['G&AExpense']
    merge_clean_line_item(df_quarterly, df_quarterly_clean, sga_list, 'SG&AExpense') 
    merge_clean_line_item(df_quarterly, df_quarterly_clean, interest_list, 'Interest')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, tax_list, 'Tax')    
    merge_clean_line_item(df_quarterly, df_quarterly_clean, sbc_list, 'SBC')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, ltdebt_list, 'LongTermDebt')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, cash_list, 'Cash')  
    merge_clean_line_item(df_quarterly, df_quarterly_clean, ltdebtm1_list, 'LTDebtDueNTM')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, ltdebtm2_list, 'LTDebtDue2Y')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, ltdebtm3_list, 'LTDebtDue3Y')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, ltdebtm4_list, 'LTDebtDue4Y')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, ltdebtm5_list, 'LTDebtDue5Y+')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, fleasem1_list, 'FinLeaseDueNTM')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, fleasem2_list, 'FinLeaseDue2Y')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, fleasem3_list, 'FinLeaseDue3Y')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, fleasem4_list, 'FinLeaseDue4Y')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, fleasem5_list, 'FinLeaseDue5Y+')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, oleasem1_list, 'OpLeaseDueNTM')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, oleasem2_list, 'OpLeaseDue2Y')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, oleasem3_list, 'OpLeaseDue3Y')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, oleasem4_list, 'OpLeaseDue4Y')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, oleasem5_list, 'OpLeaseDue5Y+')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, ppe_list, 'PPE')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, rouassets_list, 'ROUAssets')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, inta_list, 'IntangibleAssets')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, ca_list, 'CurrentAssets')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, cl_list, 'CurrentLiabilities')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, asset_list, 'TotalAssets')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, liability_list, 'TotalLiabilities')
    merge_clean_line_item(df_quarterly, df_quarterly_clean, equity_list, 'TotalStockholderEquity')
    df_quarterly_clean.dropna(how='all',inplace=True)
    
    #New Calculations -------------------------
    df_quarterly_clean['FCFE'] = df_quarterly_clean['CashFromOperations'] - df_quarterly_clean['Capex']
    df_quarterly_clean['FCFEexSBC'] = df_quarterly_clean['CashFromOperations'] - df_quarterly_clean['Capex'] - df_quarterly_clean['SBC']
    df_quarterly_clean['EBT'] = df_quarterly_clean['NetIncome'] + df_quarterly_clean['Tax'].fillna(0)
    df_quarterly_clean['EBIT'] = df_quarterly_clean['NetIncome'] + df_quarterly_clean['Tax']+ df_quarterly_clean['Interest'].fillna(0)
    df_quarterly_clean['EBITDA'] =  df_quarterly_clean['EBIT'] + df_quarterly_clean['D&A'].fillna(0)
    df_quarterly_clean['EBITDARD'] = df_quarterly_clean['EBITDA'] + df_quarterly_clean['R&DExpense'].fillna(0)
    df_quarterly_clean['NOPAT'] = df_quarterly_clean['OperatingIncome'] * (df_quarterly_clean['NetIncome']/df_quarterly_clean['EBT'])
    df_quarterly_clean['NetWorkingCapital'] = df_quarterly_clean['CurrentAssets'] - df_quarterly_clean['CurrentLiabilities']
    df_quarterly_clean['InvestedCapital'] = df_quarterly_clean['PPE'] + df_quarterly_clean['ROUAssets'].fillna(0)+ df_quarterly_clean['IntangibleAssets'].fillna(0) + df_quarterly_clean['NetWorkingCapital']
    df_quarterly_clean['ROIC'] = df_quarterly_clean['NOPAT'] / df_quarterly_clean['InvestedCapital']
    df_quarterly_clean['ROIIC'] = (df_quarterly_clean['NOPAT'] - df_quarterly_clean['NOPAT'].shift(4)) / (df_quarterly_clean['InvestedCapital'] - df_quarterly_clean['InvestedCapital'].shift(4))
    df_quarterly_clean['ROE'] = df_quarterly_clean['NetIncome'] / df_quarterly_clean['TotalStockholderEquity']
    df_quarterly_clean['ROCE'] = df_quarterly_clean['EBIT'] / (df_quarterly_clean['TotalAssets'] - df_quarterly_clean['CurrentLiabilities'])
    df_quarterly_clean['NCAV'] = df_quarterly_clean['CurrentAssets'] - df_quarterly_clean['TotalLiabilities']
    
    bs_list = ['LongTermDebt','Cash','LTDebtDueNTM','LTDebtDue2Y','LTDebtDue3Y','LTDebtDue3Y','LTDebtDue4Y','LTDebtDue5Y+','FinLeaseDueNTM','FinLeaseDue2Y','FinLeaseDue3Y','FinLeaseDue3Y','FinLeaseDue4Y','FinLeaseDue5Y+','OpLeaseDueNTM','OpLeaseDue2Y','OpLeaseDue3Y','OpLeaseDue3Y','OpLeaseDue4Y','OpLeaseDue5Y+'
               ,'PPE','ROUAssets','IntangibleAssets','CurrentAssets','CurrentLiabilities','TotalAssets','TotalLiabilities','TotalStockholderEquity','TotalStockholderEquity','NetWorkingCapital','InvestedCapital','NCAV']
    
    df_ltm_clean = df_quarterly_clean.rolling(4).sum()
    df_ltm_clean['Adjusted DSO'] = df_quarterly['AdjustedDSO']
    
    for i in bs_list:
        df_ltm_clean[i] = df_quarterly_clean[i]

    #Returns Quarterly_Clean, Annual_Clean, Quarterly_Raw, Annual_Raw
    list_df = [df_quarterly_clean,df_annual_clean,df_ltm_clean,df_quarterly,df_annual]
    return list_df


ciklist = [get_cik(i) for i in stocks]

list_df = get_financial_data(ciklist[0])

df_quarterly_clean= list_df[0].T
df_annual_clean = list_df[1].T
df_ltm_clean = list_df[2].T
df_quarterly = list_df[3].T
df_annual = list_df[3].T

#Save to CSV
#current_path = os.getcwd()
#folder_path = r"C:\Users\Heavens Base\Python Scripts\Test" 
#file_name = '\\' +stocks[0]+' - ' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") +'.csv'
#df_annual.to_csv(folder_path+file_name, index=True)


