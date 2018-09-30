import pyEX as p # iEX finance unofficial library
import pandas as pd
import csv
# all data is output as a data frame -- basically a table you can manipulate
# dfs have sql functions but with their own syntax
# resources: 
# https://codeburst.io/how-to-rewrite-your-sql-queries-in-pandas-and-more-149d341fc53e
# https://pandas.pydata.org/pandas-docs/stable/merging.html#database-style-dataframe-joining-merging

# creating an arbitrary list of symbols to pull by finding stocks similar to Amazon and Walmart

#symbols = p.peersDF('AMZN')

# appending walmart to original df and dropping duplicates
#symbols = symbols.append(p.peersDF('WMT')).drop_duplicates()

# creates list of symbols to iterate through
#symbols = symbols.index.tolist()

topCompanies = list(csv.DictReader(open('stocks_list.csv')))

companies = []
for company in topCompanies:
    companies.append(company['name'])

companies = {company:{} for company in companies}

for company in topCompanies:
    companies[company['name']]['symbol']=company['symbol']
    companies[company['name']]['industry_id']=company['industry_id']
    
arbSymbol = 'AAPL'

# arbitrarily create df object
# get stock history for last month
arbChart = p.chartDF(arbSymbol, timeframe='1m')
# add stock symbol to df to act as a foreign key for company table
arbChart['symbol']="Arbitrary"
arbChart['industry_id']=0
# get company info
arbCompany = p.companyDF(arbSymbol)
# join the 2 tables and add to rest of stocks df
stocks_df = pd.merge(arbChart,arbCompany,how='left',on=['symbol'])

for company in companies.values():
    symbol = company['symbol']
    industry_id = company['industry_id']
    # get stock history for last 1 month
    chart_df = p.chartDF(symbol, timeframe='1m')
    # add stock symbol to df to act as a foreign key for company table
    chart_df['symbol']=symbol
    # add our unique identifier for industry to the table
    chart_df['industry_id']=industry_id
    # get company info
    company_df = p.companyDF(symbol)
    # join the 2 tables and add to rest of stocks df
    stocks_df = stocks_df.append(pd.merge(chart_df,company_df,how='left',on=['symbol']))

#export data to csv
stocks_df.to_csv('test_stocks_sept.csv', index=False, encoding='utf-8')