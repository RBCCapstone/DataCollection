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

topCompanies = list(csv.DictReader(open('stocks_info.csv')))

symbols = []
for company in topCompanies:
    symbols.append(company['Symbol'])

arbSymbol = 'AAPL'

# arbitrarily create df object
# get stock history for last month
arbChart = p.chartDF(arbSymbol, timeframe='1m')
# add stock symbol to df to act as a foreign key for company table
arbChart['symbol']="Arbitrary"
# get company info
arbCompany = p.companyDF(arbSymbol)
# join the 2 tables and add to rest of stocks df
stock = pd.merge(arbChart,arbCompany,how='left',on=['symbol'])

for symbol in symbols:
    # get stock history for last 1 month
    chart = p.chartDF(symbol, timeframe='1m')
    # add stock symbol to df to act as a foreign key for company table
    chart['symbol']=symbol
    # get company info
    company = p.companyDF(symbol)
    # join the 2 tables and add to rest of stocks df
    stock = stock.append(pd.merge(chart,company,how='left',on=['symbol']))

#export data to csv
stock.to_csv('test_stocks_sept.csv', index=False, encoding='utf-8')