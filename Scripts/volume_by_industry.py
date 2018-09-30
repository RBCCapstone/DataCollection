import pandas as pd

stocks_df = pd.read_csv('test_stocks.csv') # stocks, companies, 

# SELECT i.NAME , SUM(VOLUME) 
# FROM stocks AS s 
# JOIN companies AS c ON s.COMPANY_ID = c.ID 
# JOIN industries AS i ON c.INDUSTRY_ID = i.ID
# GROUP BY 1

# Merge resources:
#   https://stackoverflow.com/questions/23668427/pandas-three-way-joining-multiple-dataframes-on-columns
#   http://pandas.pydata.org/pandas-docs/stable/generated/pandas.merge.html#pandas.merge
# sum, group by: https://stackoverflow.com/questions/39922986/pandas-group-by-and-sum/39923815
vol_df = stocks_df.groupby(['date', 'industry_id'])['volume'].sum() 

vol_df.reset_index()

vol_df.to_csv('vol_by_industry.csv', header=True, encoding='utf-8')