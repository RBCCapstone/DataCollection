# DataCollection
Scripts to access APIs 

NewsApi.py - Pulls information using the NewsAPI and writes to articles.csv

articles.csv - Stores the information pulled by NewsAPI

stocksInfo.py - Pulls historical stock info for the last 5 years using pyEX library. Stocks are of the first 25 entries from S&P 500 Consumer Goods.

SampleStocks.csv - output of stocksInfo.py

TextClassification.py - Selects features from articlestestingdata1.csv . Outputs most correlated unigrams and bigrams to defined classes
