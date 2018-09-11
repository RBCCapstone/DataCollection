# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 15:19:44 2018

@author: Padmanie
This script collects 5 years of exchange rate data from CAD to USD and writes it to csv
It will be used in relation to stock values pulled by another script

"""
#import libraries
import datetime
import requests
import pandas as pd

#function that converts datetime object to the currency api URL format
def dateToText(inDate):
    dateTxt = str(inDate.year) + "-" + str(inDate.month) + "-" + str(inDate.day)
    return dateTxt

#initialize at Jan. 1 2013
startDate = "13/01/01"
startDate = datetime.datetime.strptime(startDate, "%y/%m/%d")
fromDate = startDate
td = datetime.datetime.today()

#initialize dataframe and make first request to API
df = pd.DataFrame(columns=['date', 'exchange_rate'])
toDate = fromDate + datetime.timedelta(days=99)       
fromTxt = dateToText(fromDate)
toTxt = dateToText(toDate)
#CAD to USD selected
main_url = "http://currencies.apps.grandtrunk.net/getrange/" + fromTxt + "/" + toTxt + "/CAD/USD"
results = requests.get(main_url)
curdata = results.text.split("\n")

#makes requests up to today (+ up to 98 days in the future)
# nb any future 'exchange rates' are just the same rate as today's
while fromDate < td :
    print(fromDate)
    for date in curdata:
        df2 = date.split(' ')
        try:
            d = {'date': [df2[0]], 'exchange_rate': [df2[1]]}
            df2 = pd.DataFrame(data=d)    
            df = df.append(df2, ignore_index=True)            
        except:
            continue
    fromDate = toDate + datetime.timedelta(days=1)
    toDate = fromDate + datetime.timedelta(days=99)       
    fromTxt = dateToText(fromDate)
    toTxt = dateToText(toDate)
    main_url = "http://currencies.apps.grandtrunk.net/getrange/" + fromTxt + "/" + toTxt + "/CAD/USD"
    results = requests.get(main_url)
    curdata = results.text.split("\n")

# write results to csv - current file path is Padmanie's personal computer.
df.to_csv(path_or_buf='C:\\Users\\Padmanie\\Documents\\GitHub\\DataCollection\\currencies.csv', header=True)
