# News API
# This script pulls data from various news sources
# Written by Jesse & Jade


#import newsapi
import requests
import csv
import os

#import API key (environment variable)
newsapiKey = os.environ['NEWSAPI_KEY']

def NewsFromBBC(j):
  
    # BBC news api
    for i in range(1,j):
        main_url = " https://newsapi.org/v2/everything?sources=bbc-news&pageSize=100&page=" + str(i) + "&apiKey=" + newsapiKey  
        # fetching data in json format
             
        articlesToCSV(main_url,i)

def articlesToCSV(main_url, k):
    # getting all articles in a string article
    open_bbc_page = requests.get(main_url).json()  
    article = open_bbc_page["articles"]

    # empty list which will contain all trending news
    titles = []
    description = []
    url = []
    
    for ar in article:
        titles.append(ar["title"])
        description.append(ar["description"])
        url.append(ar["url"])        

             # writing all trending news to csv        
    with open('articles.csv', 'a', newline='') as f:
        articlewriter = csv.writer(f, delimiter = ',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(titles)):
            s = (((k-1)*100) + i + 1, titles[i], description[i], url[i] )
            articlewriter.writerow(s)
    f.close()
    

# Driver Code
if __name__ == '__main__': 
    # function call
    NewsFromBBC(11) 

