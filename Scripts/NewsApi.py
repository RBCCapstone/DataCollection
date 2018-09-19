# News API
# This script pulls data from various news sources
# Written by Jessie & Jade


#import newsapi
import requests
import csv
import os

#import API key (environment variable)
newsapiKey = os.environ['NEWSAPI_KEY']

def NewsFromBBC(querylist):
  
    completequery = ""

    #Create a string to query the url        
    for i in range(len(querylist)): #defaults at index 0  
        #Create a string to query the url     
        if i < len(querylist)-1:
            completequery += querylist[i]
            completequery += " OR "
        else:
            completequery += querylist[i]
        
    print(completequery)
    
    
        
    #Find the first page
    main_url = " https://newsapi.org/v2/everything?q=(" + completequery + ")&sources=bloomberg,business-insider,cnbc,fortune,financial-times,financial-post,the-economist,\
    the-wall-street-journal&pageSize=100&page=1&apiKey=" + newsapiKey  
    #print(main_url)
    
    # fetching data in json format
    open_bbc_page = requests.get(main_url).json() 
    totalResults = open_bbc_page["totalResults"]
    #print(totalResults)
    
    #Write to CSV by page, until all articles in URL are written
    j = 1
    while int(totalResults) > 0:
        articlesToCSV(main_url, j)
        j = j + 1
        main_url = " https://newsapi.org/v2/everything?q=(" + completequery + ")&sources=bloomberg,business-insider,cnbc,fortune,financial-times,financial-post,the-economist,\
        the-wall-street-journal&from=2018-07-01&to=2018-09-19&pageSize=100&page=" + str(j) + "&apiKey=" + newsapiKey  
        totalResults = totalResults - 100 
        #print(totalResults)
        #print(j)
            
def articlesToCSV(main_url, k):
    # getting all articles in a string article
    open_bbc_page = requests.get(main_url).json()  
    article = open_bbc_page["articles"]
    
    # empty list which will contain all trending news
    titles = []
    description = []
    url = []
    publishedAt = []
    content = []
    
    for ar in article:
        titles.append(ar["title"])
        description.append(ar["description"])
        url.append(ar["url"])
        publishedAt.append(ar["publishedAt"])
        content.append(ar["content"])                

             # writing all trending news to csv        
    with open('articles.csv', 'a', newline='',encoding='utf-8-sig') as f:
        articlewriter = csv.writer(f, delimiter = ',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(titles)):
            s = (((k-1)*100) + i + 1, titles[i], description[i], url[i], publishedAt[i], content[i])
            articlewriter.writerow(s)
    f.close()
    

# Driver Code
if __name__ == '__main__': 
    # function call
    #News APi can only take 20 queries
    #querylist = ["Amazon", "Walmart", "Home Depot", "Comcast", "Disney", "Netflix", "McDonald's", "Costco", "Lowe's", "Twenty-First Century", "Century Fox", "Starbucks", "Charter Communications", "TJX", "American Tower", "Simon Property", "Las Vegas Sands", "Crown Castle", "Target", "Carnival", "Marriott", "Sherwin-Williams", "Prologis"]
    querylisttemp = ["Amazon","Walmart", "Home Depot", "Comcast", "Disney", "Netflix"]
    NewsFromBBC(querylisttemp) 
