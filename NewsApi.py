<<<<<<< HEAD
# News API
# This script pulls data from various news sources
# Written by Jesse & Jade


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
        the-wall-street-journal&pageSize=100&page=" + str(j) + "&apiKey=" + newsapiKey  
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
    
    for ar in article:
        titles.append(ar["title"])
        description.append(ar["description"])
        url.append(ar["url"])        

             # writing all trending news to csv        
    with open('articles.csv', 'a', newline='',encoding='utf-8') as f:
        articlewriter = csv.writer(f, delimiter = ',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(titles)):
            s = (((k-1)*100) + i + 1, titles[i], description[i], url[i] )
            articlewriter.writerow(s)
    f.close()
    

# Driver Code
if __name__ == '__main__': 
    # function call
    #querylist = ["Amazon", "Walmart", "Home Depot", "Comcast", "Disney", "Netflix", "McDonalds", "McDonald", "McDonald's", "Costco", "Lowe", "Lowe's", "Twenty-First Century", "Century Fox", "Starbucks", "Charter Communications", "Chart Communication", "TJX", "American Tower", "Simon Property", "Las Vegas Sands", "Crown Castle", "Target", "Carnival", "Marriott", "Sherwin-Williams", "Prologis"]
    querylisttemp = ["Marriott","tjx","Disney"]
    NewsFromBBC(querylisttemp) 

=======
<<<<<<< HEAD
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

=======
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

>>>>>>> de75b4d318a68ea7fecd433e4916a059079ac4fa
>>>>>>> origin/master
