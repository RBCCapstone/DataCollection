# News API
# This script pulls data from various news sources
# Written by Jessie & Jade


#import newsapi
import requests
import csv
import os

#import API key (environment variable)
newsapiKey = os.environ['NEWSAPI_KEY']

def News(querylist, sources, fromdate, todate):
  
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
    main_url = " https://newsapi.org/v2/everything?q=(" + completequery + ")&sources=" + sources + "from=" + fromdate + "&to=" + todate + "&pageSize=100&page=1&apiKey=" + newsapiKey  

    
    # fetching data in json format
    open_bbc_page = requests.get(main_url).json() 
    totalResults = open_bbc_page["totalResults"]
    print(totalResults)
    
    #Write to CSV by page, until all articles in URL are written
    j = 1
    articlesToCSV(main_url, j) #print to csv at page 1 first
    totalResults = totalResults - 100 
    print(totalResults)
    
    while int(totalResults) > 0:
        j = j + 1 #start printing to csv at page 2
        main_url = " https://newsapi.org/v2/everything?q=(" + completequery + ")&sources=" + sources + "from=" + fromdate + "&to=" + todate + "&pageSize=100&page=" + str(j) + "&apiKey=" + newsapiKey  
        articlesToCSV(main_url, j)
        totalResults = totalResults - 100
        print(totalResults)
  
    
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
    #oldquerylist = ["Amazon", "Walmart", "Home Depot", "Comcast", "Disney", "Netflix", "McDonald's", "Costco", "Lowe's", "Twenty-First Century", "Century Fox", "Starbucks", "Charter Communications", "TJX", "American Tower", "Simon Property", "Las Vegas Sands", "Crown Castle", "Target", "Carnival", "Marriott", "Sherwin-Williams", "Prologis"]
    
    #AgriCompaniesStocks = ["GPRE", "CF", "SMG", "TSN", "DF", "NTR", "MOS", "ADM", "FDP", "CVGW"]
    AgriCompanies= ["(Green Plains)", "(CF Industries)", "(Miracle-Gro)", "(Miracle Gro)", "(Tyson Foods)", "(Dean Foods)", "Nutrien", "(Mosaic Company)", "(Archer-Daniels)","Archer Daniels", "(Del Monte)", "(Calavo Growers)"]
    SourcesPt1 = "abc-news,al-jazeera-english,associated-press,australian-financial-review,axios,bbc-news,bloomberg,business-insider,cbc-news,cbs-news,cnbc,cnn,financial-post,financial-times,fortune,fox-news,google-news,google-news-ca,independent,msnbc,national-greographic"
    SourcesPt2 = "national-review, nbc-news,newsweek,new-york-magazine,politico,recode,reuters,new-scientist,techcrunch,the-globe-and-mail,the-economist,the-huffinton-post,the-new-york-times,the-wall-street-journal,the-washington-post,time,usa-today,wired"
    BusinessSources = "bloomberg,cnbc,fortune,financial-times,financial-post,the-economist,the-wall-street-journal" #business-insider excluded. 
    
    #Define Dates to Gather Data
    Pull_From = "2017-10-30"
    Pull_To = "2018-10-28"
    
    #Define Companies to query on, if more than one word, include brackets
    RetailCompaniesStocks = ["GPS", "FL", "LB", "MAC", "KIM", "TJX", "CVS", "HD", "BBY", "LOW"]
    RetailCompanies1 = ["(Gap Inc)", "(Foot Locker)", "(L Brands)", "Macerich", "Kimco", "TJX", "CVS", "(Home Depot)", "(Best Buy)", "(Lowe's)" ]
    RetailCompanies2 = ["Walmart", "Target", "Amazon", "Kroger", "Walgreens", "Target", "Kohl's", "(Dollar General)", "(Bed Bath and Beyond)", "Safeway"]
    News(AgriCompanies, BusinessSources, Pull_From, Pull_To) 
