
# coding: utf-8

# Takes the following inputs: manual (1,0), Pull from, pull to (can be left blank), CompanyList (1-5) , where 5 is all 19 companies

#import libraries
import requests
import csv
import os
import pandas as pd
from datetime import datetime, timedelta


# In[1]:

# News API
# This script pulls data from various news sources
# Written by Jessie & Jade

#import API key (environment variable)
#newsapiKey = os.environ['NEWSAPI_KEY']
newsapiKey = 'abd1cde781dc46b385045b20e214a7e8'

def News(querylist, sources, fromdate, todate):
  

    #Create a string to query the url    
    completequery = ""    
    for i in range(len(querylist)): #defaults at index 0  
        #Create a string to query the url     
        if i < len(querylist)-1:
            completequery += querylist[i]
            completequery += " OR "
        else:
            completequery += querylist[i]
     
    #Inform user on articles to print           
    print("Gathering articles on "+ completequery+ " from: "+fromdate+" to "+todate)
    
    #Find the first page
    main_url = " https://newsapi.org/v2/everything?q=(" + completequery + ")&sources=" + sources + "&from=" + fromdate + "&to=" + todate + "&pageSize=100&page=1&apiKey=" + newsapiKey  

    
    # fetching data in json format
    open_bbc_page = requests.get(main_url).json() 
    totalResults = open_bbc_page["totalResults"]
    print(totalResults)
    
    #Write to CSV by page, until all articles in URL are written
    j = 1
    articlesToCSV(main_url, j) #print to csv at page 1 first
    totalResults = totalResults - 100 
   
    
    while int(totalResults) > 0:
        j = j + 1 #start printing to csv at page 2
        main_url = " https://newsapi.org/v2/everything?q=(" + completequery + ")&sources=" + sources + "&from=" + fromdate + "&to=" + todate + "&pageSize=100&page=" + str(j) + "&apiKey=" + newsapiKey
        articlesToCSV(main_url, j)
        totalResults = totalResults - 100

    
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
    source = []
    
    for ar in article:
        titles.append(ar["title"])
        description.append(ar["description"])
        url.append(ar["url"])
        source.append(ar["source"])
        publishedAt.append(ar["publishedAt"])
        content.append(ar["content"])                

             # writing all trending news to csv        
    with open('articles.csv', 'a', newline='',encoding='utf-8-sig') as f:
        articlewriter = csv.writer(f, delimiter = ',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(titles)):
            s = (((k-1)*100) + i + 1, titles[i], description[i], url[i], source[i], publishedAt[i], content[i])
            articlewriter.writerow(s)
    f.close()


# In[2]:


def sortarticles():
    #If You would like the articles sorted as well, run this code before opening the articles.csv file 
    path = 'Data'
    
    articles = pd.read_csv("articles.csv", header= None)
    articles.columns = ["index", "title", "description", "url", "source", "date", "content"]
    articles.head()
    
    # convert column to datetype
    articles['date']=pd.to_datetime(articles.date)
    
    #Sort by date and export as xlsx (easier to work with as xlsx)
    articles = articles.sort_values(by='date')
    
    
    if not os.path.exists(path):
        os.makedirs(path)

    writer = pd.ExcelWriter(os.path.join(path, 'newsApiOutput.xlsx'), engine='xlsxwriter')
    articles.to_excel(writer,'Sheet1')
    writer.save()

# In[3]:


# Driver Code
def main(manual, Pullfrom, Pullto, CompanyList):
    # function call
    
    #News APi can only take 20 queries, different querying alternatives are be
    #oldquerylist = ["Amazon", "Walmart", "Home Depot", "Comcast", "Disney", "Netflix", "McDonald's", "Costco", "Lowe's", "Twenty-First Century", "Century Fox", "Starbucks", "Charter Communications", "TJX", "American Tower", "Simon Property", "Las Vegas Sands", "Crown Castle", "Target", "Carnival", "Marriott", "Sherwin-Williams", "Prologis"]
    
    #AgriCompaniesStocks = ["GPRE", "CF", "SMG", "TSN", "DF", "NTR", "MOS", "ADM", "FDP", "CVGW"]
    #AgriCompanies= ["(Green Plains)", "(CF Industries)", "(Miracle-Gro)", "(Miracle Gro)", "(Tyson Foods)", "(Dean Foods)", "Nutrien", "(Mosaic Company)", "(Archer-Daniels)","Archer Daniels", "(Del Monte)", "(Calavo Growers)"]
    #SourcesPt1 = "abc-news,al-jazeera-english,associated-press,australian-financial-review,axios,bbc-news,bloomberg,business-insider,cbc-news,cbs-news,cnbc,cnn,financial-post,financial-times,fortune,fox-news,google-news,google-news-ca,independent,msnbc,national-greographic"
    #SourcesPt2 = "national-review, nbc-news,newsweek,new-york-magazine,politico,recode,reuters,new-scientist,techcrunch,the-globe-and-mail,the-economist,the-huffinton-post,the-new-york-times,the-wall-street-journal,the-washington-post,time,usa-today,wired"
    BusinessSources = "bloomberg,cnbc,fortune,financial-times,financial-post,the-economist,the-wall-street-journal" #business-insider excluded. 
    
    #Define Dates to Gather Data, can set manual dates or use Today - 1
    #Today's date 
    today = datetime.today().strftime('%Y-%m-%dT%H:%M:00')
    yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:00')
    
    if manual:
        Pull_From = Pullfrom
        Pull_To = Pullto
    else:
        Pull_From = yesterday
        Pull_To = today

    #Define Companies to query on, if more than one word, include brackets
    RetailCompaniesStocks = ["GPS", "FL", "LB", "MAC", "KIM", "TJX", "CVS", "HD", "BBY", "LOW"]
    RetailCompanies1 = ["(Gap Inc)", "(Foot Locker)", "(L Brands)", "Macerich", "Kimco", "TJX", "CVS", "(Home Depot)", "(Best Buy)", "(Lowe's)" ]
    RetailCompanies2 = ["Walmart"]
    RetailCompanies3 = ["(Target's)", "TGT"]
    RetailCompanies4 = ["Amazon"]
    RetailCompanies5 = ["Walgreens", "Kohl's", "(Dollar General)", "(Bed Bath and Beyond)", "Safeway","Kroger"]
    RetailCompaniesAll = ["(Gap Inc)", "(Foot Locker)", "(L Brands)", "Macerich", "Kimco", "TJX", "CVS", "(Home Depot)", "(Best Buy)", "(Lowe's)","Walmart", "(Target's)", "TGT", "Amazon", "Kroger","Walgreens", "Kohl's", "(Dollar General)", "(Bed Bath and Beyond)", "Safeway" ]
    
    
    
    #Run to collect articles that fit within your query (for Team use)
    if CompanyList == 6:
        News(RetailCompaniesAll, BusinessSources, Pull_From, Pull_To)
    elif CompanyList == 1:
        News(RetailCompanies1, BusinessSources, Pull_From, Pull_To)
    elif CompanyList == 2:
        News(RetailCompanies2, BusinessSources, Pull_From, Pull_To)
    elif CompanyList == 3:
        News(RetailCompanies3, BusinessSources, Pull_From, Pull_To)
    elif CompanyList == 4:
        News(RetailCompanies4, BusinessSources, Pull_From, Pull_To)
    else:
        News(RetailCompanies5, BusinessSources, Pull_From, Pull_To)


    #Function call below to sort the articles by date
    sortarticles()
    
