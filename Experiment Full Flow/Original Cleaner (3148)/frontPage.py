
# coding: utf-8

# What this script does
# 
# Upon a user clicking refresh
# * Run pipeline
# * Pull up most top terms & frequencies for ticker [48h?]
# * Pull up most market-moving news [50 articles] [48h?]

# In[ ]:


import os
from pathlib import Path
import json
import pandas as pd
#import pipeline as pl


# In[1]:


def FrontPage(articleDB, trendingTermsDB):
    # number of top articles
    # todo; change to only 'predicted relevant' articles
    numArts = 40
    
    # get articles
    art = articleDB.iloc[0:numArts][['title','source', 'date', 'origContent']]
    art['tags'] = list(map(lambda x: x.split(','), articleDB.iloc[0:numArts]['tags_top_5']))
    # grab related article IDs
    rel_arts = list(map(lambda x: x.split(','), articleDB.iloc[0:numArts]['related_articles']))
    # use IDs to grab related article title, source, url, turn into little dictionaries and add to art
    art['related_articles'] = list(map(lambda num: articleDB.iloc[num][['title','source','url']].to_dict(orient='records'), rel_arts))
    
    artDict = art.to_dict(orient='records')
        
    # get top terms
    tuples = [tuple(x) for x in trendingTermsDB.values]
    topTerms = tuples[:10]
    
    # output final json
    frontpage = {"topterms":topTerms, "articles":artDict}
    with open("frontPage.json", "w") as write_file:
        json.dump(frontpage, write_file)
    
    return frontpage
