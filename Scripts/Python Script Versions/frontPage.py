
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
    numArts = 10
    
    # get articles
    art = articleDB.iloc[0:numArts][['title','source', 'date']]
    art['tags'] = list(map(lambda x: x.split(','), articleDB.iloc[0:numArts]['tags_top_5']))
    artDict = art.to_dict(orient='records')
        
    # get top terms
    tuples = [tuple(x) for x in trendingTermsDB.values]
    topTerms = tuples[:10]
    
    # output final json
    frontpage = {"topterms":topTerms, "articles":artDict}
    with open("frontPage.json", "w") as write_file:
        json.dump(frontpage, write_file)
    

