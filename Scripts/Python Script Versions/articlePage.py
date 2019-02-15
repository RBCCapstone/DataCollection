
# coding: utf-8

# # ArticlePage.json output file
# 
# What this script does:
# * input: article_id
# * grab: article title, content, etc, & related articles
# * take related article_ids and grab their titles & sources
# * output: according to output_test_script format
# 
# What this script needs to do:
# * incorporate markdown tagging

# In[2]:


import os
from pathlib import Path
import json
import pandas as pd


# In[18]:


def ArticlePage(mainID):
    DATA_DIR = "Data"
    ARTICLES_DIR = os.path.join(DATA_DIR, "FinalArticles.xlsx")

    arts = pd.read_excel("FinalArticles.xlsx")
    print(arts.columns)
    #arts = arts.set_index('article_id')

    # Get the Main Article
    mainArt = arts.loc[mainID, : ][['title', 'url', 'date', 'content','tags', 'related_articles']]
    #mainArt = mainArt.drop(['Classify', 'market_moving'])
    mainArt['date'] = str(mainArt['date'])
    mainArt = mainArt.to_dict()

    # Get the related articles - Titles and sources
    relIDs = mainArt['related_articles'].split(', ')
    print(relIDs)
    relArts = []
    relArts.append([arts.loc[int(relID), : ][['title', 'url']] for relID in relIDs])
    df = pd.DataFrame(relArts[0])
    df = df.rename(columns={'url': 'source'})
    df['article_id'] = relIDs
    relDict = df.to_dict(orient='records')

    articlePage = { "main article": mainArt, "related articles": relDict}

    with open("articlePage.json", "w") as write_file:
        json.dump(articlePage, write_file)
    return


# In[19]:


ArticlePage(2)

