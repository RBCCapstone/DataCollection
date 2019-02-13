
# coding: utf-8

# In[20]:


def DataClean(raw):
    #import FeatureEncoding
    import os
    from os import listdir
    from os.path import isfile, join
    from pathlib import Path
    import pandas as pd
    import FeatureEncoding
    import re, string

    # Import Article Data including corresponding Y values
    #DATA_DIR = "Data"    
    #RAW_DIR = os.path.join(DATA_DIR, filename)
    #raw = pd.read_excel(RAW_DIR)

    #remove blanks (NaN)
    df = raw.dropna(subset = ['content', 'title']) 

    #remove blocked articles without content
    df = df[df.content.str.contains("Your usage has been flagged") == False]
    df = df[df.content.str.contains("To continue, please click the box") == False]

    #remove duplicates by url
    df = df.drop_duplicates(subset=['url'], keep='first')

    #remove duplicates by content
    df = df.drop_duplicates(subset=['content'], keep='first')

    #remove punctuation, keep orig content
    df['origContent'] = df['content']
    pattern = re.compile('[^0-9a-zA-Z ]+')
    content= map(lambda x: pattern.sub('', x), df['content'])
    df['content']=list(content)

    # Output Cleaned Article Data
    # rename index column 
    df.rename(columns={'index': 'article_id'}, inplace=True)
    
    ## Commented out excel interaction ##
    #OUTPUT_DIR = os.path.join(DATA_DIR, "cleanedArticles.csv")
    #pd.DataFrame.to_csv(df, path_or_buf=OUTPUT_DIR)
    
    #OUTPUT_DIR = os.path.join(DATA_DIR, "cleanedArticles.xlsx")
    #writer = pd.ExcelWriter(OUTPUT_DIR)
    #df.to_excel(writer,'Sheet1')
    #writer.save()
    return df


# In[21]:


def test():
    filename = "sortedarticlesRetail-List-1.xlsx"
    DataClean(filename)


# In[22]:


#test()

