# coding: utf-8

# # Feature Encoding
# 
# The following script formats articles by features for linear regression (first attempt).  
# It takes in a list of features and a set of articles, converts to lowercase and creates an encoded matrix (dense)  
# While this isn't the cleverest method, it provides a usable input for setting up our initial linear regression code.
# 
# ### Limitations:
# * Proper Nouns should keep their capitals
# * Punctuation/Stemming etc not incorporated
# * Bi-grams not accommodated
# * Could be converted to space matrix
# * No log function incorporated at this point
#     
# 

# In[1]:


#importing libraries
import pandas as pd
import numpy as np
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import os
from pathlib import Path
from tqdm import tqdm


# In[2]:


#import pyscripter

#relevant_nbs = ['FeatureSelection.ipynb']
#relevant_nbs = pyscripter.nb_to_py(relevant_nbs)
#print("y print 2x?")
#print(relevant_nbs)
#pyscripter.import_scripts(['FeatureSelection.py'])


# In[3]:


def loadData():
    DATA_DIR = "Data"
    FEATURES_DIR = os.path.join(DATA_DIR, "retailFeatureSet.csv")
    ARTICLES_DIR = os.path.join(DATA_DIR, "cleanedArticles.xlsx")
    
    fts = pd.read_csv(FEATURES_DIR)
    for col in fts.columns:
        if not (col.strip() == 'target_group'):
            fts = fts.drop([col], axis = 1)
    fts.columns = ['index']
    fts['index'] = list(map(lambda x: x.strip(), fts['index']))
    arts = pd.read_excel(ARTICLES_DIR)
    artText = arts.iloc[:,5] 
    artID = arts['article_id']   #**
    data = {'fts':fts, 'artText': artText, 'artID': artID} #**
    return data


# In[4]:


def binEncoding(data):
    print("Binary Encoding")
    fts = data['fts']
    artText = data['artText']
    
    df_rows = []
    tokenizer = RegexpTokenizer(r'\w+')

    for art in tqdm(artText):
        if type(art) == str: 
            body = art.lower()
            #body = clean_file_text(body)
            ##art_words = tokenizer.tokenize(body)
            ##df_rows.append([1 if word in art_words else 0 for word in fts['index']])
            body = body.split() #**
            wordsCounter = Counter(body)
            df_rows.append([1 if word in wordsCounter else 0 for word in fts['index']])
        else:
            df_rows.append([0 for word in fts['index']])
    X = pd.DataFrame(df_rows, columns = fts['index'].values)
    return X


# In[5]:


def tfEncoding(data):
    print("tf Encoding")
    fts = data['fts']
    artText = data['artText']
    
    tf_rows = []
    for art in artText:
        if type(art) == str:
            body = art.lower()
            body = body.split()
            wordsCounter = Counter(body)
            tf_rows.append([wordsCounter[word] if word in wordsCounter else 0 for word in fts['index']])
        else:
            tf_rows.append([0 for word in fts['index']])
    X = pd.DataFrame(tf_rows, columns = fts['index'].values)  
    return X


# In[6]:


def tfidfEncoding(data):
    print("tifidf Encoding")
    fts = data['fts']

    # Base calculations
    binX = binEncoding(data)
    tfX = tfEncoding(data)
    
    # Calculate idf
    df_row = [binX[word].sum() for word in fts['index']]
    idf = [1/(df+1) for df in df_row]
    #transpose list (not the cleverest method)
    idf_row = []
    idf_row.append(idf)
    idf_list = pd.DataFrame(idf_row, columns = fts['index'])
    
    # Extract term frequencies
    tf = tfX.values
    # Set up loop to multiply each article (row) by the idf per term (col)
    tf_idf = []
    r, c = tf.shape
    for art in range(0,r):
        tf_idf.append(tf[art]*idf)
    tf_idf = pd.DataFrame(tf_idf, columns = fts['index'])
    X = tf_idf
    return X


# In[7]:


def encoding(encodeType, **kwargs):
    # 0 for Binary Encoding
    # 1 for Term Frequency Encoding
    # 2 for TF-IDF Encoding
    # If you'd like to save as csv, use "csv = True"
        
    # Load up data
    data = loadData()
    # Run corresponding encoding type and pass data
    options = {0 : binEncoding,
                1 : tfEncoding,
                2 : tfidfEncoding,}
    
    X = options[encodeType](data)
    X['article_id'] = data['artID'].values #**
    print(X.head())
    
    # Save as csv file in CLASSIFICATION data folder =)
    if ('csv' in kwargs) and (kwargs['csv']):
        
        # File path for this file
        file_name = options[encodeType].__name__ + '.csv'
        thispath = Path().absolute()
        #OUTPUT_DIR = os.path.join(thispath.parent.parent, "Classification", "Data", file_name)
        # if the following line throws an error, use the line after to save in same folder
        OUTPUT_DIR = os.path.join(thispath, "Data", file_name)
        pd.DataFrame.to_csv(X, path_or_buf=OUTPUT_DIR)
        #pd.DataFrame.to_csv(X, path_or_buf=file_name)
    
    # Return Panda DataFrame
    return X
    


def main(): # Stuff to do when run from the command line    
    encoding(0, csv = True)
    pass  
 


# In[9]:


#testcell

#X = encoding(1, csv=True)
#X = encoding(2, csv=True)
#X = encoding(0, csv=True)
#X.head()


# In[ ]:





# In[ ]:




