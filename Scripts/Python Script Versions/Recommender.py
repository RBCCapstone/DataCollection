
# coding: utf-8

# In[20]:


#import bin-encoding data
import pandas as pd
import numpy as np
import scipy.stats as stats
import sklearn
import random
import os
from pathlib import Path
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report 
from sklearn.metrics.pairwise import cosine_similarity


# In[21]:


def LoadData(filename):
    DATA_DIR = "Data"
    ENCODING_DIR = os.path.join(DATA_DIR, filename)
    data = pd.read_csv(ENCODING_DIR, index_col=[0])
    return data


# In[33]:


def pairup(Csim, npV, rows, X,Y):
    for i in range(rows):
        #find most related articles indexed
        a = npV[i,:]
        index = np.argpartition(a, -4)[-4:]
        index2= index[np.argsort(a[index])]

        #show the index in X matrix
        print(i)
        print(index)
        print(index2)
        #show the similarity value
        print(a[index2])

        related = []
        #ensure that same article is not ranked as the most similar article
        for j in range(3,-1,-1):
            if i == index2[j]:
                pass #do not count the same article as most related
            elif len(related) == 3:
                pass
            else:
                related.append(str(X.iloc[index2[j]]['article_id']))

        Y.at[i, 'related_articles'] = ', '.join(related)

    Final = Y[['article_id', 'related_articles']]
        
    return Final
    


# In[23]:


def recommender(Encoding, contextoutput):
    X = LoadData(Encoding)
    Temp = X.drop(columns=['article_id'])
    
    #Similarity matrix between each article
    Csim = cosine_similarity(Temp)

    #convert to numpy
    npV = np.asarray(Csim)
    rows = np.size(npV,0)
    
    #set a temporary copy
    Y = X
    
    #match most related articles by article index
    finalMatches = pairup(Csim, npV, rows, X,Y)
    
    #load final table output
    DATA_DIR = "Data"
    ENCODING_DIR = os.path.join(DATA_DIR, contextoutput)
    contextTable = pd.read_excel(ENCODING_DIR, index_col=[0])
    
    finalTable = contextTable.merge(finalMatches, on='article_id', how='left')
    return finalTable
    

