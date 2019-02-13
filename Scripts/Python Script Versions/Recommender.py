
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


# In[33]:


def pairup(npV, rows, Y):
    for i in range(rows):
        #find most related articles indexed
        a = npV[i,:]
        index = np.argpartition(a, -4)[-4:]
        index2= index[np.argsort(a[index])]

        #show the index in X matrix
        #print(i)
        #print(index)
        #print(index2)
        #show the similarity value
        #print(a[index2])

        related = []
        #ensure that same article is not ranked as the most similar article
        for j in range(3,-1,-1):
            if i == index2[j]:
                pass #do not count the same article as most related
            elif len(related) == 3:
                pass
            else:
                related.append(str(index2[j]))

        Y.at[i, 'related_articles'] = ', '.join(related)

    return Y[['related_articles']]


# In[23]:


def recommender(Encoding, contextTable):
    Encoded = Encoding.drop(columns=['article_id'])
    
    #Similarity matrix between each article
    Csim = cosine_similarity(Encoded)

    #convert to numpy
    npV = np.asarray(Csim)
    rows = np.size(npV,0)

    
    #match most related articles by article index
    finalMatches = pairup(npV, rows, Encoded)
    finalTable = contextTable.join(finalMatches, how='left')
    
    return finalTable
    

