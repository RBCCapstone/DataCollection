
# coding: utf-8

# In[1]:


#Get features (stops words removed) by tokenizing corpus - no stemming in baseline
#Binary encoding
#Assign target group 
#Use mutual information to get final feature set


# In[2]:


import os
import re
from pathlib import Path
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.feature_selection import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import datasets
import matplotlib
import matplotlib.pyplot as plt


# In[3]:


def importData():
    #Import Labelled Data
    DATA_DIR = "Data"
    thispath = Path().absolute()
    #dtype = {"index": str, "title": str, "description": str, "url": str, "date": str, "Retail Relevance": str, "Economy Relevant": str, "Market moving": str}
    RET_ARTICLES = os.path.join(DATA_DIR, "retailarticles-18-11-06.xlsx")

    
    df = pd.read_excel(RET_ARTICLES)

    try:
        df.head()
    except:
        pass
    return df


# In[4]:


def assignStopWords(): 
    #Stop_words list Options
    #Variation 1: added stop words starting at 'one'
    stop_words = {'audio','i', 'me', 'us', 'my','myself','we','our','ours', 'ourselves','you', 'your', 'yours', 'yourself', 'yourselves','he',	 'him',	 'his',	 'himself',	 'she',	 'her',	 'hers',	 'herself',	 'it',	 'its',	 'itself',	 'they','them','their', 'theirs', 'themselves', 'what', 'which', 'who','whom', 'this', 'that', 'these', 'those',	 'am',	 'is',	 'are',	 'was',	 'were',	 'be',	 'been',	 'being',	 'have',	 'has',	 'had',	 'having',	 'do',	 'does',	 'did',	 'doing',	 'a',	 'an',	 'the',	 'and',	 'but',	 'if',	 'or',	 'because',	 'as',	 'until',	 'while',	 'of',	 'at',	 'by',	 'for',	 'with',	 'about',	 'into',	 'through',	 'during',	 'before',	 'after',	 'to',	 'from','up','down','in','out','on','off','over',	 'under',	 'again',	 'further',	 'then',	 'once',	 'here',	 'there',	 'when',	 'where',	 'why',	 'how',	 'all',	 'any',	 'both',	 'each',	 'few',	 'more',	 'most',	 'other',	 'some',	 'such',	 'no',	 'nor',	 'not',	 'only','own','same', 'so','than', 'too','very','s','t','can', 'will', 'just','don','should', 'now','one','two','twenty','three','thirty','four','forty','five','fifty','six','sixty','seven','seventy','eight','eighty','nine','ninety','ten','co','re','percent','make','example','would','18','says','put','includes','keep','already','continue','even','17','asked','enough','might','ve','8','amp','seems','ai','get','team','fox','side','give','tell','take','across','non','fact','0','looks','7','pace','monday','tuesday','wednesday','thursday','friday','saturday','use','30','11','read','programme','please','something','50','60','leave','using','car','musk', 'name','january','february','march','april','may','june','july','august','september','october','november','december'}

    #from nltk.corpus import stopwords
    #stop_words = set(stopwords.words('english'))
    #print(stop_words)
    return stop_words


# In[5]:


def corpus_count_words(df, stop_words):
    tokenizer = RegexpTokenizer(r'\w+')
    word_counter = Counter()
    for row in df.itertuples(index=True, name='Pandas'):
            attribute = str((row, 'content'))
            file_words = tokenizer.tokenize(attribute)
            #keep lowercased words that are not stop words as features
            file_wordsNS = [word.lower() for word in file_words if not word.lower() in stop_words]
            # remove words that are numbers
            file_wordsN = [word.lower() for word in file_wordsNS if not word.isnumeric()]
            #remove words with a word length less than 4 (i.e. 1-3)
            file_wordsF = [word.lower() for word in file_wordsN if not len(word)<4]
            word_counter.update(file_wordsF)
    return word_counter


# In[6]:


# news_cnt = corpus_count_words(df1,stop_words)


# In[7]:


# news_cnt.most_common(30)


# In[8]:


#Binary encoding for features, also appends retail target group
def binary_encode_features(newsarticles, top_words):
    tokenizer = RegexpTokenizer(r'\w+')
    df_rows = []
    for row in newsarticles.itertuples(index=True, name='Pandas'):
            attribute = str((row, 'content'))
            file_words = tokenizer.tokenize(attribute)
            df_rows.append([1 if word.lower() in file_words else 0 for word in top_words])      
    X = pd.DataFrame(df_rows, columns = top_words)
    
    return X


# In[13]:


def mutualInformation(B_Encoding, y, top_words): 
    #Estimate mutual information for a discrete target variable.
    #Mutual information (MI) [1] between two random variables is a non-negative value, which measures the dependency between the variables.
    #It is equal to zero if and only if two random variables are independent, and higher values mean higher dependency.
    featureVals= mutual_info_classif(B_Encoding, y, discrete_features='auto', n_neighbors=3, copy=True, random_state=None)
    
    np.asarray(featureVals)

    Temp= pd.DataFrame(featureVals, columns = ['MI_Values'])
 
    Final = Temp.assign(target_group = top_words)
    
    Highest_Features = Final.nlargest(10000, 'MI_Values')
    
    return Highest_Features


# In[14]:


def selectFeatures(**kwargs):
    df = importData()
    stop_words = assignStopWords()
    
    #Select subset of orig data
    df1 = df[['content','Retail Relevance']]    
    news_cnt = corpus_count_words(df1, stop_words)
    
    num_features = 10000
    top_words = [word for (word, freq) in news_cnt.most_common(num_features)]
    B_Encoding = binary_encode_features(df1, top_words)
    print(B_Encoding.head())
    y = df['Retail Relevance']
    B_Encoding.assign(target_group=y)
      
    
    Highest_Features = mutualInformation(B_Encoding, y, top_words)
    Highest_Features = pd.DataFrame(Highest_Features)
    
    # Save as csv file in DATACOLLECTION data folder (bc it's needed for encoding script)
    if ('csv' in kwargs) and (kwargs['csv']):
        
        # File path for this file
        file_name = 'retailFeatureSet-PMI.csv'
        thispath = Path().absolute()
        OUTPUT_DIR = os.path.join(thispath, "Data", file_name)
        
        # if the following line throws an error, use the line after to save in same folder
        pd.DataFrame.to_csv(Highest_Features, path_or_buf=OUTPUT_DIR)
        #pd.DataFrame.to_csv(Highest_Features, path_or_buf=file_name)
    
    print(Highest_Features)
    return Highest_Features


# In[15]:


def main():
    HF = selectFeatures(csv = True)
    return HF


# In[16]:


Highest_Features = main()


# In[17]:


#print(pd.DataFrame(Highest_Features['target_group']))


# In[18]:


featureSet = pd.DataFrame(Highest_Features['target_group'])
    
# Save as csv file in DATACOLLECTION data folder (bc it's needed for encoding script)


# File path for this file
file_name = 'retailFeatureSet.csv'
thispath = Path().absolute()
OUTPUT_DIR = os.path.join(thispath, "Data", file_name)

# if the following line throws an error, use the line after to save in same folder
pd.DataFrame.to_csv(featureSet, path_or_buf=OUTPUT_DIR)


# In[19]:


import matplotlib.pyplot as plt
plt.plot(Highest_Features['MI_Values'].values)
plt.ylabel('MI Score')
plt.axis([0, 250, 0, 0.16])
plt.show()


# In[15]:


Highest_Features['MI_Values'].values


# In[12]:


Highest_Features

