
# coding: utf-8

# # Content Extraction
# 
# The purpose of this code is to highlight key terms for articles that are determined to be "impactful". 
# 
# This step would be done after the article has been determined "impactful".
# 
# Resources:
# http://vipulsharma20.blogspot.com/2017/03/sharingan-newspaper-text-and-context.html
# https://github.com/vipul-sharma20/sharingan/blob/master/sharingan/summrizer/context.py
# http://nltk.sourceforge.net/doc/en/ch03.html

# In[1]:


import os
import sys
from pathlib import Path

# Data packages
import math
import pandas as pd
import numpy as np

#Progress bar
from tqdm import tqdm

#Counter
from collections import Counter

#Operation
import operator

#Natural Language Processing Packages
import re
import nltk

## Download Resources
nltk.download("vader_lexicon")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")

from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tag import PerceptronTagger
from nltk.data import find

## Machine Learning
import sklearn
import sklearn.metrics as metrics
from sklearn.feature_selection import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import datasets

from collections import OrderedDict
import pprint as pp


# In[2]:


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


# In[48]:


#def SelectFeaturesNP():
articleDf = importData()
# creating a new column with a cleaned up date so that it is possible to filter easily
articleDf['dateCleaned'] = pd.to_datetime(articleDf['date'].str[0:10])
print(articleDf.shape)


# In[92]:


articleDf.head(100)


# In[5]:


# Part of Speech Tagging
# Google: https://en.wikipedia.org/wiki/Part-of-speech_tagging
tagger = PerceptronTagger()
pos_tag = tagger.tag


# In[4]:


# This grammar is described in the paper by S. N. Kim,
# T. Baldwin, and M.-Y. Kan.
# Evaluating n-gram based evaluation metrics for automatic
# keyphrase extraction.
# Technical report, University of Melbourne, Melbourne 2010.
grammar = r"""
    NBAR:
        {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
        
    NP:
        {<NBAR>}
        {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
"""


# In[ ]:


# Create phrase tree
chunker = nltk.RegexpParser(grammar)


# In[36]:


# Noun Phrase Extraction Support Functions
#from nltk.corpus import stopwords
#stopwords = stopwords.words('english')
stopwords = [ 
    # months
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "Decemeber",
    # uninformative pronouns
    "myself", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "him", "his", "himself", "she", "her", "hers", "herself", "its", "itself", "they", "them", "their", "theirs", "themselves", 
    # other useless stop words -- decided to keep words that are implicit of future (e.g. can, should, will etc.)
    "what", "which", "who", "whom", "this", "that", "these", "those", "are", "was", "were", "been", "being", "have", "has", "had", "having", "does", "did", "doing", "the", "and", "but", "because", "while", "for", "with", "about", "into", "through", "during", "before", "after", "from", "down", "out", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "nor", "not", "only", "own", "same", "than", "too", "very", "just", "don", "now"]
lemmatizer = nltk.WordNetLemmatizer()
stemmer = nltk.stem.porter.PorterStemmer()

# generator, generate leaves one by one
def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter = lambda t: t.label()=='NP' or t.label()=='JJ' or t.label()=='RB'):
        yield subtree.leaves()

# stemming, lematizing, lower case... 
def normalise(word):
    """Normalises words to lowercase and stems and lemmatizes it."""
    word = word.lower()
    word = stemmer.stem(word)
    word = lemmatizer.lemmatize(word)
    return word

# stop-words and length control
def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword."""
    accepted = bool(2 <= len(word) <= 40
        and word.lower() not in stopwords)
    return accepted

# generator, create item once a time
def get_terms(tree):
    for leaf in leaves(tree):
        term = [normalise(w) for w,t in leaf if acceptable_word(w) ]
        # Phrase only
        if len(term)>1:
            yield term
            
# Flatten phrase lists to get tokens for analysis
def flatten(npTokenList):
    finalList =[]
    for phrase in npTokenList:
        token = ''
        for word in phrase:
            token += word + ' '
        finalList.append(token.rstrip())
    return finalList


# In[6]:


"""
Utility functions for filtering content
originally written by: vipul-sharma20
modifications made by: jadekhiev
"""
from nltk import tokenize
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

stopwords = [
    # months
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "decemeber",
    # symbols that don't separate a sentence
    '$','“','”','’','—',
    # specific article terms that are useless
    "read", "share", "file", "'s","i", "photo", "percent","s", "t", "inc.", "corp", "group", "inc", "corp.", "source", "bloomberg", "CNBC",
    # useless pronouns
    "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "co.", "inc.",
    # etc
    "the", "a", "of", "have", "has", "had", "having"
    #"am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "while", "of", "at", "by", "for", "about", "into", "through", "during", "before", "after", "to", "from", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "just", "don", "now"
    ]


def getWords(sentence):
    """
    Extracts words/tokens from a sentence
    :param sentence: (str) sentence
    :returns: list of tokens
    """
    words = word_tokenize(sentence)
    words = ([word for word in words if word.lower() not in stopwords])
    #print(words)
    return words


def getParagraphs(content):
    """
    Exctracts paragraphs from the the text content
    :param content: (str) text content
    :returns: list of paragraphs
    """
    paraList = content.split('\n\n')
    return paraList


def getSentences(paragraph):
    """
    Extracts sentences from a paragraph
    :param paragraph: (str) paragraph text
    :returns: list of sentences
    """
    indexed = {}
    sentenceList = tokenize.sent_tokenize(paragraph)
    for i, s in enumerate(sentenceList):
        indexed[i] = s
    return sentenceList, indexed


# In[69]:


# -*- coding: utf-8 -*-

"""
Script to extract important topics from content
originally written by: vipul-sharma20
modifications made by: jadekhiev
"""

import nltk
#nltk.download('brown')
from nltk.corpus import brown

train = brown.tagged_sents(categories='news')

# backoff regex tagging
regex_tag = nltk.RegexpTagger([
     #(r'[$][0-9]+\s[MmBbTt]\S+','DV'), #dollar value 
     (r'^[-\:]?[0-9]+(.[0-9]+)?$', 'CD'),
     (r'.*able$', 'JJ'),
     (r'^[A-Z].*$', 'NNP'),
     (r'.*ly$', 'RB'),
     (r'.*s$', 'NNS'),
     (r'.*ing$', 'VBG'),
     (r'.*ed$', 'VBD'),
     (r'.[\/\/]\S+', 'URL'), #URL / useless
     (r'.*', 'NN')
])

unigram_tag = nltk.UnigramTagger(train, backoff=regex_tag)
bigram_tag = nltk.BigramTagger(train, backoff=unigram_tag)
trigram_tag = nltk.TrigramTagger(train, backoff=bigram_tag)

# custom defined Context Free Grammar (CFG) by vipul
cfg = dict()
cfg['NNP+NNP'] = 'NNP'
cfg['NN+NN'] = 'NNI'
cfg['NNP+NNI'] = 'NNI'
cfg['NNI+NN'] = 'NNI'
cfg['NNI+NNI'] = 'NNI'
cfg['NNI+NNP'] = 'NNI'
cfg['JJ+JJ'] = 'JJ'
cfg['JJ+NN'] = 'NNI'
cfg['CD+CD'] = 'CD'
cfg['NPI+NNP'] = 'NNP' # this is specific for collecting terms with the word deal
cfg['NNI+RP'] = 'NNI' # collects terms like "heats up"
cfg['RB+NN'] = 'NNP'# combination for monetary movement e.g. quarterly[RB] profit[NN] fell [VBD]
cfg['NNP+VBD'] = 'VPI' #VBP = a verb phrase
cfg['MD+VB'] = 'VPI' # collects terms like "will lose" (verb phrase incomplete)
cfg['MD+NN'] = 'VPI' # collects terms like "will soar" (verb phrase incomplete)
cfg['VPI+NN'] = 'VP' # collects terms like "will lose ground"
cfg['NNI+VP'] = 'VP' # collects terms like "index will soar"
cfg['NN+VPI'] = 'VP' # collects terms like "index will soar"
cfg['NNP+VPI'] = 'VP' # collects terms like "index will soar"
cfg['VPI+TO'] = 'VPI' # collect past participle verbs with to e.g. pledged to
cfg['VBN+TO'] = 'VBN' # collect past participle verbs with to e.g. pledged to
cfg['VBN+NN'] = 'VP' # collects terms like "pledged to adapt"

def get_info(content):
    words = getWords(content)
    temp_tags = trigram_tag.tag(words)
    tags = re_tag(temp_tags)
    normalized = True
    while normalized:
        normalized = False
        #print("len tag: ", len(tags))
        #pp.pprint(DictGroupBy(tags))
        for i in range(0, len(tags) - 1):
            #print("i: ", i)
            tagged1 = tags[i]
            if i+1 >= len(tags) - 1:
                break
            tagged2 = tags[i+1]
            
            # when word = deal and next word is tagged IN (with, for, etc.) 
            if tagged1[0]=='deal' and tagged2[1]=='IN':
                tags.pop(i)
                tags.pop(i)
                re_tagged = tagged1[0] + ' ' + tagged2[0]
                pos='NPI'
                tags.insert(i, (re_tagged, pos))
                normalized = True
            
            else: 
                key = tagged1[1] + '+' + tagged2[1]
                pos = cfg.get(key)       
                if pos:
                    tags.pop(i)
                    tags.pop(i)
                    re_tagged = tagged1[0] + ' ' + tagged2[0]
                    tags.insert(i, (re_tagged, pos))
                    normalized = True

    final_context = []
    for tag in tags:
        if tag[1] == 'NNP' or tag[1] == 'NNI' or tag[1] == 'VP':
            final_context.append(tag[0])
    return final_context


def re_tag(tagged):
    new_tagged = []
    for tag in tagged:
        if tag[1] == 'NP' or tag[1] == 'NP-TL':
            new_tagged.append((tag[0], 'NNP'))
        elif tag[1][-3:] == '-TL':
            new_tagged.append((tag[0], tag[1][:-3]))
        elif tag[1][-1:] == 'S':
            new_tagged.append((tag[0], tag[1][:-1]))
        else:
            new_tagged.append((tag[0], tag[1]))
    return new_tagged


# In[60]:


#artNum = 2
#content = articleDf['content'].iloc[artNum]
dateFilteredDf = articleDf[articleDf['dateCleaned'].isin(pd.date_range('2017-11-15', '2017-11-15'))]


# In[65]:


content = []
for index, row in dateFilteredDf.iterrows():
    content.append(row['content'])


# In[66]:


content


# In[73]:


context=[]
for contents in content:
    context.extend(get_info(contents))


# In[74]:


context


# In[75]:


wordCount = countWords(context)


# In[76]:


wordCount


# In[10]:


def countWords(wordList):
    return dict(Counter(wordList))


# In[11]:


def DictGroupBy(input):
    res = OrderedDict()
    for v, k in input:
        if k in res: res[k].append(v)
        else: res[k] = [v]
    return res


# In[79]:


print(wordCount)


# In[24]:


# partial stop words list used
context = [term for term in context]# if not (''in term ==True) and len(term.split()) > 1]
wordCount = countWords(context)
print("title: " + articleDf['title'].iloc[artNum])
print("description: " + articleDf['description'].iloc[artNum])
print("url: " + articleDf['url'].iloc[artNum])
print("content: " + content)
print("context:")
print([term for term, count in wordCount.items()])


# In[200]:


wordCount


# # Consider:
# 
# ## useful things:
# * things that happened = past tense verbs (VBD)
# * things currently happening = VBG
# * things that could potentially happen = modal auxiliary (can, should, will) (MD)
# * prepositions such as with (IN)
# 
# ## useless things:
# * names of writers, news sources, photographers
# * URLs
