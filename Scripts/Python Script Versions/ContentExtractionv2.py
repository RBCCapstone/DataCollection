
# coding: utf-8

# In[ ]:


# imports
import os
import sys
from pathlib import Path

#Counter
from collections import Counter
from collections import OrderedDict
import pprint as pp

# Data packages
import math
import pandas as pd
import numpy as np

#Operation
import operator

#Natural Language Processing Packages
import re
import nltk

from nltk import tokenize
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import nltk
#nltk.download('brown')
from nltk.corpus import brown


# In[ ]:


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


# In[ ]:


"""
    Utility functions for filtering content
    originally written by: vipul-sharma20
    modifications made by: jadekhiev
"""
def getWords(sentence):
    """
    Extracts words/tokens from a sentence
    :param sentence: (str) sentence
    :returns: list of tokens
    """
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

def countWords(wordList):
    return dict(Counter(wordList))


# In[ ]:


class ExtractContext(content):
"""
Script to extract important topics from content
originally written by: vipul-sharma20
modifications made by: jadekhiev
"""
    def __init__(self):
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

    def get_info(self, content):
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


    def re_tag(self, tagged):
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


# In[ ]:


class SampleExtract():
"""
Sample context extractor for random article
"""
    def __init__(self):
        # import articles
        articleDf = importData()
        # create new column with a cleaned up date so that it is possible to filter easily
        articleDf['dateCleaned'] = pd.to_datetime(articleDf['date'].str[0:10])
        # select random article
        artNum = np.rand(0,len(articlesDf.index))
        content = articleDf['content'].iloc[artNum]
        context = ExtractContext(content)

    def displayResults(self, context):
        #context = [term for term in context]# if not (''in term ==True) and len(term.split()) > 1]
        wordCount = countWords(context)
        print("title: " + articleDf['title'].iloc[artNum])
        print("description: " + articleDf['description'].iloc[artNum])
        print("url: " + articleDf['url'].iloc[artNum])
        print("content: " + content)
        print("context:")
        print([term for term, count in wordCount.items()])


# In[ ]:


class DateRangeExtract(startDate, endDate):
"""
Context extractor for specific date range
"""
    def __init__(self):
        # import articles
        articleDf = importData()
        # create new column with a cleaned up date so that it is possible to filter easily
        articleDf['dateCleaned'] = pd.to_datetime(articleDf['date'].str[0:10])
        # extract date range
        dateFilteredDf = articleDf[articleDf['dateCleaned'].isin(pd.date_range(startDate, endDate))]
    
    def contextExtract(self, dateFilteredDf):
        

