#Script to extract important topics from content
#originally written by: vipul-sharma20
#modifications made by: jadekhiev

# imports
import os
import sys
from pathlib import Path

# imports required utility functions
import string
from collections import Counter

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
nltk.download('brown')
from nltk.corpus import brown

#Progress bar
from tqdm import tqdm

# Import articles
def importData(filename):
    """
    Import data into df
    """
    #Import Labelled Data
    DATA_DIR = "Data"
    thispath = Path().absolute()
    ARTICLES = os.path.join(DATA_DIR, filename)
    
    df = pd.read_excel(ARTICLES)

    try:
        df.head()
    except:
        pass
    
    return df

# PoS Tagger and CFG Definitions
# train tagger with browns news corpus
train = brown.tagged_sents(categories='news')

# custom regex tagging
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

# PoS Browns Corpus Tagging: https://en.wikipedia.org/wiki/Brown_Corpus
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
cfg['NNI+RP'] = 'NNI' # collects terms like "heats up" -- RP = adverb particle
cfg['RB+NN'] = 'NNP'# combination for monetary movement e.g. quarterly[RB] profit[NN] fell [VBD] -- RB = adverb
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

# Utility functions for context extraction
def getWords(sentence):
    stopwords = [
        # months
        "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
        # symbols that don't separate a sentence
        '$','“','”','’','—',
        # specific article terms that are useless
        "read", "share", "file", "'s","i", "photo", "percent","s", "t", "inc.", "corp", "group", "inc", "corp.", "source", "bloomberg", "cnbc",
        # useless pronouns
        "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "co.", "inc.",
        # etc
        "the", "a", "of", "have", "has", "had", "having"
        #"am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "while", "of", "at", "by", "for", "about", "into", "through", "during", "before", "after", "to", "from", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "just", "don", "now"
        ]

    words = word_tokenize(sentence)
    words = [word for word in words if word.lower() not in stopwords and len(word)>2]

    return words

def countWords(wordList):
    return dict(Counter(wordList))

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

# extract all unigrams based on all words pulled from context extraction
def unigramBreakdown(fullContext):
    # to be used as frequency count
    stopwords = ["myself", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "him", "his", "himself", "she", "her", "hers", "herself", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "are", "was", "were", "been", "being", "have", "has", "had", "having", "does", "did", "doing",  "the", "and", "but", "if", "or", "because", "until", "while", "for", "with", "about", "into", "through", "during", "before", "after", "from", "down", "out", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "nor", "not", "only", "own", "same", "than", "too", "very", "can", "will", "just", "don", "should", "now", "past", "year", "month", "day"]   
    
    # separates each word for each article => list of list
    articleUnigrams = []
    for term in fullContext:
        articleUnigrams.extend(term.split())
    
    # remove stop words and punctuation
    translator = str.maketrans('', '', string.punctuation)
    unigrams = [term.lower().translate(translator) for term in articleUnigrams if term.lower() not in stopwords and len(term)>2]
    # count frequency of terms
    # unigrams = countWords(unigrams)
    
    return unigrams

# extracts unigrams AND bigrams pulled by context extraction
def bigramBreakdown(fullContext):
    bigrams = []
    
    # remove stop words and punctuation
    translator = str.maketrans('', '', string.punctuation)
    bigrams.extend([term.lower().translate(translator) for term in fullContext if len(term.split()) < 3])
    
    return bigrams

# did this because I couldn't good way to write the switcher to switch to a non-function
def ngramDummy(fullContext):
    return fullContext

# PMI For Tag Ranking
# return binary representation of article in terms of all keyphrases pulled
def dfTransform(df, term_column):
    # df is the article df ;
    keyterms = []
    for article in df[term_column].values:
        keyterms.extend([word.lstrip() for word in (article.split(','))])
    keyterms = set(keyterms) # deduplicate terms by casting as set
    
    # for each article and each keyword: give 1 if keyword in article and 0 if not
    encodedArticle = []
    for i in tqdm(df.index):
        articleTerms = ([word.lstrip() for word in (df[term_column].iloc[i].split(','))])
        encodedArticle.append([1 if word in articleTerms else 0 for word in keyterms])
    
    # set up dataframe
    binEncDf = pd.DataFrame(encodedArticle)
    # use keywords as columns
    binEncDf.columns = keyterms
    # keep article_id and prediction from original table
    df = df.rename(columns={'prediction': 'mkt_moving'}) # changed it from prediction because that was also a keyterm
    binEncDf = df[['article_id','mkt_moving']].join(binEncDf)
    
    return binEncDf

# Simple example of getting pairwise mutual information of a term
def pmiCal(df, x, label_column='mkt_moving'):
    pmilist=[]
    for i in [0,1]:
        for j in [0,1]:
            px = sum(df[label_column]==i)/len(df)
            py = sum(df[x]==j)/len(df)
            pxy = len(df[(df[label_column]==i) & (df[x]==j)])/len(df)
            if pxy==0:#Log 0 cannot happen
                pmi = math.log((pxy+0.0001)/(px*py+0.0001))
            else:
                pmi = math.log(pxy/(px*py+0.0001))
            pmilist.append([i]+[j]+[px]+[py]+[pxy]+[pmi])
    pmiDf = pd.DataFrame(pmilist)
    pmiDf.columns = ['x','y','px','py','pxy','pmi']
    
    return pmiDf

def pmiIndivCal(df,x,gt, label_column='mkt_moving'):
    px = sum(df[label_column]==gt)/len(df)
    py = sum(df[x]==1)/len(df)
    pxy = len(df[(df[label_column]==gt) & (df[x]==1)])/len(df)
    if pxy==0:#Log 0 cannot happen
        pmi = math.log((pxy+0.0001)/(px*py+0.0001))
    else:
        pmi = math.log(pxy/(px*py))
    
    return pmi

# calculate all the pmi for all tags across all articles and store top 5 tags for each article in df
def pmiForAllCal(artDf, binaryEncDf, term_column, label_column='mkt_moving'): 
    
    for i in tqdm(artDf.index): # for all articles
        terms = set(([word.lstrip() for word in (artDf[term_column].iloc[i].split(','))]))
        pmineglist = []

        for word in terms:
            pmineglist.append([word]+[pmiIndivCal(binaryEncDf,word,0,label_column)])
        
        pmineglist = pd.DataFrame(pmineglist)
        pmineglist.columns = ['word','pmi']
        artDf.at[i,'tags_top_5'] = (',').join(word for word in pmineglist.sort_values(by='pmi', ascending=True).head(5)['word'])   
    return artDf

# Functions to run extraction and rank tags

# Tag ranking using PMI
def calculatePMI(artDf, termType):
    # use PMI to calculate top 10 terms that should be displayed for each article    
    # get binary encoding of articles represented as uni- and bigrams
    binaryEncDf = dfTransform(artDf, termType)
    articleDf_ranked = pmiForAllCal(artDf, binaryEncDf, termType)
    
    return articleDf_ranked, binaryEncDf

# find most popular keyterms mentioned in news
def frequencyCounter(binEncDf):
    # sum each column of binary encoded articles
    # output should be a dataframe with: word | 3 of articles mentioning word
    freqDf = binEncDf.drop('article_id', axis=1).sum(axis=0, skipna=True).sort_values(ascending=False).to_frame().reset_index()
    freqDf.columns = ['word','freq_articles']
    
    return freqDf

# Retrieve context
def retrieveContext(articleDB, termType='bigrams'):
    # import classified articles
    articleDf = articleDB
    
    breakdown = {
        'ngrams': ngramDummy, # store n-grams pulled from context extraction
        'bigrams': bigramBreakdown, # store bigrams and unigrams captured by context extraction
        'unigrams': unigramBreakdown # store unigrams captured by separating all terms pulled by context extraction
        }
    
    for i in articleDf.index:
        # get context for articles
        keyterms = get_info(articleDf['content'].iloc[i])  
        articleDf.at[i, 'tags'] = ', '.join(breakdown[termType](keyterms))    
    
    # returns article Df with new column for top tags
    articleDf, binaryEncDf = calculatePMI(articleDf, 'tags')
    
    # returns most popular terms mentioned across all articles
    trendingTermsDf = frequencyCounter(binaryEncDf)
    
    #Save as excel file (better because weird characters encoded correctly)
    #DATA_DIR = "Data"
    #OUTPUT_DIR = os.path.join(DATA_DIR, "results_context.xlsx")
    #writer = pd.ExcelWriter(OUTPUT_DIR)
    #articleDf.to_excel(writer,'Sheet1')
    #writer.save()
    
    #Save as excel file (better because weird characters encoded correctly)
    #DATA_DIR = "Data"
    #OUTPUT_DIR = os.path.join(DATA_DIR, "trending_terms.xlsx")
    #writer = pd.ExcelWriter(OUTPUT_DIR)
    #trendingTermsDf.to_excel(writer,'Sheet1')
    #writer.save()

    return articleDf, trendingTermsDf