import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
import pandas as pd
import re, string

def DataClean(articleDf):
    #remove blanks (NaN)
    articleDf = articleDf.dropna(subset = ['content', 'title']) 

    #remove blocked articles without content
    articleDf = articleDf[articleDf.content.str.contains('Your usage has been flagged', case=False) == False]
    articleDf = articleDf[articleDf.content.str.contains('To continue, please click the box', case=False) == False]

    # vidoes/ads/commentary
    articleDf = articleDf[articleDf.description.str.contains('The "Fast Money" traders share their first moves for the market open.', case=False) == False]
    articleDf = articleDf[articleDf.description.str.contains('stuff we think you', case=False) == False]
    articleDf = articleDf[articleDf.title.str.contains('opinion', case=False) == False]
    articleDf = articleDf[articleDf.content.str.contains('awesome deals', case=False) == False]
    
    #remove transcripts
    articleDf = articleDf[articleDf.title.str.contains('transcript', case=False) == False]

    #remove cramer
    articleDf = articleDf[articleDf.title.str.contains('cramer', case=False) == False]

    #remove articles with less than words which is the lower end of the boxplot
    articleDf = articleDf[articleDf['content'].str.split().str.len() > 300]

    #remove duplicates
    # by self-identified repeat
    articleDf = articleDf[articleDf.title.str.contains('rpt', case=False) == False]
    # by title
    articleDf = articleDf.drop_duplicates(subset=['title'], keep='first')
    # by content
    articleDf = articleDf.drop_duplicates(subset=['content'], keep='first')
    # by decription
    articleDf = articleDf.drop_duplicates(subset=['description'], keep='first')

    #clean content, maintain punctuation
    articleDf['origContent'] = articleDf['content']  
    # remove 
    #Replace new lines with spaces
    pat_amp = re.compile('amp;')
    articleDf['cleanContent'] = list(map(lambda x: pat_amp.sub('', x), articleDf['content']))
    #Replace new lines with spaces
    pat_url = re.compile('[a-z]+?[.]?[a-z]+?[.]?[a-z]+[.]?[\/\/]\S+')
    articleDf['cleanContent'] = list(map(lambda x: pat_url.sub('', x), articleDf['cleanContent']))
    pat_https = re.compile('https://')
    articleDf['cleanContent'] = list(map(lambda x: pat_https.sub('', x), articleDf['cleanContent']))

    articleDf = articleDf.reset_index(drop=True)
    for i in articleDf.index:
        article = articleDf['cleanContent'].iloc[i].split('\r\n')

        # remove lines with no period
        article[:] = [sentence for sentence in article if '.' in sentence]
        # remove lines with less than 5 words
        article[:] = [sentence for sentence in article if len(sentence.split())>5]

        # remove lines with terms that are associated with promotions or credits
        article[:] = [sentence for sentence in article if not('get breaking news' in sentence.lower())]
        article[:] = [sentence for sentence in article if not('click here' in sentence.lower())]
        article[:] = [sentence for sentence in article if not('write to' in sentence.lower())]
        article[:] = [sentence for sentence in article if not('subscribe' in sentence.lower())]
        article[:] = [sentence for sentence in article if not('read more' in sentence.lower())]
        article[:] = [sentence for sentence in article if not('read or share' in sentence.lower())]
        article[:] = [sentence for sentence in article if not('reporting by' in sentence.lower())]
        article[:] = [sentence for sentence in article if not('twitter, instagram' in sentence.lower())]
        article[:] = [sentence for sentence in article if not('Photo' in sentence)]
        article[:] = [sentence for sentence in article if not('copyright' in sentence.lower())]
        article[:] = [sentence for sentence in article if not('©' in sentence.lower())]
        article[:] = [sentence for sentence in article if not('get our daily' in sentence.lower())]

        #print(article)
        articleDf.at[i,'cleanContent']=' '.join(article)
    
    # remove punctuation for feature selection
    pat_punctuation = re.compile('[^0-9a-zA-Z\s]+')
    articleDf['content'] = list(map(lambda x: pat_punctuation.sub(' ', x), articleDf['cleanContent']))
    
    return articleDf