#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 17:55:43 2018

@author: Ayan Gedleh and Jessie Diep
"""
#Works on base environment. Must pip install pandas and matplotlib in any newly created environments (ex. rbc)
import pandas as pd
import matplotlib.pyplot as plt

#Replace with location of data file
df = pd.read_csv("/Users/Legendary/Documents/DataCollection/articles_testingdata1.csv")
df.head()

col = ['Class', 'Description']
df = df[col]
df = df[pd.notnull(df['Description'])]
df.columns = ['Class', 'Description']

df['category_id'] = df['Class'].factorize()[0]
category_id_df = df[['Class', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Class']].values)
df.head()
 
 
fig = plt.figure(figsize=(4,3))
df.groupby('Class').Description.count().plot.bar(ylim=0)
plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer
 
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=1, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.Description).toarray()
labels = df.category_id
features.shape

from sklearn.feature_selection import chi2
import numpy as np
N = 4
for Class, category_id in sorted(category_to_id.items()):
   features_chi2 = chi2(features, labels == category_id)
   indices = np.argsort(features_chi2[0])
   feature_names = np.array(tfidf.get_feature_names())[indices]
   unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
   bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
   print("# '{}':".format(Class))
   print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
   print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))