#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 17:55:43 2018

@author: Ayan Gedleh and Jessie Diep
@Code from: Susan Li - https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f
"""
#Works on base environment. Must pip install pandas and matplotlib in any newly created environments (ex. rbc)
import pandas as pd
import matplotlib.pyplot as plt

#Replace with location of data file
df = pd.read_csv("/Users/Legendary/Documents/DataCollection/articles_testingdata1.csv", encoding = "ISO-8859-1")
df.head()

#Select required columns from data
col = ['Class', 'Description', 'PublishedAt']
df = df[col]
df = df[pd.notnull(df['Description'])]
df.columns = ['Class', 'Description','PublishedAt']

df['category_id'] = df['Class'].factorize()[0]
category_id_df = df[['Class', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Class']].values)
df.head()
 
#Plot data, it shows that currently we have imbalanced classes (data for retail >)
fig = plt.figure(figsize=(8,6))
fig.suptitle('Articles per Class', fontsize=18, fontweight='bold')
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_title('Articles per Class')

ax.set_xlabel('Class', fontsize=14)
ax.set_ylabel('No. of Articles', fontsize=14)

df.groupby('Class').Description.count().plot.bar(ylim=0)
plt.show()

#Use Bag of Words to extract tokens
from sklearn.feature_extraction.text import TfidfVectorizer
 
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=1, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.Description).toarray()
labels = df.category_id
features.shape

#Use Chi2 Correlation for feature selection
from sklearn.feature_selection import chi2
import numpy as np
N = 4
for Class, category_id in sorted(category_to_id.items()):
  
   features_chi2 = chi2(features, labels == category_id)
   
   indices = np.argsort(features_chi2[0])

   feature_names = np.array(tfidf.get_feature_names())[indices]
   #print(feature_names)
   
   unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
   bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
   print("# '{}':".format(Class))
   print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
   print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))