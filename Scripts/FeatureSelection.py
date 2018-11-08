
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[21]:


df = pd.read_csv("/Users/ayangedleh/Desktop/datacollection/Data/Data to Clean/retailarticles YTD (new)_merged.csv", encoding= "ISO-8859-1")
df.head()


# In[3]:


col = ['category','content', 'pub_date']
df = df[col]


# In[30]:


df = df[pd.notnull(df['content'])]
df.columns = ['category', 'content','pub_date']
df.head()


# In[23]:


df['category_id'] = df['Category'].factorize()[0]
category_id_df = df[['Category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Category']].values)
df.head()


# In[26]:


fig = plt.figure(figsize=(4,3))
df.groupby('Category').content.count().plot.bar(ylim=0)
plt.show()


# In[28]:


from sklearn.feature_extraction.text import TfidfVectorizer

vector = TfidfVectorizer(sublinear_tf=True, min_df=1, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = vector.fit_transform(df.content).toarray()
labels = df.category_id
features.shape


# In[29]:


from sklearn.feature_selection import chi2
import numpy as np
N = 50

for category, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(vector.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}':".format(category))
    print("  . Most correlated unigrams:\n {}".format('\n '.join(unigrams[-N:])))
    print("  . Most correlated bigrams:\n. {}".format('\n '.join(bigrams[-N:]))) 

