# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 14:32:36 2018

@author: Padmanie
"""

import pandas as pd

num_words = 0
checkwords = ['no','to', 'in', 'for']
for a in checkwords:
    print(a)

with open('articles.csv', 'r') as f:
    for line in f:
        line = line.split(',')
        headline = line[1]
        no = line[0]
        print(headline)
        words = headline.split()
        num_words = len(words)

        dfln = [line[0],]
        print(num_words)
        
df = pd.DataFrame()
df.columns=['no','to', 'in', 'for']


        

