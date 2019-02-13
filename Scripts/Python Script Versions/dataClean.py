def DataClean(rawDf):
    #import FeatureEncoding
    import os
    from os import listdir
    from os.path import isfile, join
    from pathlib import Path
    import pandas as pd
    import re, string

    #remove blanks (NaN)
    df = rawDf.dropna(subset = ['content', 'title']) 

    #remove blocked articles without content
    df = df[df.content.str.contains('Your usage has been flagged', case=False) == False]
    df = df[df.content.str.contains('To continue, please click the box', case=False) == False]
    
    # vidoes/ads/commentary
    df = df[df.description.str.contains('The "Fast Money" traders share', case=False) == False]
    df = df[df.description.str.contains('stuff we think you', case=False) == False]
    df = df[df.description.str.contains('best deals on', case=False) == False]
    
    #remove transcripts
    df = df[df.title.str.contains('transcript', case=False) == False]
    
    #remove cramer
    df = df[df.title.str.contains('cramer', case=False) == False]
    
    #remove duplicates
    # by self-identified repeat
    df = df[df.title.str.contains('rpt', case=False) == False]
    # by title
    df = df.drop_duplicates(subset=['title'], keep='first')
    # by content
    df = df.drop_duplicates(subset=['content'], keep='first')
    # by decription
    df = df.drop_duplicates(subset=['description'], keep='first')
    
    #remove punctuation, keep orig content
    df['origContent'] = df['content']
    pattern = re.compile('[^0-9a-zA-Z ]+')
    content= map(lambda x: pattern.sub(' ', x), df['content'])
    df['content']=list(content)
    
    return df