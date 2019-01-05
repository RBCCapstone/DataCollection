import pandas as pd
import numpy as np
import scipy.stats as stats
import sklearn
import random
import os
from pathlib import Path
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report 
import pickle


def LoadData(filename):
    DATA_DIR = "Data"
    ENCODING_DIR = os.path.join(DATA_DIR, filename)
    data = pd.read_csv(ENCODING_DIR)
    return data

def runLogReg(filename):
    X = LoadData(filename) # This would be named to whatever today's binEncoding file is called
    artID = X['article_id']
    X = X.drop(columns=['article_id', 'Unnamed: 0'])

    print(X.head())
    classifier = pickle.load(open("ourClassifier.p", "rb"))
    
    y_predict = classifier.predict(X)
    # get log scores for train and test set
    y_log_proba = classifier.predict_log_proba(X)    
    
    #tie the scores and predictions to specific articles
    scores = pd.DataFrame(data=y_log_proba)
    scores['article_id'] = artID.values
    scores['prediction'] = y_predict
    
    thispath = Path().absolute()
    OUTPUT_DIR = os.path.join(thispath, "Data", "results_"+filename)
    pd.DataFrame.to_csv(scores, path_or_buf=OUTPUT_DIR)