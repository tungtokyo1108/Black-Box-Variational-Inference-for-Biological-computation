#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 23:15:31 2019

@author: tungutokyo
"""

import sqlite3
import pandas as pd
import numpy as np 
import nltk
import string 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Word2Vec, KeyedVectors
import pickle
import warnings
from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from collections import Counter
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import math
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

###############################################################################
########################## Loading the Amazon dataset #########################
###############################################################################

connection_sqlobject = sqlite3.connect("database.sqlite")
filtered_data = pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 """, connection_sqlobject)
filtered_data["SentimentPolarity"] = filtered_data["Score"].apply(lambda x : "Positive" if x > 3 else "Negative")
filtered_data["Class_labels"] = filtered_data["SentimentPolarity"].apply(lambda x : 1 if x == "Positive" else 0)

print("The shape of the filtered data: {}".format(filtered_data.shape))
print("The median score value: {}".format(filtered_data["Score"].mean()))
print("The number of positive and negative before the removal of duplicate data.")
print("{}".format(filtered_data["SentimentPolarity"].value_counts()))

filtered_duplicates = filtered_data.drop_duplicates(subset={"UserId", "ProfileName", "Time", "Text"}, 
                                                    keep="first", inplace=False)

















































