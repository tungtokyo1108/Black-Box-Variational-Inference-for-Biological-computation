#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:19:34 2019

Reference:
    - https://github.com/rsmahabir/Amazon-Fine-Food-Reviews-Analysis/blob/master/04.%20K-NN%20on%20Amazon%20Fine%20Food%20Reviews%20Dataset.ipynb
    - https://github.com/PushpendraSinghChauhan/Amazon-Fine-Food-Reviews/blob/master/Apply%20K-NN%20on%20Amazon%20reviews%20dataset%20.ipynb

@author: tungutokyo
"""

import sqlite3
import pandas as pd 
import numpy as np
import math
import nltk
import string 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle
import warnings
warnings.filterwarnings("ignore")
from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from collections import Counter
from matplotlib.colors import ListedColormap
import re

###############################################################################
########################## Loading the Amazon dataset #########################
###############################################################################

connection_sqlobject = sqlite3.connect("database.sqlite")
filtered_data = pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 """, 
                                  connection_sqlobject)

filtered_data["SentimentPolarity"] = filtered_data["Score"].apply(lambda x : "Positive" if x > 3 else "Negative")
filtered_data["Class_Labels"] = filtered_data["SentimentPolarity"].apply(lambda x : 1 if "Positive" else 0)

print ("The shape of the filtered matrix: {}".format(filtered_data.shape))
print ("The median score value: {}".format(filtered_data["Score"].mean()))
print ("The number of positive and negative reviews before removal of duplication data")
print ("{}".format(filtered_data["SentimentPolarity"].value_counts()))

###############################################################################
############################# Pre-processing dataset ##########################
###############################################################################

""" Removing duplication entries base on the past knowledge """
filtered_duplicates = filtered_data.drop_duplicates(subset={"UserId", "ProfileName", "Time", "Text"}, keep="first", inplace=False)
final_data = filtered_duplicates[filtered_duplicates.HelpfulnessNumerator <= 
                                 filtered_duplicates.HelpfulnessDenominator]
print("The shape of the data matrix after deduplication : {}".format(final_data.shape))
print("The median score value ofter deduplication: {}".format(final_data["Score"].mean()))
print("The number of positive and negative reviews after the removal of duplicate data.")
print("{}".format(final_data["SentimentPolarity"].value_counts()))

retained_per = (final_data["SentimentPolarity"].size*1.0)/(filtered_data["SentimentPolarity"].size*1.0)*100
removed_per = 100 - retained_per 
print("The percentage of data removed : {:.2f}".format(removed_per))
print("The percentage of data retained : {:.2f}".format(retained_per))

""" Create a sampled dataset keeping only wanted columns """
sampled_data = final_data.drop(labels=["Id", "ProductId", "UserId", "Score", "ProfileName", "Summary",
                                       "HelpfulnessNumerator", "HelpfulnessDenominator"], axis=1)
print("The shape of the sampled data: {}".format(sampled_data.shape))
sampled_data = sampled_data.sort_values("Time", axis=0, ascending=False, inplace=False, kind="quicksort",
                                        na_position="last")
sampled_data = sampled_data.reset_index()
sampled_data = sampled_data.drop(labels=["index"], axis=1)
sampled_data["SentimentPolarity"].value_counts().plot(kind="bar", color=["green", "blue"], alpha=0.5,
            title="Distribution of Positive and Negative reviews", figsize=(10,10))

""" Data Cleaning Stage """
def removeHtml(sentence):
    pattern = re.compile("<.*?>")
    cleaned_text = re.sub(pattern, ' ', sentence)
    return cleaned_text

def removePunctuations(sentence):
    cleaned_text = re.sub('[^a-zA-Z]', ' ', sentence)
    return cleaned_text
sno = SnowballStemmer(language="english")

default_stopwords = set(stopwords.words("english"))
remove_not = set(["not"])
custom_stopwords = default_stopwords - remove_not

count = 0
string = " "
data_corpus = []
all_positive_words = []
all_negative_words = []
stemed_word = ''
for review in sampled_data["Text"].values:
    filtered_sentence = []
    sentence=removeHtml(review)
    for word in sentence.split():
        for cleaned_words in removePunctuations(word).split():
            if ((cleaned_words.isalpha()) & (len(cleaned_words)>2)):
                if (cleaned_words.lower() not in custom_stopwords):
                    stemed_word = (sno.stem(cleaned_words.lower()))
                    filtered_sentence.append(stemed_word)
                    if (sampled_data["SentimentPolarity"].values)[count] == 'Positive':
                        all_positive_words.append(stemed_word)
                    if (sampled_data["SentimentPolarity"].values)[count] == "Negative":
                        all_negative_words.append(stemed_word)
                else:
                    continue
            else:
                continue
    string = " ".join(filtered_sentence)
    data_corpus.append(string)
    count += 1
print("The length of the data corpus is : {}".format(len(data_corpus)))













































