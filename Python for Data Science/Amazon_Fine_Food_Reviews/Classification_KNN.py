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
from sklearn.decomposition import TruncatedSVD
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
filtered_data["Class_Labels"] = filtered_data["SentimentPolarity"].apply(lambda x : 1 if x == "Positive" else 0)

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

string = " "
not_stemmed_corpus = []
for review in sampled_data["Text"].values:
    filtered_sentence = []
    sentence = removeHtml(review)
    for word in sentence.split():
        for cleaned_words in removePunctuations(word).split():
            if cleaned_words.isalpha():
                filtered_sentence.append(cleaned_words.lower())
            else:
                continue
    string = " ".join(filtered_sentence)
    not_stemmed_corpus.append(string)
    
sampled_data["CleanedText"] = data_corpus
sampled_data["PreserveStopwords"] = not_stemmed_corpus
print("The number of positive and negative reviews after cleaning data")
print("{}".format(sampled_data["SentimentPolarity"].value_counts()))

connection_sqlobject = sqlite3.connect("sampled_dataset_all_reviews.sqlite")
c = connection_sqlobject.cursor()
connection_sqlobject.text_factory = str
sampled_data.to_sql("Reviews", connection_sqlobject, schema=None, if_exists="replace", index=True, 
                    index_label=None, chunksize=None, dtype=None)

###############################################################################
############################### KNN-algorithm #################################
###############################################################################

def knn_algorithm(X_train, y_train, X_test, y_test, vectorizationType):
    X_train = X_train
    y_train = y_train
    X_test = X_test
    y_test = y_test
    
    algorithms = ["brute"]
    for algo in algorithms:
        print("Starting Cross Validation steps for {} model using {} algorithm".format(
                vectorizationType, algo.upper()))
        k_values = list(np.arange(1,50,2))
        cross_val_scores = []
        
        if algo == "kd_tree":
            svd = TruncatedSVD(n_components=100)
            X_train = svd.fit_transform(X_train)
            X_test = svd.fit_transform(X_test)
        
        for k in k_values:
            knn_classifier = KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm=algo, 
                                                  p=2, metric='minkowski', n_jobs=-1)
            accuracies = cross_val_score(knn_classifier, X_train, y_train, cv=10, 
                                         scoring='accuracy', n_jobs=-1)
            cross_val_scores.append(accuracies.mean())
            
        errors = [1 - x for x in cross_val_scores]
        optimal_k = k_values[errors.index(min(errors))]
        print("The optimal number of neighbors for {} algorithm is : {}".format(algo.upper(), optimal_k))
        
        plt.figure(figsize=(15,10))
        plt.plot(k_values, errors, color="green", linestyle="dashed", linewidth=2, marker="o", 
                 markerfacecolor="red", markersize=10)
        for xy in zip(k_values, np.round(errors,3)):
            plt.annotate("(%s, %s)" % xy, xy=xy, textcoords="data")
        plt.title("Plot for Errors vs K Values")
        plt.xlabel("Number of Neighbors K for {} algorithm".format(algo.upper()))
        plt.ylabel("Error")
        plt.show()
        
        print("The error for each k value using {} algorithm: {}".format(algo.upper(), np.round(errors,3)))
        
        knn_classifier = KNeighborsClassifier(n_neighbors=optimal_k, weights="distance", algorithm="kd_tree",
                                              p=2, metric="minkowski", n_jobs=6)
        knn_classifier.fit(X_train, y_train)
        y_pred = knn_classifier.predict(X_test)
        
        print("The performance evaluation for {} mode".format(vectorizationType))
        
        test_accuracy = accuracy_score(y_test, y_pred, normalize=True) * 100
        points = accuracy_score(y_test, y_pred, normalize=False)
        print("The number of accurate predictions out of {} data points on unseen data for K = {} is {}".format(
                X_test.shape[0], optimal_k, points))
        print("Accuracy of the KNN model using {} algorithm on unseen data for K = {} is {}".format(
                algo.upper(), optimal_k, np.round(test_accuracy, 2)))
        
        # The precision, recall, F1 score
        print("Precision of the KNN model using {} algorithm on unseen data for K = {} is {}".format(
                algo.upper(), optimal_k, np.round(metrics.precision_score(y_test, y_pred), 4)))
        print("Recall of the KNN model using {} algorithm on unseen data for K = {} is {}".format(
                algo.upper(), optimal_k, np.round(metrics.recall_score(y_test, y_pred), 4)))
        print("F1 Score of the KNN model using {} algorithm on unseen data for K = {} is {}".format(
                algo.upper(), optimal_k, np.round(metrics.f1_score(y_test, y_pred), 4)))
        
        # Classification Report
        print("Classification report for {} model is using {} algorithm : ".format(
                vectorizationType, algo.upper()))
        print(metrics.classification_report(y_test, y_pred))
        
        # The confusion matrix for the running model
        print("The confusion matrix for {} model using {} algorithm".format(
                vectorizationType, algo.upper()))
        class_names = ["negative", "positive"]
        df_heatmap = pd.DataFrame(confusion_matrix(y_test, y_pred), index=class_names, columns=class_names)
        fig = plt.figure(figsize=(10,7))
        sns.heatmap(df_heatmap, annot=True, fmt="d")
        plt.ylabel("Predicted label", size=15)
        plt.xlabel("True label", size=15)
        plt.title("Confusion Matrix", size=20)
        plt.show()
        
        # Save information of model 
        info_model_KNN = [vectorizationType, optimal_k, np.round(np.array(errors).mean(),4), 
                          np.round(1-metrics.accuracy_score(y_test, y_pred),4), 
                          np.round(metrics.f1_score(y_test, y_pred),4)]
        with open("info_model_KNN.txt", "a") as filehandle:
            filehandle.writelines("%s " % iterator for iterator in info_model_KNN)
            filehandle.writelines("\n")
            
        del (X_train, y_train, X_test, y_test, knn_classifier)
        

###############################################################################
################### KNN-algorithm for Bag of Words ############################
###############################################################################

connection_sqlobject = sqlite3.connect("sampled_dataset_all_reviews.sqlite")
sampled_dataset = pd.read_sql_query(""" SELECT * FROM Reviews """, connection_sqlobject)

X = sampled_dataset["CleanedText"].values
y = sampled_dataset["Class_Labels"]

split = 20000
X_train = X[0:split,]
y_train = y[0:split,]
X_test = X[split:30000,]
y_test = y[split:30000,]

# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)

cv_object = CountVectorizer().fit(X_train)

print("Creating the BOW vectors using the cleaned corpus")
X_train_vectors = cv_object.transform(X_train)
X_test_vectors = cv_object.transform(X_test)

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler(with_mean=False)
scalar.fit(X_train_vectors)
X_train_vectors = scalar.transform(X_train_vectors)
X_test_vectors = scalar.transform(X_test_vectors)

del(sampled_dataset, X, y, X_train, X_test)

knn_algorithm(X_train_vectors, y_train, X_test_vectors, y_test, "Bag-of-Words")
