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
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV

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
final_data = filtered_duplicates[filtered_duplicates.HelpfulnessNumerator <= 
                                 filtered_duplicates.HelpfulnessDenominator]
print("The shape of the deduplicated data: {}".format(final_data.shape))
print("The median score value after deduplication: {}".format(final_data["Score"].mean()))
print("The number of positive and negative after the removal of duplicate data.")
print("{}".format(final_data["SentimentPolarity"].value_counts()))

###############################################################################
############################# Pre-processing dataset ##########################
###############################################################################

import re
from nltk.stem.snowball import SnowballStemmer
sno = SnowballStemmer(language="english")

def removeHtml(sentence):
    pattern = re.compile("<.*?>")
    cleaned_text = re.sub(pattern, ' ', sentence)
    return cleaned_text

def removePunctuation(sentence):
    cleaned_text = re.sub("[^a-zA-Z]", " ", sentence)
    return cleaned_text

sampled_dataset = final_data.drop(labels=["Id", "ProductId", "UserId", "Score", "ProfileName", 
                    "HelpfulnessNumerator", "HelpfulnessDenominator", "Summary"], axis=1)
print("The shape of the sampled dataset after dropping unwanted columns: {}".format(sampled_dataset.shape))

sampled_dataset = sampled_dataset.sort_values("Time", axis=0, ascending=False, inplace=False, 
                                              kind="quicksort", na_position="last")
sampled_dataset = sampled_dataset.reset_index()
sampled_dataset = sampled_dataset.drop(labels=["index"], axis=1)
sampled_dataset["SentimentPolarity"].value_counts().plot(kind="bar", color=["green", "blue"], alpha=0.5,
               title="Distribution of Positive and Negative reviews after deduplication", figsize=(10,10))

default_stopwords = set(stopwords.words("english"))
remove_not = set(["not"])
custom_stopwords = default_stopwords - remove_not

count = 0
string = " "
data_corpus = []
all_positive_words = []
all_negative_words = []
stemed_word = " "
for review in sampled_dataset["Text"].values:
    filtered_sentence = []
    sentence = removeHtml(review)
    for word in sentence.split():
        for cleaned_words in removePunctuation(word).split():
            if ((cleaned_words.isalpha()) & (len(cleaned_words) > 2)):
                if (cleaned_words.lower() not in custom_stopwords):
                    stemed_word = (sno.stem(cleaned_words.lower()))
                    filtered_sentence.append(stemed_word)
                    if (sampled_dataset["SentimentPolarity"].values)[count] == "Positive":
                        all_positive_words.append(stemed_word)
                    if (sampled_dataset["SentimentPolarity"].values)[count] == "Negative":
                        all_negative_words.append(stemed_word)
                else:
                    continue
            else:
                continue
    string = " ".join(filtered_sentence)
    data_corpus.append(string)
    count += 1
print("The length of the data corpus: {}".format(len(data_corpus)))

string = " "
not_stemmed_corpus = []
for review in sampled_dataset["Text"].values:
    filtered_sentence = []
    sentence = removeHtml(review)
    for word in sentence.split():
        for cleaned_words in removePunctuation(word).split():
            if (cleaned_words.isalpha()):
                filtered_sentence.append(cleaned_words.lower())
            else:
                continue
    string = " ".join(filtered_sentence)
    not_stemmed_corpus.append(string)
    
sampled_dataset["CleanedText"] = data_corpus
sampled_dataset["PreserveStopWords"] = not_stemmed_corpus
                
print("The number of positive and negative reviews after data cleaning")
print("{}".format(sampled_dataset["SentimentPolarity"].value_counts()))

connection_sqlobject = sqlite3.connect("sampled_dataset_NB.sqlite")
c = connection_sqlobject.cursor()
connection_sqlobject.text_factory = str
sampled_dataset.to_sql("Reviews", connection_sqlobject, schema=None, if_exists="replace", index=True,
                       index_label=None, chunksize=None, dtype=None)
freq_positive = nltk.FreqDist(all_positive_words)
freq_negative = nltk.FreqDist(all_negative_words)
print("The most common positive words: {}".format(freq_positive.most_common(20)))
print("The most common negative words: {}".format(freq_negative.most_common(20)))

###############################################################################
######################### NaiveBayes-algorithm ################################
###############################################################################

def standardize(X_train_vectors, X_test_vectors):
    from sklearn.preprocessing import StandardScaler
    scalar = StandardScaler(with_mean=False)
    scalar.fit(X_train_vectors)
    X_train_vectors = scalar.transform(X_train_vectors)
    X_test_vectors = scalar.transform(X_test_vectors)
    print("The shape of the X_train_vectors is: {}".format(X_train_vectors.shape))
    print("The shape of the X_test_vectors is: {}".format(X_test_vectors.shape))
    return (X_train_vectors, X_test_vectors)

def top_features(nb_classifier, vectorizer_object):
    neg_class_prob_sorted = (-nb_classifier.feature_log_prob_[0,:]).argsort()
    pos_class_prob_sorted = (-nb_classifier.feature_log_prob_[1,:]).argsort()
    neg_class_features = np.take(vectorizer_object.get_feature_names(), neg_class_prob_sorted[:50])
    pos_class_features = np.take(vectorizer_object.get_feature_names(), pos_class_prob_sorted[:50])
    print("The top 50 most frequent words from the positive class are: \n")
    print(pos_class_features)
    print("The top 50 most frequent words from the negative class are: \n")
    print(neg_class_features)
    del(neg_class_prob_sorted, pos_class_prob_sorted, neg_class_features, pos_class_features)

def performance(nb_classifier, vectorizationType, X_train, y_train, X_test, y_test, optimal_alpha, mse):
    print("-"*100)
    print("~~~~~~~~~~~~~~~~~~ PERFORMANCE EVALUATION ~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Detailed report for the {} Vectorization".format(vectorizationType))
    
    y_pred = nb_classifier.predict(X_test)
    
    test_accuracy = accuracy_score(y_test, y_pred, normalize=True) * 100
    points = accuracy_score(y_test, y_pred, normalize=False)
    print("The number of accurate predictions out of {} data points on unseen data is {}".format(
            X_test.shape[0], points))
    print("Accuracy of the {} model on unseen data is {}".format(
            vectorizationType, np.round(test_accuracy, 2)))
    
    print("Precision of the {} model on unseen data is {}".format(
            vectorizationType, np.round(metrics.precision_score(y_test, y_pred), 4)))
    print("Recall of the {} model on unseen data is {}".format(
            vectorizationType, np.round(metrics.recall_score(y_test, y_pred), 4)))
    print("F1 score of the {} model on unseen data is {}".format(
            vectorizationType, np.round(metrics.f1_score(y_test, y_pred), 4)))
    
    print("Classification report for {} model: \n".format(vectorizationType))
    print(metrics.classification_report(y_test, y_pred))
    
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
    # The total number of actual positives
    p = tp + fn  
    # The total number of actual negatives
    n = fp + tn 
    TPR = tp/p
    TNR = tn/n
    FPR = fp/n
    FNR = fn/p
    print("The True Positive Rate is: {}".format(TPR))
    print("The True Negative Rate is: {}".format(TNR))
    print("The False Positive Rate is: {}".format(FPR))
    print("The False Negative Rate is: {}".format(FNR))
    
    print("Of all the reviews that the model has predicted to be positive, {}% of them are actually positive".format(
            np.round(metrics.precision_score(y_test, y_pred)*100,2)))
    print("Of all the reviews that are actually positive, the model has predicted {}% of them to be positive".format(
            np.round(metrics.recall_score(y_test, y_pred)*100,2)))
    
    print("-"*100)
    print("The confusion matrix for {} model using NaiveBayes method".format(vectorizationType))
    class_names = ["negative", "positive"]
    df_heatmap = pd.DataFrame(metrics.confusion_matrix(y_test, y_pred), index=class_names, columns=class_names)
    fig = plt.figure(figsize=(10,10))
    sns.heatmap(df_heatmap, annot=True, fmt="d")
    plt.ylabel("Predicted label", size=15)
    plt.xlabel("True label", size=15)
    plt.title("Confusion Matrix", size=20)
    plt.show()
    print("-"*100)
    
    print("The ROC_Curve for {} model using NaiveBayes method".format(vectorizationType))
    y_pred_prob = nb_classifier.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    ax.plot([0,1],[0,1], "k--", lw=2, color="r", label="Chance")
    ax.plot(fpr, tpr, label="Test ROC, auc = {}".format(round(metrics.roc_auc_score(y_test, y_pred_prob), 3)))
    plt.title("ROC_Curve", fontsize=20)
    plt.xlabel("FPR", fontsize=15)
    plt.ylabel("TPR", fontsize=15)
    ax.legend()
    plt.show()
    print("-"*100)
    
    del(X_train, y_train, X_test, y_test, vectorizationType, y_pred, y_pred_prob, nb_classifier)
    
def get_GridSearchCV_estimator(vectorizationType, X_train, y_train, X_test, y_test):
    from sklearn.model_selection import TimeSeriesSplit
    alphas = np.logspace(-5,4,100)
    tuned_parameters = [{"alpha": alphas}]
    n_folds = 10
    model = MultinomialNB()
    my_cv = TimeSeriesSplit(n_splits=n_folds).split(X_train)
    gsearch_cv = GridSearchCV(estimator=model, param_grid=tuned_parameters, cv=my_cv, scoring="f1", n_jobs=-1)
    gsearch_cv.fit(X_train, y_train)
    print("GridSearchCV completed for {} model".format(vectorizationType))
    print("Best estimator for {} model".format(vectorizationType), gsearch_cv.best_estimator_)
    print("Best Score for {} model".format(vectorizationType), gsearch_cv.best_score_)
    return gsearch_cv

def plot_errors_Grid(gsearch_cv):
    cv_result = gsearch_cv.cv_results_
    mts = cv_result["mean_test_score"]
    alphas = cv_result["params"]
    alpha_values = []
    for i in range(0, len(alphas)):
        alpha_values.append(alphas[i]["alpha"])
    mse = [1 - x for x in mts]
    
    optimal_alpha = alpha_values[mse.index(min(mse))]
    print("The optimal value of alpha is: {}".format(optimal_alpha))
    
    plt.figure(figsize=(25,20))
    plt.plot(alpha_values, mse, color="green", linestyle="dashed", linewidth=2, marker="o", 
             markerfacecolor="red", markersize=10)
    for xy in zip(alpha_values, np.round(mse, 3)):
        plt.annotate("(%s, %s)" %xy, xy=xy, textcoords="data")
    plt.title("Plot for Errors and Alpha Values", fontsize=20)
    plt.xlabel("Values of Alpha", fontsize=15)
    plt.ylabel("Errors", fontsize=15)
    plt.show()
    return (optimal_alpha, mse)

def get_RandomSearchCV_estimator(vectorizationType, X_train, y_train, X_test, y_test):
    from sklearn.model_selection import TimeSeriesSplit
    alphas = np.logspace(-5, 4, 100)
    tuned_parameters = {'alpha': np.logspace(-5, 4, 100)}
    n_folds = 10
    model = MultinomialNB()
    my_cv = TimeSeriesSplit(n_splits=n_folds).split(X_train)
    n_iter_search = 50
    randsearch_cv = RandomizedSearchCV(estimator=model, param_distributions=tuned_parameters, cv=my_cv, 
                                       scoring="f1", n_iter=n_iter_search, n_jobs=-1)
    randsearch_cv.fit(X_train, y_train)
    print("The RandomizedSearchCV completed for {} model".format(vectorizationType))
    print("The best estimator for {} model".format(vectorizationType), randsearch_cv.best_estimator_)
    print("The best score for {} model".format(vectorizationType), randsearch_cv.best_score_)
    return randsearch_cv
    
def plot_errors_Rand(randsearch_cv):
    cv_result = randsearch_cv.cv_results_
    mts = cv_result["mean_test_score"]
    alphas = cv_result["params"]
    alpha_values = []
    for i in range(0, len(alphas)):
        alpha_values.append(alphas[i]["alpha"])
        
    mse = [1 - x for x in mts]
    optimal_alpha = alpha_values[mse.index(min(mse))]
    print("THe optimal value of alpha is: {}".format(optimal_alpha))
    
    plt.figure(figsize=(25,15))
    plt.plot(alpha_values, mse, color="green", linestyle="dashed", linewidth=2, marker="o", 
             markerfacecolor="red", markersize=10)
    for xy in zip(alpha_values, np.round(mse, 3)):
        plt.annotate("(%s, %s)" % xy, xy=xy, textcoords="data")
    plt.title("Plot Error and Alpha Values", fontsize=20)
    plt.xlabel("Value of Alpha", fontsize=15)
    plt.ylabel("Errors", fontsize=15)
    plt.show()
    return (optimal_alpha, mse)

def naive_bayes_algorithm(X_train, y_train, X_test, y_test, vectorizationType, vectorizer_object, searchType):
    print("*"*100)
    print("Starting Cross Validation steps...")
    print("*"*100)
    if searchType == "GridSearchCV":
        print("Grid_Search_CV algorithm is selected")
        gsearch_cv = get_GridSearchCV_estimator(vectorizationType, X_train, y_train, X_test, y_test)
        optimal_alpha, mse = plot_errors_Grid(gsearch_cv)
        nb_classifier = gsearch_cv.best_estimator_
    if searchType == "RandomSearchCV":
        print("Random_Search_CV algorithm is selected")
        randsearch_cv = get_RandomSearchCV_estimator(vectorizationType, X_train, y_train, X_test, y_test)
        optimal_alpha, mse = plot_errors_Rand(randsearch_cv)
        nb_classifier = randsearch_cv.best_estimator_
    nb_classifier.fit(X_train, y_train)
    top_features(nb_classifier, vectorizer_object)
    performance(nb_classifier, vectorizationType, X_train, y_train, X_test, y_test, optimal_alpha, mse)
    
###############################################################################
################ NaiveBayes-algorithm for Bag of Words ########################
###############################################################################

connection_sqlobject = sqlite3.connect("sampled_dataset_NB.sqlite")
sampled_dataset = pd.read_sql_query(""" SELECT * FROM Reviews """, connection_sqlobject)

X = sampled_dataset["CleanedText"]
y = sampled_dataset["Class_labels"]

"""
split = math.floor(0.8*len(X))
X_train = X[0:split,]
y_train = y[0:split,]
X_test = X[split:,]
y_test = y[split:,]
"""
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)

cv_object = CountVectorizer().fit(X_train)

print("Creating the BOW vectors using the cleaned corpus")
X_train_vectors = cv_object.transform(X_train)
X_test_vectors = cv_object.transform(X_test)

X_train_vectors, X_test_vectors = standardize(X_train_vectors, X_test_vectors)

del(sampled_dataset, X, y, X_train, X_test)

naive_bayes_algorithm(X_train_vectors, y_train, X_test_vectors, y_test, "Bag-of-Words", 
                      cv_object, "RandomSearchCV")
