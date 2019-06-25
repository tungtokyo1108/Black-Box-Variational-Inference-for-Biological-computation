#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 14:50:37 2019

@author: tungutokyo
"""

import warnings
warnings.filterwarnings("ignore")
import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle
from tqdm import tqdm
from datetime import datetime as dt
import os
from sklearn import decomposition
  
###############################################################################
####### Loading the Amazon dataset, after removing duplication data ###########
###############################################################################

connection_sqlobject = sqlite3.connect("totally_processed_DB.sqlite")
processed_db = pd.read_sql_query(""" SELECT * FROM Reviews """, connection_sqlobject)

df0 = processed_db[processed_db['Class_Labels'] == 0]
df1 = processed_db[processed_db['Class_Labels'] == 1]
df_balncd = pd.concat([df1.sample(20000, random_state=0), df0.sample(20000, random_state=0)])

X_train = df_balncd['CleanedText']
labels = df_balncd['Class_Labels']
df_balncd.head()

df_balncd["SentimentPolarity"].value_counts().plot(kind='bar', color=['green', 'blue'], 
         title='Distribution of Positive and Negative reviews', figsize=(10,10), alpha=0.5)


###############################################################################
####################### Dimensionality reduction - PCA ########################
###############################################################################

def standardize(data):
    from sklearn.preprocessing import StandardScaler
    data_std = StandardScaler().fit_transform(data)
    return data_std

def pca_vis(data, labels):
    st = dt.now()
    pca = decomposition.PCA(n_components=4)
    pca_reduced = pca.fit_transform(data)
    
    print("The shape of transformed data", pca_reduced.shape)
    print(pca_reduced[0:4])
    
    pca_data = np.vstack((pca_reduced.T, labels)).T
    print("The shape of data with labels", pca_data.shape)
    
    pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal_component", "2nd_principal_component",
                                                  "3rd_principal_component", "4th_principal_component", "labels"))
    print(pca_df.head())
    
    #sns.FacetGrid(pca_df, hue="labels", height=6).map(plt.scatter, 
    #             "1st_principal_component", "2nd_principal_component").add_legend()
    
    fig, ax = plt.subplots(1,2, figsize=(20,10))
    ax[0] = sns.scatterplot(x="1st_principal_component", y="2nd_principal_component", 
              hue="labels", style="labels", data=pca_df, alpha=0.5, ax=ax[0])
    ax[0].set_xlabel("1st_principal_component ({}%)".format(round(pca.explained_variance_ratio_[0]*100),2), fontsize=15)
    ax[0].set_ylabel("2nd_principal_component ({}%)".format(round(pca.explained_variance_ratio_[1]*100),2), fontsize=15)
    ax[1] = sns.scatterplot(x="3rd_principal_component", y="4th_principal_component",
              hue="labels", style="labels", data=pca_df, alpha=0.5, ax=ax[1])
    ax[1].set_xlabel("3rd_principal_component ({}%)".format(round(pca.explained_variance_ratio_[2]*100),2), fontsize=15)
    ax[1].set_ylabel("4th_principal_component ({}%)".format(round(pca.explained_variance_ratio_[3]*100),2), fontsize=15)
    
    plt.show()
    print("Time taken to perform Principal Component Analysis: {}".format(dt.now()-st))

def pca_redu(data):
    pca = decomposition.PCA()
    pca_data = pca.fit_transform(data)
    plt.figure(figsize=(15,10))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.axis('tight')
    plt.grid()
    plt.axhline(0.95, c='r')
    plt.xlabel("Number of components", fontsize=15)
    plt.ylabel("Cumulative explained variance", fontsize=15)
    plt.legend()

def pca_results(data, pca):
    
    # Dimension indexing 
    dimensions = ['Dimension {}'.format(i) for i in range(1, len(pca.components_)+1)]
    
    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4))
    # components.index = dimensions
    
    # PCA explained variance 
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    variance_ratios.index = dimensions
    
    fig, ax = plt.subplots(figsize=(10,15))
    
    components.plot(ax=ax, kind='barh', fontsize=10)
    ax.set_xlabel("Feature Weights", fontsize=15)
    ax.set_yticklabels(dimensions, rotation=0, fontsize=10)
    
    for i, ev in enumerate(pca.explained_variance_ratio_):
        ax.text(ax.get_xlim()[1] + 0.05, i-0.1, "Explained Variance\n %.4f"%(ev), rotation=-90, fontsize=10)
        
    return pd.concat([variance_ratios, components], axis = 1)

###############################################################################
############################### Avg Word2Vec ##################################
###############################################################################

word2vec_corpus = []
for sentence in X_train:
    word2vec_corpus.append(sentence.split())
print("The size of the word2vec text corpus: ", len(word2vec_corpus))

word2vec_model = Word2Vec(sentences=word2vec_corpus, size=200, min_count=5, workers=6)
word2vec_words = list(word2vec_model.wv.vocab)

def vectorize_w2v(dataset, word2vec_model, word2vec_words):
    word2vec_corpus = []
    for sentence in dataset:
        word2vec_corpus.append(sentence.split())
    
    sent_vectors = []
    for sentence in tqdm(word2vec_corpus):
        sent_vec = np.zeros(200)
        count_words = 0
        for word in sentence:
            if word in word2vec_words:
                word_vectors = word2vec_model.wv[word]
                sent_vec += word_vectors
                count_words += 1
        if count_words != 0:
            sent_vec /= count_words
        sent_vectors.append(sent_vec)
    sent_vectors = np.array(sent_vectors)
    return sent_vectors

X_train_vectors = vectorize_w2v(X_train, word2vec_model, word2vec_words)
X_train_vectors = standardize(X_train_vectors)
print("The shape of our Avg Word2vec train vectorizer ", X_train_vectors.shape)

pca_redu(X_train_vectors)
pca_vis(X_train_vectors, labels)

###############################################################################
######################### TFIDF weighted W2V ##################################
###############################################################################

# Store the list of words for each review 
word2vec_corpus = []
for sentence in X_train:
    word2vec_corpus.append(sentence.split())
    
# Consider only those words which occurs at least 5 times with min_count=5
word2vec_model = Word2Vec(sentences=word2vec_corpus, size=200, min_count=5, workers=8)
word2vec_words = list(word2vec_model.wv.vocab)

"""
- Initializing the TF-IDF contructor with review texts
- HTML tags and punctuations are removed 
- Stopwords are preserved 
"""
tf_idf_object = TfidfVectorizer(ngram_range=(1,1)).fit(X_train)

"""
- This method returns the Average Word2Vec vectors for all reviews in a given dataset
"""
def vectorize_tfidf_w2v(dataset, tf_idf_object, word2vec_model, word2vec_words):
    word2vec_corpus = []
    for sentence in dataset:
        word2vec_corpus.append(sentence.split())
    
    # Use the earlier TF-IDF object to vectorize test and cv data
    tf_idf_matrix = tf_idf_object.transform(dataset)
    tfidf_features = tf_idf_object.get_feature_names()
    
    # Build a dictionary with words as a key and the idfs as value
    dictionary = dict(zip(tfidf_features, list(tf_idf_object.idf_)))
    
    # Algorithm for finding the TF-IDF weight average word2vec vectors 
    tfidf_sent_vectors = []
    row = 0
    for sentence in tqdm(word2vec_corpus):
        # This is used to add word vectors and find averages at each iteration
        sent_vec = np.zeros(200)
        # This will store the count of the words with a valid vector in each review text
        weight_sum = 0
        for word in sentence:
            if ((word in word2vec_words) and (word in tfidf_features)):
                word_vectors = word2vec_model.wv[word]
                tf_idf = dictionary[word]*(sentence.count(word)/len(sentence))
                sent_vec += (word_vectors * tf_idf)
                weight_sum += tf_idf
        if weight_sum != 0:
            sent_vec /= weight_sum
        tfidf_sent_vectors.append(sent_vec)
        row += 1
    
    tfidf_sent_vectors = np.array(tfidf_sent_vectors)
    return tfidf_sent_vectors

X_train_vectors = vectorize_tfidf_w2v(X_train, tf_idf_object, word2vec_model, word2vec_words)
X_train_vectors = standardize(X_train_vectors)
        
print ("The shape of our Avg Word2Vec train vectorizer ", X_train_vectors.shape)

pca_redu(X_train_vectors)
pca_vis(X_train_vectors, labels)
    
