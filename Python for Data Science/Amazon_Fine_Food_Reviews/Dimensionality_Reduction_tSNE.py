#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 10:55:32 2019

Reference: 
    - https://distill.pub/2016/misread-tsne/
    - https://github.com/rsmahabir/Amazon-Fine-Food-Reviews-Analysis/blob/master/03.%20T-SNE%20on%20Amazon%20Fine%20Food%20Reviews.ipynb
    - https://github.com/PushpendraSinghChauhan/Amazon-Fine-Food-Reviews/blob/master/t-SNE%20visualization%20of%20Amazon%20reviews%20with%20polarity%20based%20color-coding.ipynb

@author: tungutokyo
"""
import sqlite3
import pandas as pd
import numpy as np
import nltk
import string 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle
import warnings
import re
from nltk.stem.snowball import SnowballStemmer
warnings.filterwarnings("ignore")

###############################################################################
########################## Loading the Amazon dataset #########################
###############################################################################

connection_sqlobject = sqlite3.connect('database.sqlite')

# Filter only positive and negative reviews. Do not consider reviews with score = 3
filtered_data = pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 """, connection_sqlobject)
filtered_data['SentimentPolarity'] = filtered_data['Score'].apply(lambda x : 'Positive' if x > 3 else 'Negative')
filtered_data['Class_Labels'] = filtered_data['SentimentPolarity'].apply(lambda x : 1 if x == 'Positive' else 0)

print("The shape of the filtered matrix: {} ".format(filtered_data.shape))
print("The median score values: {} ".format(filtered_data['Score'].mean()))
print("The number of positive and negative reviews before removing the duplicate data")
print("{}".format(filtered_data["SentimentPolarity"].value_counts()))

# Remove the duplicate entries based on the past knowledge 
filtered_duplicates = filtered_data.drop_duplicates(subset={"UserId", "ProfileName", "Time", "Text"}, 
                                                    keep='first', inplace=False)
# Remove the entries where HelpfulnessNumerator > HelpfulnessDenominator
final_data = filtered_duplicates[filtered_duplicates.HelpfulnessNumerator <= filtered_duplicates.HelpfulnessDenominator]
print("The shape of the data matrix after deduplication: {}".format(final_data.shape))
print("The median score value after deduplication: {}".format(final_data['Score'].mean()))
print("The number of positive and negative reviews after removing the duplicate data")
print("{}".format(final_data["SentimentPolarity"].value_counts()))

final_data["SentimentPolarity"].value_counts().plot(kind='bar', color=['green', 'blue'], alpha=0.5,
          title="Distribution of Sentiments after removing deduplication", figsize=(10,10))

###############################################################################
############### Create randomly the 10000-sample dataset ######################
###############################################################################

sampled_dataset = final_data.sample(n=10000, replace=False, random_state=0).reset_index()
print("The shape of the sampled data: {}".format(sampled_dataset.shape))

sampled_dataset = sampled_dataset.drop(labels=['index', 'Id', 'ProductId', 'UserId', 'Score', 
                    'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Summary'], axis=1)
print("The shape of the sampled dataset after dropping coloumns: {}".format(sampled_dataset.shape))

sampled_dataset = sampled_dataset.sort_values('Time', axis=0, ascending=False, inplace=False, 
                                              kind='quicksort', na_position='last')
sampled_dataset = sampled_dataset.reset_index()
sampled_dataset = sampled_dataset.drop(labels=['index'], axis=1)
sampled_dataset["SentimentPolarity"].value_counts().plot(kind="bar", color=['Green', 'blue'], alpha=0.5, 
               title="Distribution of Sentiments after sampling", figsize=(10,10))

###############################################################################
########################### Data Cleaning stage ###############################
###############################################################################

def removeHtml(sentence):
    pattern = re.compile('<.*?>')
    cleaned_text = re.sub(pattern, ' ', sentence)
    return cleaned_text

def removePunctuations(sentence):
    cleaned_text = re.sub('[^a-zA-Z]', ' ', sentence)
    return cleaned_text

sno = SnowballStemmer(language='english')

# Remove the word 'not' from stopwords
default_stopwords = set(stopwords.words('english'))
remove_not = set(['not'])
custom_stopwords = default_stopwords - remove_not

# Build a data corpus by removing all stopwords except 'not'
data_corpus = []
for sentence in sampled_dataset['Text'].values:
    review = sentence 
    review = removeHtml(review)
    review = removePunctuations(review)
    # Convert each review to lower case
    review = review.lower()
    # Split each sentence into words
    review = review.split()
    review = [sno.stem(word) for word in review if not word in set(custom_stopwords)]
    review = ' '.join(review)
    data_corpus.append(review)

print("The length of the data corpus is : {}".format(len(data_corpus)))
sampled_dataset['CleanedText'] = data_corpus

# Build a data corpus by all HTML tags and punctuations. Stopwords are preserved 
data_corpus = []
for sentence in sampled_dataset['Text'].values:
    review = sentence
    review = removeHtml(review)
    review = removePunctuations(review)
    data_corpus.append(review)
sampled_dataset['RemovedHTML'] = data_corpus

connection_sqlobject = sqlite3.connect('sampled_dataset_PCA_10000.sqlite')
c = connection_sqlobject.cursor()
connection_sqlobject.text_factory = str
sampled_dataset.to_sql('Reviews', connection_sqlobject, schema=None, if_exists='replace', index=True, 
                       index_label=None, chunksize=None, dtype=None)

###############################################################################
############################## T-SNE method ###################################
###############################################################################

from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.preprocessing import StandardScaler

def tsne(dataset, labels, perplexity):
    model = TSNE(n_components=2, random_state=0, n_jobs=8, perplexity=perplexity, n_iter=5000)
    tsne_data = model.fit_transform(dataset)
    tsne_data = np.vstack((tsne_data.T, labels)).T
    tsne_df = pd.DataFrame(data=tsne_data, columns=("Dimension 1", "Dimension 2", "labels"))
    print("T-SNE plot for perplexity = {}".format(perplexity))
    return tsne_df
    
""" Bag of Words """

connection_sqlobject = sqlite3.connect('sampled_dataset_PCA_10000.sqlite')
sampled_dataset = pd.read_sql_query(""" SELECT * FROM Reviews """, connection_sqlobject)
cv_object = CountVectorizer()
bow_matrix_10000 = cv_object.fit_transform(sampled_dataset['CleanedText']).toarray()
scalar = StandardScaler(with_mean=False)
standardized_data = scalar.fit_transform(bow_matrix_10000)
dataset = pd.DataFrame(standardized_data)
labels = sampled_dataset['SentimentPolarity']
del(bow_matrix_10000, standardized_data, sampled_dataset)

tsne_df = tsne(dataset, labels, 20)
#plt.figure(figsize=(10,10))
#sns.scatterplot(x="Dimension 1", y="Dimension 2", data=tsne_df, hue="labels")
sns.FacetGrid(tsne_df, hue="labels", hue_order=["Positive", "Negative"], size=8).map(
        plt.scatter, 'Dimension 1', 'Dimension 2', edgecolor="w").add_legend()
plt.title("T-SNE of Bag of Words with perplexity = 20 and n_iter = 5000", fontsize=15)
plt.show()


""" A TF-IDF model using RemovedHTML texts """

connection_sqlobject = sqlite3.connect('sampled_dataset_PCA_10000.sqlite')
sampled_dataset = pd.read_sql_query(""" SELECT * FROM Reviews """, connection_sqlobject)

tf_idf_object = TfidfVectorizer(ngram_range=(1,1))
final_tf_idf_vectors = tf_idf_object.fit_transform(sampled_dataset['RemovedHTML']).toarray()
print("The type of count vectorizer : ", type(tf_idf_object))
print("The shape of the TFIDF vectorizer : ", final_tf_idf_vectors.shape)
print("The number of unique words : ", len(final_tf_idf_vectors[0]))

scalar = StandardScaler(with_mean=False)
standardized_data = scalar.fit_transform(final_tf_idf_vectors)
print("The shape of standardized data : {}".format(standardized_data.shape))

dataset = pd.DataFrame(standardized_data)
labels = sampled_dataset['SentimentPolarity']

del(final_tf_idf_vectors, standardized_data, sampled_dataset)

tsne_20 = tsne(dataset, labels, 20)
sns.FacetGrid(tsne_20, hue="labels", hue_order=['Positive', 'Negative'], size=8).map(
        plt.scatter, 'Dimension 1', 'Dimension 2', edgecolor="w").add_legend()
plt.title("T-SNE of TF-IDF with perplexity = 20 and n_iter = 5000", fontsize=15)

tsne_50 = tsne(dataset, labels, 50)
sns.FacetGrid(tsne_50, hue="labels", hue_order=['Positive', 'Negative'], size=8).map(
        plt.scatter, 'Dimension 1', 'Dimension 2', edgecolor="w").add_legend()
plt.title("T-SNE of TF-IDF with perplexity = 50 and n_iter = 5000", fontsize=15)

tsne_100 = tsne(dataset, labels, 100)
sns.FacetGrid(tsne_100, hue="labels", hue_order=['Positive', 'Negative'], size=8).map(
        plt.scatter, 'Dimension 1', 'Dimension 2', edgecolor="w").add_legend()
plt.title("T-SNE of TF-IDF with perplexity = 100 and n_iter = 5000", fontsize=15)


""" The Average Word2Vec """

connection_sqlobject = sqlite3.connect('sampled_dataset_PCA_10000.sqlite')
sampled_dataset = pd.read_sql_query(""" SELECT * FROM Reviews """, connection_sqlobject)

word2vec_corpus = []
for sentence in sampled_dataset['RemovedHTML'].values:
    word2vec_corpus.append(sentence.split())
print(sampled_dataset['RemovedHTML'].values[0])
print("-"*200)
print(word2vec_corpus[0])
print("The size of the word2vec text corpus : {}".format(len(word2vec_corpus)))

# Consider only those words occurs at least 5 times 
word2vec_model = Word2Vec(sentences=word2vec_corpus, size=100, min_count=5, workers=6)
word2vec_words = list(word2vec_model.wv.vocab)
print("The number of words that occured minimum 5 times : {}".format(len(word2vec_words)))
print("The sample words from word2vec_words list : ", word2vec_words[0:50])

# Check most similar words present for any given words
word2vec_model.wv.most_similar('run')

# The average word2vec for each sentence/review will be stored 
sent_vectors = []
for sentence in word2vec_corpus:
    sent_vec = np.zeros(100)
    count_words = 0
    for word in sentence:
        if word in word2vec_words:
            word_vectors = word2vec_model.wv[word]
            sent_vec += word_vectors
            count_words += 1
    if count_words != 0:
        sent_vec /= count_words
    sent_vectors.append(sent_vec)
print("The length of the sentence vectors : {}".format(len(sent_vectors)))
print("The size of each vector : {}".format(len(sent_vectors[0])))
sent_vectors = np.array(sent_vectors)

scalar = StandardScaler(with_mean=True)
standardized_data = scalar.fit_transform(sent_vectors)

dataset = pd.DataFrame(standardized_data)
labels = sampled_dataset['SentimentPolarity']

del(sent_vectors, standardized_data, sent_vec, sentence, sampled_dataset)

tsne_20 = tsne(dataset, labels, 20)
tsne_50 = tsne(dataset, labels, 50)
tsne_100 = tsne(dataset, labels, 100)
sns.FacetGrid(tsne_20, hue="labels", hue_order=['Positive', 'Negative'], size=8).map(
        sns.scatterplot, 'Dimension 1', 'Dimension 2', edgecolor='w').add_legend()
sns.FacetGrid(tsne_50, hue="labels", hue_order=['Positive', 'Negative'], size=8).map(
        sns.scatterplot, 'Dimension 1', 'Dimension 2', edgecolor='w').add_legend()
sns.FacetGrid(tsne_100, hue="labels", hue_order=['Positive', 'Negative'], size=8).map(
        sns.scatterplot, 'Dimension 1', 'Dimension 2', edgecolor='w').add_legend()
