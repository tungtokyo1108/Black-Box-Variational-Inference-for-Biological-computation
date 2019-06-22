#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 08:54:24 2019

Data Source: https://www.kaggle.com/snap/amazon-fine-food-reviews
Reference: 
    - https://github.com/PushpendraSinghChauhan/Amazon-Fine-Food-Reviews
    - https://github.com/rsmahabir/Amazon-Fine-Food-Reviews-Analysis
    - https://github.com/krpiyush5/Amazon-Fine-Food-Review

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
from sklearn import metrics
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle
from tqdm import tqdm
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from wordcloud import WordCloud 

connection_sqlobject = sqlite3.connect('database.sqlite')
filter_data = pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 """, connection_sqlobject)

filter_data['SentimentPolarity'] = filter_data['Score'].apply(lambda x : 'Positive' if x > 3 else 'Negative')
filter_data['Class_Labels'] = filter_data['SentimentPolarity'].apply(lambda x : 1 if x == 'Positive' else 0)

print("The number of data points in our data", filter_data.shape[0])
print("The number of features in our data", filter_data.shape[1])
filter_data.head(10)

###############################################################################
#################### Data Clearning: Deduplication ############################
###############################################################################

# Check user id provided review more than one time for different products 
display = pd.read_sql_query(""" SELECT UserId, ProfileName, Time, Score, Text, COUNT(*) 
            FROM Reviews 
            GROUP BY UserId 
            HAVING COUNT(*)>1""", connection_sqlobject)
print(display.shape)
display.head()

display[display['UserId'] == '#oc-R115TNMSPFT9I7']
display['COUNT(*)'].sum()

# Checking the number of entries that came from UserId="AZY10LLTJ71NX"
display = pd.read_sql_query(""" SELECT * 
                            FROM Reviews
                            WHERE Score != 3 AND UserId = "AZY10LLTJ71NX" 
                            ORDER BY ProductID """, connection_sqlobject)
display.head()

print("The shape of the filtered matrix : {}".format(filter_data.shape))
print("The median score values: {}".format(filter_data['Score'].mean()))
print("The number of positive and negative reviews before the removal of duplicate data")
print(filter_data["SentimentPolarity"].value_counts())

sorted_data = filter_data.sort_values('Time', axis=0, ascending=False, inplace=False, kind='quicksort',
                                      na_position='last')
filtered_duplicates = sorted_data.drop_duplicates(subset={"UserId", "ProfileName", "Time", "Text"}, 
                                                  keep='first', inplace=False)
print("The shape of the data matrix after deduplication, Stage 1: {}".format(filtered_duplicates.shape))

final_data = filtered_duplicates[filtered_duplicates.HelpfulnessNumerator <= filtered_duplicates.HelpfulnessDenominator]
print("The shape of the data matrix after deduplication, Stage 2: {}".format(final_data.shape))
print("The median score values after deduplication: {}".format(final_data["Score"].mean()))
print("The number of positive and negative reviews after the removal of duplicatie data.")
print(final_data["SentimentPolarity"].value_counts())

# Check to see how much of data still remain 
print("Check to see how much percentage of data still remains.")
retained_per = (final_data['SentimentPolarity'].size*1.0)/(filter_data['SentimentPolarity'].size*1.0)*100
removed_per = 100 - retained_per 
print("Percentage of redundant data removed: {}".format(removed_per))
print("Percentage of original data retained: {}".format(retained_per))

# Delete unwanted variables to free up memory space 
del(filtered_duplicates, filter_data, sorted_data)
final_data.head(10)

# Display distribution of Positive and Negative reviews 
final_data["Class_Labels"].value_counts().plot(kind='bar', color=['green', 'blue'], 
          title='Distribution of Positive and Negative reviews after De-duplication', figsize=(10,10), alpha=0.5)

###############################################################################
######################## Preprocessing Review Text ############################
###############################################################################

# Printing some random reviews from the deduplication dataset
sent_1 = final_data['Text'].values[0]
print(sent_1)
print("Review Polarity: {}".format(final_data['SentimentPolarity'].values[0]))
print("-"*200)

sent_2 = final_data['Text'].values[1000]
print(sent_2)
print("Review Polarity: {}".format(final_data['SentimentPolarity'].values[1000]))
print("-"*200)

sent_3 = final_data['Text'].values[1500]
print(sent_3)
print("Review Polarity: {}".format(final_data['SentimentPolarity'].values[1500]))
print("-"*200)

sent_4 = final_data['Text'].values[5000]
print(sent_4)
print("Review Polarity: {}".format(final_data['SentimentPolarity'].values[4000]))
print("-"*200)

sent_5 = final_data['Text'].values[12500]
print(sent_5)
print("Review Polarity: {}".format(final_data['SentimentPolarity'].values[12500]))
print("-"*200)

sent_6 = final_data['Text'].values[255500]
print(sent_6)
print("Review Polarity: {}".format(final_data['SentimentPolarity'].values[255500]))
print("-"*200)

# Remove urls from text python
sent_1 = re.sub(r"http\S+", " ", sent_1)
sent_2 = re.sub(r"http\S+", " ", sent_2)
sent_3 = re.sub(r"http\S+", " ", sent_3)
sent_4 = re.sub(r"http\S+", " ", sent_4)
sent_5 = re.sub(r"http\S+", " ", sent_5)
sent_6 = re.sub(r"http\S+", " ", sent_6)

print(sent_1, "\n")
print("-"*200)
print(sent_2, "\n")
print("-"*200)
print(sent_3, "\n")
print("-"*200)
print(sent_4, "\n")
print("-"*200)
print(sent_5, "\n")
print("-"*200)
print(sent_6, "\n")
print("-"*200)

def removeHtml(sentence):
    pattern = re.compile('<.*?>')
    clearned_text = re.sub(pattern, ' ', sentence)
    return clearned_text

print(removeHtml(sent_1) + "\n")
print(removeHtml(sent_2) + "\n")
print(removeHtml(sent_3) + "\n")
print(removeHtml(sent_4) + "\n")
print(removeHtml(sent_5) + "\n")
print(removeHtml(sent_6) + "\n")

# Expand the reviews x is a input string of any length.
# Convert all the words to lower case 
def decontracted(x):
    x = str(x).lower()
    x = x.replace(",000,000", " m").replace(",000", "k").replace("′", "'").replace("’", "'")\
    .replace("won't", " will not").replace("cannot", " can not").replace("can't", " can not")\
    .replace("n't", " not").replace("what's", " what is").replace("it's", " it is")\
    .replace("'ve", " have").replace("'m", " am").replace("'re", "are")\
    .replace("he's", " he is").replace("she's", " she is").replace("'s", " own")\
    .replace("%", " percent").replace("₹", " rupee "). replace("$", " dollar")\
    .replace("€", " euro ").replace("'ll", " will").replace("how's"," how has").replace("y'all"," you all")\
    .replace("o'clock"," of the clock").replace("ne'er"," never").replace("let's"," let us")\
    .replace("finna"," fixing to").replace("gonna"," going to").replace("gimme"," give me").replace("gotta"," got to").replace("'d"," would")\
    .replace("daresn't"," dare not").replace("dasn't"," dare not").replace("e'er"," ever").replace("everyone's"," everyone is")\
    .replace("'cause'"," because")
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    return x

sent_1 = re.sub("\S*\d\S*", " ", sent_1).strip()
print(sent_1)

""" Data Clearning Stage """
# Remove words with numbers python
def removeNumbers(sentence):
    sentence = re.sub("\S*\d\S*", " ", sentence).strip()
    return (sentence)

# Function to clean html tags from a sentence 
def removeHtml(sentence):
    pattern = re.compile('<.*?>')
    cleaned_text = re.sub(pattern, ' ', sentence)
    return cleaned_text

# Remove URL from sentences
def removeURL(sentence):
    text = re.sub(r"http\S+", " ", sentence)
    sentence = re.sub(r"www.\S+", " ", text)
    return (sentence)

# Keep only words containing letters A-Z and a-z.
def removePunctuations(sentence):
    cleaned_text = re.sub('[^a-zA-Z]', ' ', sentence)
    return (cleaned_text)

# Remove words like 'xxxxxx', 'testtttting'
def removePatterns(sentence):
    cleaned_text = re.sub("\\s*\\b(?=\\w*(\\w)\\1{2,})\\w*\\b", ' ', sentence)
    return (cleaned_text)

# Stemming and stopwords removal 
from nltk.stem.snowball import SnowballStemmer
sno = SnowballStemmer(language='english')

#Removing the world 'not' from stopwords
default_stopwords = set(stopwords.words('english'))
remove_not = set(['no', 'nor', 'not'])
custom_stopwords = default_stopwords - remove_not

"""

- Check the distribution of streammed word lengths across the whole review dataset to understand 
what is the length of maximum number of words. 
- Keep only the words which has a length less than that of a specific length
 
"""
total_words = []
for review in tqdm(final_data['Text'].values):
    filtered_sentence = []
    review = decontracted(review)
    review = removeNumbers(review)
    review = removeHtml(review)
    review = removeURL(review)
    review = removePunctuations(review)
    review = removePatterns(review)
    for  cleaned_words in review.split():
        if ((cleaned_words not in custom_stopwords)):
            stemed_word = (sno.stem(cleaned_words.lower()))
            total_words.append(stemed_word)
            
# Get list of unique words 
total_words = list(set(total_words))
# List to hold the length of each words used in all the reviews used across the whole dataset
dist = []
for i in tqdm(total_words):
    length = len(i)
    dist.append(length)
    
plt.figure(figsize=(10,10))
plt.hist(dist, color = "green", edgecolor="blue", bins = 100, alpha=0.5)
plt.title("Distribution of the length of Words across all reviews", fontsize=20)
plt.xlabel("Word Lengths", fontsize=15)
plt.xticks(fontsize=10)
plt.ylabel("Number of Words", fontsize=15)
plt.yticks(fontsize=10)

"""
Combining all the above data cleaning methologies as discussed above 
Processing review texts

"""
preprocessed_reviews = []
all_positive_words = []
all_negative_words = []

# Iterator to iterate through the list of reviews and check if a given review belongs to positive or negative
count = 0
string = ' '
stemed_word = ' '
for review in tqdm(final_data['Text'].values):
    filtered_sentence = []
    review = decontracted(review)
    review = removeNumbers(review)
    review = removeHtml(review)
    review = removeURL(review)
    review = removePunctuations(review)
    review = removePatterns(review)
    for cleaned_words in review.split():
        if ((cleaned_words not in custom_stopwords) and (2<len(cleaned_words)<16)):
            stemed_word = (sno.stem(cleaned_words.lower()))
            filtered_sentence.append(stemed_word)
            if (final_data['SentimentPolarity'].values)[count] == 'Positive':
                all_positive_words.append(stemed_word)
            if (final_data['SentimentPolarity'].values)[count] == 'Negative':
                all_negative_words.append(stemed_word)
            else:
                continue
    
    # Final string of cleaned words        
    review = " ".join(filtered_sentence)
    preprocessed_reviews.append(review.strip())
    count += 1

# Save the list of positive and negative words
with open('all_positive_words.pkl', 'wb') as file:
    pickle.dump(all_positive_words, file)

with open('all_negative_words.pkl', 'wb') as file:
    pickle.dump(all_negative_words, file)
    
final_data['CleanedText'] = preprocessed_reviews
print("The length of the data corpus is: {}".format(len(preprocessed_reviews)))

"""
Preprocessing for summary 

"""
preprocessed_summary = []
count = 0
string = ' '
stemed_word = ' '

for summary in tqdm(final_data['Summary'].values):
    filtered_sentence = []
    summary = decontracted(summary)
    summary = removeNumbers(summary)
    summary = removeHtml(summary)
    summary = removePunctuations(summary)
    summary = removePatterns(summary)
    for clearned_words in summary.split():
        if (2 < len(cleaned_words) < 16):
            stemed_word = (sno.stem(cleaned_words.lower()))
            filtered_sentence.append(stemed_word)
    summary = " ".join(filtered_sentence)
    preprocessed_summary.append(summary.strip())
    count += 1

final_data['CleanedSummary'] = preprocessed_summary
final_data['Combined_Reviews'] = final_data['CleanedText'].values + " " + final_data['CleanedSummary'].values

# Store final table into a SQLLite table for future
connection_sqlobject = sqlite3.connect('totally_processed_DB.sqlite')
c = connection_sqlobject.cursor()
connection_sqlobject.text_factory = str
final_data.to_sql('Reviews', connection_sqlobject, schema=None, if_exists='replace', index=True)


###############################################################################
############################# Featurization ###################################
###############################################################################

connection_sqlobject = sqlite3.connect('totally_processed_DB.sqlite')
processed_db = pd.read_sql_query(""" SELECT * FROM Reviews """, connection_sqlobject)
processed_db.head()

X_train = processed_db['CleanedText'].iloc[150000:250000,]
y_train = processed_db['Class_Labels'].iloc[150000:250000,]
X_test = processed_db['CleanedText'].iloc[250000:280000,]
y_test = processed_db['Class_Labels'].iloc[250000:280000,]
X_calib = processed_db['CleanedText'].iloc[280000:320000,]
y_calib = processed_db['Class_Labels'].iloc[280000:320000,]

processed_db["SentimentPolarity"].value_counts().plot(kind='bar', color=['green', 'blue'], 
            title='Distribution Of Positive and Negative reviews', figsize=(10,10), alpha=0.5)

"""
Bag of words 

- is a way of extracting features from text for use in modeling.
- is very simple and flexible, and can be used in a myrlab of ways for extracting features from documents
- count how many number of times a word is present in a review. 
- get a Sparse Matrix representation for all the words in the review 

"""
# Initializing the BOW constructor
cv_object = CountVectorizer(min_df = 10, max_features = 50000, dtype='float').fit(X_train)

# Print names of some random features
print("Some features names ", cv_object.get_feature_names()[100:110])
print("-"*200)

print("Creating the BOW vectors using the cleaned corpus")
X_train_vectors = cv_object.transform(X_train)
X_test_vectors = cv_object.transform(X_test)
X_calib_vectors = cv_object.transform(X_calib)

print("The type of count vectorizer ", type(X_train_vectors))
print("The shape of our train BOW vectorizer ", X_train_vectors.get_shape())
print("The shape of our test BOW vectorizer ", X_test_vectors.get_shape())
print("The number of unique words ", X_train_vectors.get_shape()[1])

with open('X_train_BOW.pkl', 'wb') as file:
    pickle.dump(X_train_vectors, file)
    
with open('y_train_BOW.pkl', 'wb') as file:
    pickle.dump(y_train, file)

with open('X_test_BOW.pkl', 'wb') as file:
    pickle.dump(X_test_vectors, file)

with open('y_test_BOW.pkl', 'wb') as file:
    pickle.dump(y_test, file)
    
with open('X_calib_BOW.pkl', 'wb') as file:
    pickle.dump(X_calib_vectors, file)

with open('y_calib_BOW.pkl', 'wb') as file:
    pickle.dump(y_calib, file)


"""

Bi-grams and n-Grams
- Consider the pairs of consequent words (bi-grams) or q sequence of n consecutive words

"""
freq_dist_positive = nltk.FreqDist(all_positive_words)
freq_dist_negative = nltk.FreqDist(all_negative_words)
print("Most Common Positive Words: ", freq_dist_positive.most_common(20))
print("-"*200)
print("Most Common Negative Words; ", freq_dist_negative.most_common(20))

cv_object = CountVectorizer(ngram_range=(1,2), min_df=10, max_features=50000, dtype='float')
cv_object.fit(X_train)
print("Some feature names ", cv_object.get_feature_names()[100:110])
print('-'*200)

print("Creating the Bi-Gram vectors using the cleaned corpus")
X_train_vectors = cv_object.transform(X_train)
X_test_vectors = cv_object.transform(X_test)
print ("The type of count vectorizer ", type(X_train_vectors))
print("The shape of our train BiGram vectorizer ", X_train_vectors.get_shape())
print("The shape of our test BiGram vectorizer ", X_test_vectors.get_shape())
print("The number of unique words ", X_train_vectors.get_shape()[1])

with open('X_train_BiGrm.pkl', 'wb') as file:
    pickle.dump(X_train_vectors, file)

with open('y_train_BiGrm.pkl', 'wb') as file:
    pickle.dump(y_train, file)
    
with open('X_test_BiGrm.pkl', 'wb') as file:
    pickle.dump(X_test_vectors, file)
    
with open('y_test_BiGrm.pkl', 'wb') as file:
    pickle.dump(y_test, file)
    
with open('X_calib_BiGrm.pkl', 'wb') as file:
    pickle.dump(X_calib_vectors, file)
    
with open('y_calib_BiGrm.pkl', 'wb') as file:
    pickle.dump(y_calib, file)


""" 
TF-IDF
- assigns each word in a document a number that is a proportinal to its frequency in the dpcumnet 
  and inversely proportional to the number of documents in which it occurs 
- very common words  receive heavily discounted tf-idf scores, 
  in contrast to words that are very specific to the document in question.  
- https://buhrmann.github.io/tfidf-analysis.html
"""
tf_idf_vect = TfidfVectorizer(ngram_range=(1,2), min_df = 10, max_features=50000, dtype='float')
tf_idf_obj = tf_idf_vect.fit(X_train)
print("Some feature names ", tf_idf_obj.get_feature_names()[100:110])
print("-"*200)

X_train_vectors = tf_idf_obj.transform(X_train)
X_test_vectors = tf_idf_obj.transform(X_test)
X_calib_vectors = tf_idf_obj.transform(X_calib)

def top_tfidf_feats(row, features, top_n=25):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['features', 'tfidf']
    return df

top_tfidf = top_tfidf_feats(X_train_vectors[1,:].toarray()[0], tf_idf_obj.get_feature_names(), 25)
top_tfidf

with open('X_train_TFIDF.pkl', 'wb') as file:
    pickle.dump(X_train_vectors, file)

with open('y_train_TFIDF.pkl', 'wb') as file:
    pickle.dump(y_train, file)
    
with open('X_test_TFIDF.pkl', 'wb') as file:
    pickle.dump(X_test_vectors, file)
    
with open('y_test_TFIDF.pkl', 'wb') as file:
    pickle.dump(y_test, file)
    
with open('X_calib_TFIDF.pkl', 'wb') as file:
    pickle.dump(X_calib_vectors, file)
    
with open('y_calib_TFIDF.pkl', 'wb') as file:
    pickle.dump(y_calib, file)


"""
The Average Word2Vec
- Convert each word to a vector, sum them up and divide by the number of words in that particular sentence 
"""
word2vec_corpus = []
for sentence in X_train:
    # a list of words for each sentence for all the reviews 
    word2vec_corpus.append(sentence.split())

print("The size of the Word2Vec text corpus : ", len(word2vec_corpus))
# With min_count=5 considers only those words for our model which occurs at lease 5 times 
word2vec_model = Word2Vec(sentences=word2vec_corpus, size=200, min_count=5, workers=6)
word2vec_words = list(word2vec_model.wv.vocab)
print("The number of words that occured minimum 5 times : ", len(word2vec_words))

def vectorize_w2v(dataset, word2vec_model, word2vec_words):
    word2vec_corpus = []
    for sentence in dataset:
        word2vec_corpus.append(sentence.split())
    # Create average Word2Vec model by computing the average word2vec for each review 
    # The average word2vec for each sentence/review will be stored in list
    sent_vectors = []
    for sentence in tqdm(word2vec_corpus):
        # 200 dimensional array, where all elements are zero. 
        # This is used to add word vectors and find the averages at each iteration.
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
X_test_vectors = vectorize_w2v(X_test, word2vec_model, word2vec_words)
X_calib_vectors = vectorize_w2v(X_calib, word2vec_model, word2vec_words)

print("The shape of our Avg Word2Vec train vectorizer", X_train_vectors.shape)
print("The shape of our Avg Word2Vec test vectorizer", X_test_vectors.shape)
print("The shape of out Avg Word2Vec calib vectorizer", X_calib_vectors.shape)

with open('X_train_W2V.pkl', 'wb') as file:
    pickle.dump(X_train_vectors, file)

with open('y_train_W2V.pkl', 'wb') as file:
    pickle.dump(y_train, file)
    
with open('X_test_W2V.pkl', 'wb') as file:
    pickle.dump(X_test_vectors, file)
    
with open('y_test_W2V.pkl', 'wb') as file:
    pickle.dump(y_test, file)
    
with open('X_calib_W2V.pkl', 'wb') as file:
    pickle.dump(X_calib_vectors, file)
    
with open('y_calib_W2V.pkl', 'wb') as file:
    pickle.dump(y_calib, file)


"""
TFIDF weighted W2V 

"""
word2vec_corpus = []
for sentence in X_train:
    word2vec_corpus.append(sentence.split())

word2vec_model = Word2Vec(sentences=word2vec_corpus, size=200, min_count=5, workers=8)
word2vec_words = list(word2vec_model.wv.vocab)

tf_idf_object = TfidfVectorizer(ngram_range=(1,1)).fit(X_train)

def vectorize_tfidf_w2v(dataset, tf_idf_object, word2vec_model, word2vec_words):
    
    word2vec_corpus = []
    for sentence in dataset:
        word2vec_corpus.append(sentence.split())
        
    tf_idf_matrix = tf_idf_object.transform(dataset)
    tfidf_features = tf_idf_object.get_feature_names()
    
    # Buld a dictionary with words as a key, and the idfs as value
    dictionary = dict(zip(tf_idf_object.get_feature_names(), list(tf_idf_object.idf_)))
    
    # Algorithm for finding the TF-IDF weighted average word2vec vectors 
    tfidf_sent_vectors = []
    row = 0
    for sentence in tqdm(word2vec_corpus):
        sent_vec = np.zeros(200)
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
