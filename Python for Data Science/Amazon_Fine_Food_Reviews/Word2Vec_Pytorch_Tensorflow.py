#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 23:15:09 2019

Reference: 
    - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
    - https://towardsdatascience.com/learn-word2vec-by-implementing-it-in-tensorflow-45641adaf2ac
    - https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
    - https://srijithr.gitlab.io/post/word2vec/

@author: tungutokyo
"""
import sqlite3
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

import argparse
import collections
import math
import os
import random
import sys
from tempfile import gettempdir
import zipfile

###############################################################################
############## Loading the Amazon dataset after cleaning ######################
###############################################################################

connection_sqlobject = sqlite3.connect("sampled_dataset_all_reviews.sqlite")
sampled_dataset = pd.read_sql_query(""" SELECT * FROM Reviews """, connection_sqlobject)

X = sampled_dataset["CleanedText"].values

split = 50000
X_train = X[0:split,]
X_test = X[split:100000]
y_test = y[split:100000]

corpus_data = []
for sentence in X_train:
    corpus_data.append(sentence.split())
print("The size of the corpus data: {}".format(len(corpus_data)))

###############################################################################
################ Word2Vec implemented by Tensorflow ###########################
###############################################################################

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

vocabulary_size = 50000
data_index = 0

def build_dataset(words, n_words):
    
    """ 
    Build the dictionary and replace rare words with UNK token 
    
    Outputs:
        data               - list of codes (integers from 0 to vocabulary_size - 1). 
                             This is original text but words are replaced by their codes
        count              - map of words (string) to count of occurences
        dictionary         - map of words (string) to their codes (integers)
        reverse_dictionary - map codes (integer) to words (string)
    
    """
    
    count = [["UNK", -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = {}
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = []
    unk_count = 0
    for word in words:
        index = dictionary.get(word,0)
        if index == 0:
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

data, count, unused_dictionary, reversed_dictionary = build_dataset(X_train, vocabulary_size)
del X_train 

print("-"*100)
print("The most common word (+UNK): \n", count[:5])
print("-"*100)
print("The sample data: \n", data[:10], [reversed_dictionary[i] for i in data[:10]])

def generate_batch(data, batch_size, num_skips, skip_window):
    
    """ Function to generate a training batch for the skip-gram model """
    
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    
    if data_index + span > len(data):
        data_index = 0
    
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
        
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

batch, labels = generate_batch(data, batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reversed_dictionary[batch[i]], "->", labels[i, 0],
          reversed_dictionary[labels[i, 0]])
    
""" Build and train a skip-gram model """    

batch_size = 128       
embedding_size = 128    # Dimension of embedding vector
skip_window = 1         # How many words to consider left and right
num_skips = 2           # How many times to reuse an input to generate a label
num_sampled = 64        # Number of negative examples to sample

valid_size = 16         # Random set of words in the head of the distribution
valid_window = 100      # Only pick dev samples in the head of the distribution
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

graph = tf.Graph()

with graph.as_default():
    with tf.name_scope("input"):
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        
    with tf.device("/cpu:0"):
        # Look up embeddings for inputs
        with tf.name_scope("embeddings"):
            embeddings = tf.Variable(
                    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
            
        # Construct the variables for NCE loss 
        with tf.name_scope("weights"):
            nce_weights = tf.Variable(
                    tf.truncated_normal([vocabulary_size, embedding_size], 
                                        stddev=1.0 / math.sqrt(embedding_size)))
            
        with tf.name_scope("biases"):
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
            
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(
                tf.nn.nce_loss(
                        weights = nce_weights,
                        biases = nce_biases,
                        labels = train_labels, 
                        inputs = embed,
                        num_sampled = num_sampled, 
                        num_classes = vocabulary_size))
    
    tf.summary.scalar("loss", loss)
    
    with tf.name_scope("optimizer"):
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
        
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims = True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)
    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
        
""" Training model """

num_steps = 100001

with tf.compat.v1.Session(graph=graph) as session:
    
    # Initialize all variables before we use them 
    init.run()
    print("Initialized")
    
    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(data, batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        
        # Define metadata variable
        run_metadata = tf.RunMetadata()
        
        # Perform one update step by evaluating the optimizer op
        # Evaluate the merged op to get all summaries from the returned "summary" variable
        # Feed metadata variable to session for visualizing the graph in TensorBoard 
        _, summary, loss_val = session.run([optimizer, merged, loss], 
                                           feed_dict=feed_dict, 
                                           run_metadata=run_metadata)
        average_loss += loss_val
        
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 200
            print("Average loss at step", step, ": ", average_loss)
            average_loss = 0
            
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reversed_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i,:]).argsort()[1:top_k + 1]
                log_str = "Nearest to %s: " % valid_word
                for k in range(top_k):
                    close_word = reversed_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()
        
"""" Visualize the embeddings """"
    
df = final_embeddings[0:5000,]
df = pd.DataFrame(df)
y = sampled_dataset['SentimentPolarity']
y_train = y[0:5000,]

from MulticoreTSNE import MulticoreTSNE as TSNE

def tsne(dataset, labels, perplexity):
    model = TSNE(n_components=2, random_state=0, n_jobs=8, perplexity=perplexity, n_iter=5000)
    tsne_data = model.fit_transform(dataset)
    tsne_data = np.vstack((tsne_data.T, labels)).T
    tsne_df = pd.DataFrame(data=tsne_data, columns=("Dimension 1", "Dimension 2", "labels"))
    print("T-SNE plot for perplexity = {}".format(perplexity))
    return tsne_df

def tsne_plot(dataset, labels, perplexity):
    sns.FacetGrid(dataset, hue="labels", hue_order=["Positive", "Negative"], size=8).map(
            sns.scatterplot, "Dimension 1", "Dimension 2", edgecolor="w").add_legend()
    plt.title("T-SNE with perplexity = {} and n_iter = 5000".format(perplexity), fontsize=15)
    plt.show()

tsne_30 = tsne(df, y_train, 30)
tsne_plot(tsne_30, y_train, 30)


###############################################################################
################ Word2Vec implemented by Pytorch ###########################
###############################################################################



















































