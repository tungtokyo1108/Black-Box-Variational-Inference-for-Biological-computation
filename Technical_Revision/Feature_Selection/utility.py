#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 11:51:50 2019

@author: tungutokyo
"""

import numpy as np
import pandas as pd
from numpy import linalg as LA


###############################################################################
################################## Function ###################################
###############################################################################

def feature_ranking(W):
    
    """
    Ranking features according to the feature weights matrix W
    """
    T = (W*W).sum(1)
    idx = np.argsort(T,0)
    return idx[::-1]
    
def generate_diagonal_matrix(U):
    
    temp = np.sqrt(np.multiply(U,U).sum(1))
    temp[temp < 1e-16] = 1e-16
    temp = 0.5/temp
    D = np.diag(temp)
    return D
    
def calculate_l21_norm(X):
    return (np.sqrt(np.multiply(X,X).sum(1))).sum()
    
def construct_label_matrix(label_1D):
    
    label = np.array(label_1D)
    if label.ndim != 1:
        label = label.ravel()
        
    n_sample = label.shape[0]
    unique_label = np.unique(label)
    n_classes = unique_label.shape[0]
    label_matrix = np.zeros((n_sample, n_classes))
    for i in range(n_classes):
        label_matrix[label == unique_label[i], i] = 1
    
    return label_matrix

###############################################################################
############################# Test Function ###################################
###############################################################################
    
features = pd.read_csv("Data_test/Clinical_features.csv")
categories = pd.read_csv("Data_test/Clinical_categories.csv")

    
test_label_matrix = construct_label_matrix(categories)












































