#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 10:05:30 2019

Reference:
    
    https://harvard-iacs.github.io/2018-CS109A/sections/section-8/student/
    https://github.com/ageron/handson-ml2/blob/master/07_ensemble_learning_and_random_forests.ipynb
    https://www.kaggle.com/gpreda/santander-eda-and-prediction
    https://www.kaggle.com/mjbahmani/santander-ml-explainability

@author: tungutokyo
"""

import gc
import os
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from tqdm import tqdm

spam_df = pd.read_csv('spam.csv', header=None)
columns = ["Column_" + str(i+1) for i in range(spam_df.shape[1]-1)] + ['Spam']
spam_df.columns = columns
display(spam_df.head())

predictors = spam_df.drop(['Spam'], axis=1)
response = spam_df['Spam']
Xtrain, Xtest, ytrain, ytest = train_test_split(predictors, response, test_size=0.2)

###############################################################################
############################### Missing Value #################################
###############################################################################


















































