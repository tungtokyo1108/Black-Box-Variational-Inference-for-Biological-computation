# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 09:59:45 2019

Reference: https://github.com/WillKoehrsen/machine-learning-project-walkthrough/blob/master/Machine%20Learning%20Project%20Part%202.ipynb

@author: Tung1108
"""

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 60)

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

from IPython.core.pylabtools import figsize

import seaborn as sns
sns.set(font_scale = 2)

from sklearn.preprocessing import Imputer, MinMaxScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

###############################################################################
############################### Read in Data ##################################
###############################################################################

train_features = pd.read_csv('Energy_and_Water_Data_training_features.csv')
test_features = pd.read_csv('Energy_and_Water_Data_testing_features.csv')
train_labels = pd.read_csv('Energy_and_Water_Data_training_label.csv')
test_labels = pd.read_csv('Energy_and_Water_Data_testing_label.csv')

print('Training Feature Size: ', train_features.shape)
print('Testing Feature Size: ', test_features.shape)
print('Training Labels Size: ', train_labels.shape)
print('Testing Labels Size: ', test_labels.shape)

f, ax = plt.subplots(figsize=(12,8))
sns.distplot(train_labels['score'].dropna(), bins=50, 
             kde_kws={"color":"g", "lw": 3, "label": "KDE"},
             hist_kws={"linewidth": 3, "alpha":0.5, "color":"b"})
ax.set_xlabel('Score', fontsize=14)
ax.set_ylabel('Frequency', fontsize=14)
ax.set_title('Energy Star Score Distribution', fontsize=20)


###############################################################################
############# Evaluating and Comparing Machine Learning Model #################
###############################################################################

# Imputing missing values 
imputer = Imputer(strategy='median')
imputer.fit(train_features)
X = imputer.transform(train_features)
X_test = imputer.transform(test_features)
print('Missing values in training features: ', np.sum(np.isnan(X)))
print('Missing values in testing features: ', np.sum(np.isnan(X_test)))

# Make sure all values are finite 
print(np.where(~np.isfinite(X)))
print(np.where(~np.isfinite(X_test)))

# Scaling Features
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(X)
X = scaler.transform(X)
X_test = scaler.transform(X_test)

y = np.array(train_labels).reshape((-1, ))
y_test = np.array(test_labels).reshape((-1, ))


###############################################################################
############################# Models to Evaluate ##############################
###############################################################################

































