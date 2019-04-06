# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 09:08:08 2019

Reference: https://github.com/raviolli77/machineLearning_breastCancer_Python/blob/master/notebooks/02_random_forest.md

@author: Tung1108
"""

import time
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, MinMaxScaler
from sklearn.metrics import roc_curve, auc 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from urllib.request import urlopen 

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
print("Here is the data types of our columns: \n", train_features.dtypes)

###############################################################################
############################### Class Imbalance ###############################
###############################################################################

def print_target_perc(data_frame, col):
    """ Function used to print class distribution for our data set """
    try:
        # If the number of unique instances in column exceeds 20 print warning 
        if data_frame[col].nunique() > 20:
            return print('Warning: There are {0} values in {1} column which exceed the max of 20 \
                         Please try a column with lower value counts!'.format(data_frame[col].nunique(), col))
        # Stores value counts
        col_vals = data_frame[col].value_counts().sort_values(ascending=False)
        # Reset index to make index a column in data frame
        col_vals = col_vals.reset_index()
        
        # Convert to output the percentage 
        f = lambda x, y: 100 * (x / sum(y))
        for i in range(0, len(col_vals['index'])):
            print('{0} accounts for {1:.2f}% of the {2} column'.format(col_vals['index'][i], 
                          f(col_vals[col].iloc[i], col_vals[col]), col))
    except KeyError as e:
        raise KeyError('{0}: Not found. Please choose the right column name!'.format(e))

print_target_perc(train_features, 'Largest Property Use Type_Multifamily Housing')

###############################################################################
#################### Creating Training and Test Set ###########################
###############################################################################

train_features_space = train_features.iloc[:, train_features.columns != 'Largest Property Use Type_Multifamily Housing']
train_features_class = train_features.iloc[:, train_features.columns == 'Largest Property Use Type_Multifamily Housing']

training_set, test_set, class_set, test_class_set = train_test_split(train_features_space, 
                                train_features_class, test_size=0.20, random_state=42)
imputer = Imputer(strategy='median')
imputer.fit(training_set)
X = imputer.transform(training_set)
X_test = imputer.transform(test_set)
print('Missing values in training features: ', np.sum(np.isnan(X)))
print('Missing values in testing features: ', np.sum(np.isnan(X_test)))


###############################################################################
############################ Fitting Random Forest ############################
###############################################################################

fit_rf = RandomForestClassifier(random_state=42)


###############################################################################
############################ Fitting Random Forest ############################
###############################################################################

np.random.seed(42)
start = time.time()
param_dist = {'max_depth': [2,3,4],
              'bootstrap': [True, False],
              'max_features': ['auto', 'sqrt', 'log2', None], 
              'criterion': ['gini', 'entropy']}
cv_rf = GridSearchCV(fit_rf, cv=10, param_grid=param_dist, n_jobs=3)
cv_rf.fit(X, class_set)
print('Best Parameters using grid search: \n', cv_rf.best_params_)
end = time.time()
print('Time taken in grid search: {0: .2f}'.format(end-start))

fit_rf.set_params(criterion = 'gini', max_features = 'log2', max_depth = 3)

###############################################################################
############################ Out of Bag Error Rate ############################
###############################################################################

fit_rf.set_params(warm_start=True, oob_score=True)
min_estimators = 15
max_estimators = 1000
error_rate = {}
for i in range(min_estimators, max_estimators+1, 5):
    fit_rf.set_params(n_estimators=i)
    fit_rf.fit(X, class_set)
    oob_error = 1 - fit_rf.oob_score_
    error_rate[i] = oob_error
oob_series = pd.Series(error_rate)

fig, ax = plt.subplots(figsize=(10,10))
ax.set_facecolor('#fafafa')
oob_series.plot(kind='line', color='red')
plt.axhline(0.162, color='#875FDB', linestyle='--')
plt.axhline(0.154, color='#875FDB', linestyle='--')
plt.xlabel('n_estimators', fontsize=15)
plt.ylabel('OOB Error Rate', fontsize=15)
plt.title('OOB Error Rate Across various Forest Sizes', fontsize=20)

print('OOB Error rate for 400 tree is: {0:.5f}'.format(oob_series[400]))
