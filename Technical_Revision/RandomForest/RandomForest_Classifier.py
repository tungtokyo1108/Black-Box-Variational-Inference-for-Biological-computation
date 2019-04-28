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
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
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

def missing_values_table(data):
    mis_val = data.isnull().sum()
    mis_val_percent = 100 * data.isnull().sum() / len(data)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
            columns={0: 'Missing Values', 1: '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
            '% of Total Values', ascending=False).round(1)
    print("Your selected dataframe has " + str(data.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) + 
          " columns that have missing values.")
    return mis_val_table_ren_columns
missing_values_table(spam_df)


###############################################################################
########## Fitting an Optimal Single Decision Tree by Depth ###################
###############################################################################

depth, tree_start, tree_end = {}, 3, 20
for i in range(tree_start, tree_end):
    model = DecisionTreeClassifier(max_depth=i)
    scores = cross_val_score(estimator=model, X=Xtrain, y=ytrain, cv=5, n_jobs=-1)
    depth[i] = scores.mean()
    
lists = sorted(depth.items())
x, y = zip(*lists)
plt.figure(figsize=(12,12))
plt.ylabel("Cross Validation Accuracy", fontsize=15)
plt.xlabel("Maximum Depth", fontsize=15)
plt.title('Variation of Accuracy with Depth - Simple Decision Tree', fontsize=20)
plt.plot(x, y, 'b-', marker='o')
plt.show()

best_depth = sorted(depth, key=depth.get, reverse=True)[0]
print("The best depth was found to be: ", best_depth)

model = DecisionTreeClassifier(max_depth=best_depth)
model.fit(Xtrain, ytrain)

print("Accuracy, Training Set: {:.2%}".format(accuracy_score(ytrain, model.predict(Xtrain))))
print("Accuracy, Testing Set: {:.2%}".format(accuracy_score(ytest, model.predict(Xtest))))

pd.crosstab(ytest, model.predict(Xtest), margins=True, rownames=['Actual'], colnames=['Predicted'])

###############################################################################
###################### Random Forest and Bagging ##############################
###############################################################################

model = RandomForestClassifier(n_estimators=int(Xtrain.shape[1]/2), max_depth=best_depth)
model.fit(Xtrain, ytrain)

y_pred_train = model.predict(Xtrain)
y_pred_test = model.predict(Xtest)

train_score = accuracy_score(ytrain, y_pred_train)*100
test_score = accuracy_score(ytest, y_pred_test)*100

print("Accuracy, Training Set :", str(train_score)+'%')
print("Accuracy, Testing Set :", str(test_score)+'%')

feature_importance = model.feature_importances_
feature_importance = 100*(feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(12,12))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, Xtrain.columns[sorted_idx], fontsize=10)
plt.xlabel('Relative Importance', fontsize=15)
plt.title('Variable Importance', fontsize=20)

###############################################################################
###################### Compare with other models ##############################
###############################################################################

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf = LogisticRegression(solver="lbfgs", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=int(Xtrain.shape[1]/2), max_depth=best_depth, random_state=42)
svm_clf = SVC(gamma="scale", random_state=42)

voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='hard')
voting_clf.fit(Xtrain, ytrain)

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(Xtrain, ytrain)
    y_pred = clf.predict(Xtest)
    print(clf.__class__.__name__, accuracy_score(ytest, y_pred))
    
