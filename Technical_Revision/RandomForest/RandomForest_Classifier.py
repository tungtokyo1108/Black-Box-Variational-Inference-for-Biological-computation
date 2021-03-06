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
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
from tqdm import tqdm
from sklearn.utils.multiclass import unique_labels

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
y_score_test = model.decision_fuction(Xtest)

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
    
###############################################################################
########################## Bagging ensembles ##################################
###############################################################################

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier(random_state=42, max_depth=best_depth)
tree_clf.fit(Xtrain, ytrain)
y_tree_pred = tree_clf.predict(Xtest)
print("Decision Tree: ", accuracy_score(ytest, y_tree_pred))

bag_clf = BaggingClassifier(
        DecisionTreeClassifier(random_state=42, max_depth=best_depth), n_estimators=int(Xtrain.shape[1]/2), 
        max_samples=100, bootstrap=True, random_state=42)
bag_clf.fit(Xtrain, ytrain)
y_pred = bag_clf.predict(Xtest)
print("Decision Tree + Bagging: ", accuracy_score(ytest,y_pred))

bag_outof_clf = BaggingClassifier(
        DecisionTreeClassifier(random_state=42, max_depth=best_depth), n_estimators=int(Xtrain.shape[1]/2),
        max_samples=100, oob_score=True, random_state=40)
bag_outof_clf.fit(Xtrain,ytrain)
y_outof_pred = bag_outof_clf.predict(Xtest)
print("Decision Tree + Bagging + Out-of-Bagging: ", accuracy_score(ytest, y_outof_pred))

rnd_clf = RandomForestClassifier(n_estimators=int(Xtrain.shape[1]/2), max_depth=best_depth, random_state=42)
rnd_clf.fit(Xtrain, ytrain)
y_rnd_pred = rnd_clf.predict(Xtest)
print("Random Forest: ", accuracy_score(ytest,y_rnd_pred))

###############################################################################
############################# Adaboost Model ##################################
###############################################################################

model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5, random_state=42),
                           n_estimators=100, learning_rate=0.05)
model.fit(Xtrain, ytrain)

y_Ada_pred_train = model.predict(Xtrain)
y_Ada_pred_test = model.predict(Xtest)

train_score = accuracy_score(ytrain, y_Ada_pred_train)*100
test_score = accuracy_score(ytest, y_Ada_pred_test)*100

print("Accuracy, Training Set :", str(train_score)+'%')
print("Accuracy, Testing Set :", str(test_score)+'%')

train_scores = list(model.staged_score(Xtrain, ytrain))
test_scores = list(model.staged_score(Xtest, ytest))

train_scores = list(model.staged_score(Xtrain, ytrain))
test_scores = list(model.staged_score(Xtest, ytest))
plt.figure(figsize=(12,12))
plt.plot(train_scores, label='train')
plt.plot(test_scores, label='test')
plt.xlabel('Iteration', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.title("Variation of Accuracy with Iterations", fontsize=20)
plt.legend()

score_train, score_test, depth_start, depth_end = {}, {}, 2, 20
for i in tqdm(range(depth_start, depth_end)):
    model = AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=i),
            n_estimators=100, learning_rate=0.05)
    model.fit(Xtrain, ytrain)
    score_train[i] = accuracy_score(ytrain, model.predict(Xtrain))
    score_test[i] = accuracy_score(ytest, model.predict(Xtest))

lists1 = sorted(score_train.items())
lists2 = sorted(score_test.items())
x1, y1 = zip(*lists1)
x2, y2 = zip(*lists2)
plt.figure(figsize=(12,12))
plt.ylabel("Accuracy", fontsize=15)
plt.xlabel("Depth", fontsize=15)
plt.title('Variation of Accuracy with Depth - Adaboost Classifier', fontsize=20)
plt.plot(x1, y1, 'b-', label='Train')
plt.plot(x2, y2, 'g-', label='Test')
plt.legend()
plt.show()

###############################################################################
################################ Light GBM ####################################
###############################################################################

features = [column for column in spam_df.columns if column not in ['Spam']]
target = spam_df['Spam']

param = {
        'bagging_freq': 5,
        'bagging_fraction': 0.4,
        'boost_from_average': 'false',
        'boost': 'gbdt',
        'feature_fraction': 0.05,
        'learning_rate': 0.01,
        'max_depth': -1,
        'metric': 'auc',
        'min_data_in_leaf': 80,
        'min_sum_hessian_in_leaf': 10.0,
        'num_leaves': 13,
        'num_threads': 8,
        'tree_learner': 'serial',
        'objective': 'binary',
        'verbosity': 1
        }
folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=42)
oof = np.zeros(len(spam_df))
predictions = np.zeros(len(Xtest))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(predictors.values, response.values)):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(spam_df.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(spam_df.iloc[val_idx][features], label=target.iloc[val_idx])
    num_round = 1000000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=1000, early_stopping_rounds=3000)
    oof[val_idx] = clf.predict(spam_df.iloc[val_idx][features], num_iteration=clf.best_iteration)
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    predictions += clf.predict(Xtest, num_iteratoration=clf.best_iteration) / folds.n_splits
print("CV score: {:<8.5f}".format(roc_auc_score(target,oof)))
    
cols = (feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False).index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(15,20))
sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance", ascending=False))
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.title('Feature importance (averaged/folds)', fontsize=25)
plt.tight_layout()

###############################################################################
################# Stochastic Gradient Boosting ################################
###############################################################################

model = xgb.XGBClassifier()
subsample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
n_estimators = [50, 100, 150, 200]
max_depth = [2, 4, 6, 8, 10]
# param_grid = dict(subsample=subsample)
# colsample_grid = dict(colsample_bytree=subsample)
param_grid = dict(colsample_bylevel = subsample, max_depth=max_depth, n_estimators=n_estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
# grid_search = GridSearchCV(model, colsample_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(Xtrain, ytrain)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
plt.figure(figsize=(12,12))
plt.errorbar(subsample, means, yerr=stds)
plt.title("XGBoost subsample vs Log Loss", fontsize=20)
plt.xlabel('Subsample', fontsize=15)
plt.ylabel('Log Loss', fontsize=15)

model_best = xgb.XGBClassifier(colsample_bylevel=0.1, max_depth=10, n_estimators=150)
model_best.fit(Xtrain,ytrain)
feature_importance_XGB = model_best.feature_importances_
feature_importance_XGB = 100*(feature_importance_XGB / feature_importance_XGB.max())
sorted_idx_XGB = np.argsort(feature_importance_XGB)
pos_XGB = np.arange(sorted_idx_XGB.shape[0]) + .5

plt.figure(figsize=(12,12))
plt.barh(pos_XGB, feature_importance_XGB[sorted_idx_XGB], align='center')
plt.yticks(pos_XGB, Xtrain.columns[sorted_idx_XGB], fontsize=10) 
plt.xticks(fontsize=15)
plt.xlabel('Relative Importance', fontsize=15)
plt.title('Variable Importance of XGB', fontsize=20)

###############################################################################
############################# Confusion Matrix ################################
###############################################################################

cm = confusion_matrix(ytest, y_pred_test)
cmap = sns.diverging_palette(220, 10, as_cmap=True)
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt='d', cbar=False, cmap=cmap, linewidth=.05)
plt.xlabel('Predicted label', fontsize=10)
plt.ylabel('True lablel', fontsize=10)
plt.title('Confusin Matrix', fontsize=15)

def plot_confusion_matrix(y_true, y_pred, normalize=False,
                          title=None, cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    
    print(cm)
    
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]))
    ax.set_title(title, fontsize=20)
    ax.set_ylabel('True label', fontsize=15)
    ax.set_xlabel('Predicted label', fontsize=15)
    
    # Rotate the tick labels and set their aligment
    plt.setp(ax.get_xticklabels(), ha='right', rotation_mode="anchor")
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max()/2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), 
                    ha = "center", va = "center",
                    color = "white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

np.set_printoptions(precision=2)

plot_confusion_matrix(ytest, y_pred_test, title='Confusion matrix, without normalization')
plot_confusion_matrix(ytest, y_pred_test, normalize=True,
                      title='Normalized confusion matrix')
plt.show()
