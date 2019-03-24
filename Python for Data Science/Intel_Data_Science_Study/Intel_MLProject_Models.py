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
sns.set(font_scale = 1.5)

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

def mae(y_true, y_pred):
    return np.mean(abs(y_true-y_pred))

def fit_and_evaluate(model):
    model.fit(X,y)
    model_pred = model.predict(X_test)
    model_mae = mae(y_test, model_pred)
    return model_mae

lr = LinearRegression()
lr_mae = fit_and_evaluate(lr)
print('Linear Regression Performance on the test set: MAE = %0.4f' % lr_mae)

svm = SVR(C=1000, gamma = 0.1)
svm_mae = fit_and_evaluate(svm)
print('Support Vector Machine Regression Performance on the test set: MAE = %0.4f' 
      % svm_mae)

random_forest = RandomForestRegressor(random_state=60)
random_forest_mae = fit_and_evaluate(random_forest)
print('Random Forest Regression Performance on the test set: MAE = %0.4f' 
      % random_forest_mae)

gradient_boosted = GradientBoostingRegressor(random_state=60)
gradient_boosted_mae = fit_and_evaluate(gradient_boosted)
print('Gradient Boosted Regression Performance on the test set: MAE = %0.4f'
      % gradient_boosted_mae)

knn = KNeighborsRegressor(n_neighbors=10)
knn_mae = fit_and_evaluate(knn)
print('K-Nearest Neighbors Regression Performance on the test set: MAE = %0.4f'
      % knn_mae)

model_comparison = pd.DataFrame({'model': ['Linear Regression', 'Support Vector Machine', 
                                           'Random Forest', 'Gradient Boosted',
                                           'K-Nearest Neighbors'],
                                 'mae': [lr_mae, svm_mae, random_forest_mae, 
                                         gradient_boosted_mae, knn_mae]})
model_comparison = model_comparison.sort_values('mae', ascending=False)

f, ax = plt.subplots(figsize=(12,8))
sns.barplot(x="mae", y="model", data=model_comparison, palette='Blues_d', alpha=1)
ax.set_xlabel('Mean Absolute Error', fontsize=14)
ax.set_ylabel('Model', fontsize=14)
ax.set_title('Model Comparison on Test MAE', fontsize=20)


###############################################################################
############################# Model Optimization ##############################
###############################################################################

# Loss function to be optimized
loss = ['ls', 'lad', 'huber']

# Number of trees used in the boosting process
n_estimators = [100, 500, 900, 1100, 1500]

# Maximum depth of each tree
max_depth = [2, 3, 5, 10, 15]

# Minimum number of samples per leaf 
min_samples_leaf = [1, 2, 4, 6, 8]

# Minimum number of samples to split a node 
min_samples_split = [2, 4, 6, 10]

# Maximum number of features to consider for making splits 
max_features = ['auto', 'sqrt', 'log2', None]

# Define the grid of hyperparameters to search 
hyperparameter_grid = {'loss': loss,
                       'n_estimators': n_estimators,
                       'max_depth': max_depth, 
                       'min_samples_leaf': min_samples_leaf,
                       'min_samples_split': min_samples_split,
                       'max_features': max_features}
model = GradientBoostingRegressor(random_state=42)
# Set up the random search with 4-fold cross validation
random_cv = RandomizedSearchCV(estimator=model, 
                               param_distributions=hyperparameter_grid,
                               cv=4, n_iter=25, 
                               scoring = 'neg_mean_absolute_error',
                               n_jobs = -1, verbose = 1, 
                               return_train_score = True, random_state=42)
random_cv.fit(X,y)

random_results = pd.DataFrame(random_cv.cv_results_).sort_values('mean_test_score',
                             ascending = False)
random_results.head(10)
random_cv.best_estimator_

# Create a range of trees to evaluate 
trees_grid = {'n_estimators':[100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 
                              650, 700, 750, 800]}
model = GradientBoostingRegressor(loss = 'lad', max_depth = 5, min_samples_leaf = 6,
                                  min_samples_split = 6, max_features = None, 
                                  random_state = 42)
grid_search = GridSearchCV(estimator = model, param_grid=trees_grid, cv = 4, 
                           scoring = 'neg_mean_absolute_error', verbose = 1, 
                           n_jobs = -1, return_train_score = True)
grid_search.fit(X,y)

results_grid_search = pd.DataFrame(grid_search.cv_results_)
f, ax = plt.subplots(figsize=(12,8))
plt.style.use('fivethirtyeight')
plt.plot(results_grid_search['param_n_estimators'], 
         -1 * results_grid_search['mean_test_score'], label = 'Testing Error')
plt.plot(results_grid_search['param_n_estimators'],
         -1 * results_grid_search['mean_train_score'], label = 'Training Error')
plt.legend()
ax.set_xlabel('Number of Tree', fontsize=20)
ax.set_ylabel('Mean Absolute Error', fontsize=20)
ax.set_title('Performance & Number of Trees', fontsize=25)


###############################################################################
############## Evaluate Final Model on the Test Set ###########################
###############################################################################

default_model = GradientBoostingRegressor(random_state = 42)
final_model = grid_search.best_estimator_
final_model

default_model.fit(X,y)
final_model.fit(X,y)
default_pred = default_model.predict(X_test)
final_pred = final_model.predict(X_test)
print('Default model performance on the test set: MAE = %0.4f' % mae(y_test, default_pred))
print('Final model performance on the test set: MAE = %0.4f' % mae(y_test, final_pred))

f, ax = plt.subplots(figsize=(12,8))
sns.kdeplot(final_pred, label='Predictions', shade=True, alpha=0.25)
sns.kdeplot(y_test, label='Values', shade=True, alpha=0.15)
ax.set_xlabel('Energy Star Score', fontsize=14)
ax.set_ylabel('Density', fontsize=14)
ax.set_title('Test Values and Predictions', fontsize=20)

f, ax = plt.subplots(figsize=(12,8))
residuals = final_pred - y_test
sns.distplot(residuals,bins=50, kde_kws={"color":"r", "lw":3, "label":"KDE"},
             hist_kws={"alpha":0.5, "color":"g"})
ax.set_xlabel('Error', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
ax.set_title('Distribution of Residuals', fontsize=20)
