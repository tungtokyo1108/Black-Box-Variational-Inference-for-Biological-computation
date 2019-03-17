# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:14:05 2019

Reference: https://github.com/WillKoehrsen/machine-learning-project-walkthrough/blob/master/Machine%20Learning%20Project%20Part%203.ipynb

@author: Tung1108
"""

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 60)

import matplotlib.pyplot as plt

from IPython.core.pylabtools import figsize

import seaborn as sns
# sns.set(font_scale = 1.5)

from sklearn.preprocessing import Imputer, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import tree

# import lime 
# import lime.lime_tabulars

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


###############################################################################
########################### Recreate Final Model ##############################
###############################################################################

imputer = Imputer(strategy = 'median')
imputer.fit(train_features)

X = imputer.transform(train_features)
X_test = imputer.transform(test_features)

y = np.array(train_labels).reshape((-1,))
y_test = np.array(test_labels).reshape((-1,))

def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))

model = GradientBoostingRegressor(loss='lad', max_depth=5, max_features=None, 
                                  min_samples_leaf=6, min_samples_split=6, 
                                  n_estimators=800, random_state=42)
model.fit(X,y)
model_pred = model.predict(X_test)
print('Final Model Performance on the test set: MAE = %0.4f' % mae(y_test, model_pred))


###############################################################################
########################### Feature Importances ###############################
###############################################################################

features_results = pd.DataFrame({'feature': list(train_features.columns), 
                                'importance': model.feature_importances_})
features_results = features_results.sort_values('importance', 
                        ascending = False).reset_index(drop=True)
features_results.head(10)

f, ax = plt.subplots(figsize=(8,8))
sns.barplot(x='importance', y='feature', data=features_results.loc[:9, :], 
            palette='Blues_d')
ax.set_xlabel('Relative Importance', fontsize=14)
ax.set_title('Feature Importances from Random Forest', fontsize=20)


###############################################################################
################ Feature Importances for Feature Selection ####################
###############################################################################

most_important_features = features_results['feature'][:10]
indices = [list(train_features.columns).index(x) for x in most_important_features]
X_reduced = X[:, indices]
X_test_reduced = X_test[:, indices]
print('Most important training features shape: ', X_reduced.shape)
print('Most important testing features shape: ', X_test_reduced.shape)

lr = LinearRegression()
lr.fit(X,y)
lr_full_pred = lr.predict(X_test)
lr.fit(X_reduced,y)
lr_reduced_pred = lr.predict(X_test_reduced)
print('Linear Regression Full Results: MAE = %0.4f' %mae(y_test, lr_full_pred))
print('Linear Regression Reduced Results: MAE = %0.4f' %mae(y_test, lr_reduced_pred))

model_reduced = GradientBoostingRegressor(loss='lad', max_depth=5, max_features=None, 
                                          min_samples_leaf=6, min_samples_split=6, 
                                          n_estimators=800, random_state=42)
model_reduced.fit(X_reduced,y)
model_reduced_pred = model_reduced.predict(X_test_reduced)
print('Gradient Boosted Reduced Results: MAE = %0.4f' % mae(y_test, model_reduced_pred))


###############################################################################
##################### Examining a Single Decision Tree ###3####################
###############################################################################

single_tree = model_reduced.estimators_[105][0]
tree.export_graphviz(single_tree, out_file = 'tree.dot',
                     rounded = True, 
                     feature_names = most_important_features,
                     filled = True)
single_tree
































