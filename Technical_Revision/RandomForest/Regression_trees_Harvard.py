#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 10:23:36 2019

Reference: https://harvard-iacs.github.io/2018-CS109A/labs/lab-9/solutions/
           https://github.com/ageron/handson-ml2/blob/master/06_decision_trees.ipynb

@author: tungutokyo
"""

import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd 
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn.apionly as sns
sns.set_style("whitegrid")
sns.set_context("poster")
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from graphviz import Source
from sklearn.tree import export_graphviz
import os

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "decision_trees"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok = True)

elect_df = pd.read_csv("county_level_election.csv")
elect_df.head()

X = elect_df[['population', 'hispanic', 'minority', 'female', 'unemployed', 'income', 'nodegree', 
              'bachelor', 'inactivity', 'obesity', 'density', 'cancer']]
response = elect_df['votergap']
Xtrain, Xtest, ytrain, ytest = train_test_split(X,response,test_size=0.2)

# General Trees 
x = Xtrain['minority'].values
o = np.argsort(x)
x = x[o]
y = ytrain.values
y = y[o]
fig, ax = plt.subplots(1,2,figsize=(12,6))
ax[0].plot(x,y,'.')
ax[0].set_title('Minority and votergap',fontsize=20)
ax[1].plot(np.log(x),y,'.')
ax[1].set_title('Log-minority and votergap',fontsize=20)

plt.figure(figsize=(12,12))
sns.stripplot(np.log(x),y, jitter=0.25, size=8, linewidth=.5, alpha=0.5, edgecolor="gray")
plt.title('Jittering with stripplot', fontsize=20)

plt.figure(figsize=(12,12))
sns.jointplot(np.log(x),y,kind="reg",space=0,color="g",height=10)
plt.xlabel('Log-minority')
plt.ylabel('Votergrap')

fig, ax = plt.subplots(figsize=(12,12))
ax.plot(np.log(x),y,'.', alpha=0.5)
xx = np.log(x).reshape(-1,1)
for i in [1,5,10]:
    dtree = DecisionTreeRegressor(max_depth=i)
    dtree.fit(xx,y)
    ax.plot(np.log(x), dtree.predict(xx), label=str(i), alpha=0.8, lw=4)
plt.legend()

# The maximum depth of the tree

tree_reg1 = DecisionTreeRegressor(random_state=42, max_depth=2)
tree_reg2 = DecisionTreeRegressor(random_state=42, max_depth=3)
tree_reg1.fit(xx,y)
tree_reg2.fit(xx,y)

def plot_regression_predictions(tree_reg, X, y, axes=[-1,5,-80,100],ylabel="$y$"):
    # x1 = np.linspace(axes[0],axes[1],500).reshape(-1,1)
    y_pred = tree_reg.predict(X)
    plt.axis(axes)
    plt.xlabel("$x_1$", fontsize=18)
    if ylabel:
        plt.ylabel(ylabel, fontsize=18, rotation=0)
    plt.plot(X,y,"b.")
    plt.plot(X, y_pred, "r.-", linewidth=2, label=r"$\hat{y}$")

export_graphviz(tree_reg1, out_file=os.path.join(IMAGES_PATH, "regression_tree.dot"),
                feature_names = ["x1"], rounded=True, filled=True)
Source.from_file(os.path.join(IMAGES_PATH, "regression_tree.dot"))

export_graphviz(tree_reg2, out_file=os.path.join(IMAGES_PATH, "regression_tree.dot"),
                feature_names = ["x2"], rounded=True, filled=True)
Source.from_file(os.path.join(IMAGES_PATH, "regression_tree.dot"))

plt.figure(figsize=(20,10))
plt.subplot(121)
plot_regression_predictions(tree_reg1, xx, y)
for split, style in ((3.629, "k-"), (1.764, "k--"), (3.918, "k--")):
    plt.plot([split, split], [-80,100], style, linewidth=3)
plt.text(2.8, 90, "Depth=0", fontsize=15)
plt.text(1, -70, "Depth=1", fontsize=13)
plt.text(4, 80, "Depth=1", fontsize=13)
plt.legend(loc="upper center", fontsize=18)
plt.title("max_depth=2", fontsize=20)

plt.subplot(122)
plot_regression_predictions(tree_reg2, xx, y, ylabel=True)
for split, style in ((3.517, "k-"), (1.767, "k--"), (3.92, "k--")):
    plt.plot([split, split], [-80,100], style, linewidth=3)
for split in (1.045, 2.446, 3.822, 4.22):
    plt.plot([split, split], [-80,100], "k:", linewidth=2)
plt.text(0.3,-70, "Depth=2", fontsize=13)
plt.title("max_depth=3", fontsize=20)
plt.show()


# The minimum number of samples required to split an internal node
plt.figure(figsize=(12,10))
plt.plot(np.log(x),y,'.')
xx = np.log(x).reshape(-1,1)
for i in [500, 200, 100, 20]:
    dtree = DecisionTreeRegressor(max_depth=6, min_samples_split=i)
    dtree.fit(xx,y)
    plt.plot(np.log(x), dtree.predict(xx), label=str(i), alpha=0.8, lw=4)
plt.legend()


# The minimum number of samples required to be at leaf node
tree_reg1 = DecisionTreeRegressor(random_state=42)
tree_reg2 = DecisionTreeRegressor(random_state=42, min_samples_leaf=10)
tree_reg1.fit(xx,y)
tree_reg2.fit(xx,y)

y_pred1 = tree_reg1.predict(xx)
y_pred2 = tree_reg2.predict(xx)

plt.figure(figsize=(12,6))
plt.subplot(121)
plt.plot(xx,y,"b.")
plt.plot(xx, y_pred1,"r.-", linewidth=2, label=r"$\hat{y}$")
#plt.axis([0,1,-0.2,1.1])
plt.xlabel("Log minority", fontsize=18)
plt.ylabel("Votergrap", fontsize=18, rotation=0)
plt.legend(loc="upper center", fontsize=18)
plt.title("No restrictions", fontsize=14)

plt.subplot(122)
plt.plot(xx,y,"b.")
plt.plot(xx,y_pred2, "r.-", linewidth=2, label=r"$\hat{y}$")
plt.xlabel("Log minority", fontsize=18)
plt.title("min_sample_left={}".format(tree_reg2.min_samples_leaf), fontsize=14)







































