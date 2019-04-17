#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:26:25 2019

Reference: 
    https://harvard-iacs.github.io/2018-CS109A/lectures/lecture-15/demo/
    https://github.com/Harvard-IACS/2018-CS109A/tree/master/content/lectures/lecture15/presentation       
    https://github.com/PacktPublishing/Artificial-Intelligence-with-Python/blob/master/Chapter%2003/code/decision_trees.py
    
@author: tungutokyo
"""
import pandas as pd 
import sys
import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import tree

import seaborn as sns
pd.set_option('display.width', 1500)
pd.set_option('display.max_columns', 100)

sns.set_context('poster')
sns.set(style="ticks")

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
import statsmodels.api as sm
from statsmodels.tools import add_constant
from statsmodels.regression.linear_model import RegressionResults
from sklearn.decomposition import PCA
from sklearn import ensemble 

"""
fit_and_plot_dt

Fit decision tree with on given data set with given depth, and plot the data/model
Input: 
    fname (string containing file name)
    depth (depth of tree)
"""
def fit_and_plot_dt(x, y ,depth, title, ax, plot_data=True, fill=True, color='Blues'):
    dt = tree.DecisionTreeClassifier(max_depth=depth)
    dt.fit(x,y)
    
    ax = plot_tree_boundary(x, y, dt, title, ax, plot_data, fill, color)
    return ax

"""
plot_tree_boundary

A funtion that visualizes the data and the decison boundaries 
Input:
    x - predictors 
    y - target
    model - the classifier you want to visualize 
    title - title for plot 
    ax - a set of axes to plot on 
Return:
    ax - with data and decision boundaries 
"""

def plot_tree_boundary(x, y, model, title, ax, plot_data=True, fill=True, color='Greens'):
    if plot_data:
        ax.scatter(x[y==1,0], x[y==1,1], c='green')
        ax.scatter(x[y==0,0], x[y==0,1], c='grey')
 
    # Create Mesh    
    interval = np.arange(min(x.min(), y.min()), max(x.max(), y.max()), 0.01)
    n = np.size(interval)
    x1, x2 = np.meshgrid(interval, interval)
    x1 = x1.reshape(-1,1)
    x2 = x2.reshape(-1,1)
    xx = np.concatenate((x1,x2), axis=1)
    
    yy = model.predict(xx)
    yy = yy.reshape((n,n))
    
    x1 = x1.reshape(n,n)
    x2 = x2.reshape(n,n)
    if fill:
        ax.contourf(x1, x2, yy, alpha=0.1, cmap=color)
    else:
        ax.contour(x1, x2, yy, alpha=0.1, cmap=color)
    
    ax.set_title(title)
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    
    return ax 

npoints = 200
# Sample two independent N(0.5^2)
data = np.random.multivariate_normal([0,0], np.eye(2)*5, size=npoints)
data = np.hstack((data, np.zeros((npoints, 1))))

data[data[:, 0]**2 + data[:,1]**2 < 3**2, 2] = np.random.choice([0,1], len(data[data[:, 0]**2 + data[:,1]**2 < 3**2]), 
                                                                    p=[0.2, 0.8])
fig, ax = plt.subplots(1,1, figsize=(10,8))
x = data[:, :-1]
y = data[:, -1]
ax.scatter(x[y==1, 0], x[y==1, 1], c='green', label='vegetation')
ax.scatter(x[y==0, 0], x[y==0, 1], c='black', label='non vegetation', alpha=0.25)
ax.set_xlabel('longitude')
ax.set_ylabel('latitude')
ax.set_title('satellite image')
ax.legend()
plt.tight_layout()
plt.show()

depths = [1, 2, 5, 1000]
fig, ax = plt.subplots(1, len(depths), figsize=(20,4))
x = data[:, :-1]
y = data[:, -1]
ind = 0
for i in depths:
    ax[ind] = fit_and_plot_dt(x,y,i,'Depth {}'.format(i), ax[ind])
    ax[ind].set_xlim(-6,6)
    ax[ind].set_ylim(-6,6)
    ind += 1


















































