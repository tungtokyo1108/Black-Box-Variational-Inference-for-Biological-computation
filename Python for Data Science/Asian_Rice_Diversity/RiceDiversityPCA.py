#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 13:46:26 2019

@author: tungutokyo
"""

import pandas as pd
import sys 
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import statsmodels.api as sm 
from statsmodels.regression.linear_model import RegressionResults
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
sns.set_context('poster')
sns.set(style="ticks")

###############################################################################

pheno = pd.read_csv('RiceDiversityPheno.csv')
pheno.info()
pheno_des = pheno.describe()

line = pd.read_csv('RiceDiversityLine.csv')
data = pd.concat([pheno, line], axis=1, sort=False)

###############################################################################
############################## 3D Dimensions ##################################
###############################################################################

Dimension_3d = ['Seed length', 'Seed width', 'Seed volume']
data_3d = data[Dimension_3d]

pca2 = PCA(n_components=2)
data_2d = pca2.fit_transform(data_3d.dropna())

# Recover the 3D points projected on the plane(PCA 2D subspace)
data_3d_inv = pca2.inverse_transform(data_2d)

# Computation of the reconstruction error
np.mean(np.sum(np.square(data_3d_inv - data_3d.dropna()), axis=1))

# Access to the principal components and the explained variance ratio 
pca2.components_
pca2.explained_variance_ratio_

# By projecting down to 2D, what is percent of the variance we lost
1 - pca2.explained_variance_ratio_.sum()

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

axes = [6.5, 14.5, 1, 3.5, 1, 3]
x1s = np.linspace(axes[0], axes[1], 10)
x2s = np.linspace(axes[2], axes[3], 10)
x1, x2 = np.meshgrid(x1s, x2s)

C = pca2.components_
R = C.T.dot(C)
z = (R[0,2] * x1 + R[1,2] * x2) / (1 - R[2,2])

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111, projection='3d')

X = data_3d.dropna().values
#ax.scatter(X[:,0], X[:,1], X[:,2])
X3D_above = X[X[:, 2] > data_3d_inv[:, 2]]
X3D_below = X[X[:, 2] <= data_3d_inv[:, 2]]

# Data points below the surface projection 
ax.plot(X3D_below[:, 0], X3D_below[:, 1], X3D_below[:, 2], "bo", alpha=0.5, color="b")

# The surface projection 
ax.plot_surface(x1,x2,z,alpha=0.1, color="k")
np.linalg.norm(C, axis=0)

for i in range(377):
    if X[i, 2] > data_3d_inv[i, 2]:
        ax.plot([X[i][0], data_3d_inv[i][0]], [X[i][1], data_3d_inv[i][1]], [X[i][2], data_3d_inv[i][2]], "k-", color="r")
    else:
        ax.plot([X[i][0], data_3d_inv[i][0]], [X[i][1], data_3d_inv[i][1]], [X[i][2], data_3d_inv[i][2]], "k-", color="r")

# The points projection         
ax.plot(data_3d_inv[:, 0], data_3d_inv[:, 1], data_3d_inv[:, 2], "k+", color="r", alpha=1)
ax.plot(data_3d_inv[:, 0], data_3d_inv[:, 1], data_3d_inv[:, 2], "k.", color="r", alpha=1)

# Data points above the surface projection
ax.plot(X3D_above[:, 0], X3D_above[:, 1], X3D_above[:, 2], "bo", color="g", alpha=0.5)

ax.set_xlabel("Seed length", fontsize=18, labelpad=10)
ax.set_ylabel("Seed width", fontsize=18, labelpad=10)
ax.set_zlabel("Seed volume", fontsize=18, labelpad=10)

#ax.add_artist(Arrow3D([0, C[0, 0]],[0, C[0, 1]],[0, C[0, 2]], mutation_scale=15, lw=1, arrowstyle="-|>", color="k"))
#ax.add_artist(Arrow3D([0, C[1, 0]],[0, C[1, 1]],[0, C[1, 2]], mutation_scale=15, lw=1, arrowstyle="-|>", color="k"))
#ax.plot([0], [0], [0], "k.")

###############################################################################
############################# Multi-Dimension #################################
###############################################################################

Col_PCA = ['Flag leaf length', 'Flag leaf width', 'Plant height', 'Panicle length', 
            'Seed length', 'Seed width', 'Seed volume', 'Seed surface area', 'Sub-population']
data_PCA_target = data[Col_PCA]
data_PCA = data_PCA_target.dropna().sample(frac=1, random_state=42).reset_index(drop=True)
data_PCA_clear = data_PCA.drop(['Sub-population'], axis=1)

pca_full = PCA()
pca_full.fit(data_PCA_clear)
data_PCA_full_tranf = pca_full.transform(data_PCA_clear)

print('First 4 principal components:\n', pca_full.components_[0:4])
print('EXplained variance ratio:\n', pca_full.explained_variance_ratio_)

plt.figure(figsize=(15,10))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel('Number of component', fontsize=15)
plt.ylabel('Cumulative explained variance', fontsize=15)


pca4 = PCA(n_components=4)
pca4.fit(data_PCA_clear)
data_PCA_tranf = pca4.transform(data_PCA_clear)

# Recover the original data from the projected data 
# There was some loss information during the projection step, 
# let compute the reconstruction error and percent loss of the variance  
data_recov = pca4.inverse_transform(data_PCA_tranf)
print('\nThe reconstruction error: ', np.mean(np.sum(np.square(data_recov - data_PCA_clear), axis=1)))
print('\nExplained variance ratio:\n', pca4.explained_variance_ratio_)
print('\nThe ratio of lost of the variance: ', 1 - pca4.explained_variance_ratio_.sum())

fig, ax = plt.subplots(1, 3, figsize=(20,5))

ax[0].scatter(data_PCA_tranf[:,0], data_PCA_tranf[:,1], color='Blue', alpha=0.2, label='train R^2')
ax[0].set_title('Dimension Reduced Data')
ax[0].set_xlabel('1st Principal Component')
ax[0].set_ylabel('2nd Principal Component')

ax[1].scatter(data_PCA_tranf[:,1], data_PCA_tranf[:,2], color='red', alpha=0.2, label='train R^2')
ax[1].set_title('Dimension Reduced Data')
ax[1].set_xlabel('2nd Principal Component')
ax[1].set_ylabel('3rd Principal Component')

ax[2].scatter(data_PCA_tranf[:,2], data_PCA_tranf[:,3], color='green', alpha=0.2, label='train R^2')
ax[2].set_title('Dimension Reduced Data')
ax[2].set_xlabel('3rd Principal Component')
ax[2].set_ylabel('4th Principal Component')

print('first pca component:\n', pca4.components_[0])
print('\nsecond pca component:\n', pca4.components_[1])
print('\nthrid pca component:\n', pca4.components_[2])
print('\nfourth pca component:\n', pca4.components_[3])

def pca_results(data, pca):
    
    # Dimension indexing 
    dimensions = ['Dimension {}'.format(i) for i in range(1, len(pca.components_)+1)]
    
    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = data.keys())
    components.index = dimensions
    
    # PCA explained variance 
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    variance_ratios.index = dimensions
    
    fig, ax = plt.subplots(figsize=(10,15))
    
    components.plot(ax=ax, kind='barh', fontsize=10)
    ax.set_xlabel("Feature Weights", fontsize=15)
    ax.set_yticklabels(dimensions, rotation=0, fontsize=10)
    
    for i, ev in enumerate(pca.explained_variance_ratio_):
        ax.text(ax.get_xlim()[1] + 0.05, i-0.1, "Explained Variance\n %.4f"%(ev), rotation=-90, fontsize=10)
        
    return pd.concat([variance_ratios, components], axis = 1)

sns.set_context("poster")
pca_results(data_PCA_clear, pca_full)

# Calculation eigenvector
def centerData(X):
    X = X.copy()
    X -= np.mean(X, axis=0)
    return X

data_center_PCA = centerData(data_PCA_clear)

# Eigen Decomposition 
eigVals, eigVecs = np.linalg.eig(data_center_PCA.T.dot(data_center_PCA))
data_center_project_t = np.dot(eigVecs.T, data_center_PCA.T)
data_center_project = data_center_project_t.T

# SVD Decomposition 
# https://github.com/scikit-learn/scikit-learn/blob/7813f7efb/sklearn/decomposition/pca.py#L325
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py
U, S, Vt = sp.linalg.svd(data_center_PCA, full_matrices=False)
V = Vt.T

# Correlation between the original variable and the principal component score.
# coef_PCA = pca_full.components_
data_PCA_tran = pd.DataFrame(np.round(data_PCA_full_tranf, 6), 
                             columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'])
data_corr_pca = pd.concat([data_PCA_clear, data_PCA_tran], axis = 1)
corr_pca = data_corr_pca.corr()
corr_pca_col = corr_pca.drop(['Flag leaf length', 'Flag leaf width', 'Plant height', 'Panicle length', 
            'Seed length', 'Seed width', 'Seed volume', 'Seed surface area'], axis=1)
corr_pca_clear = corr_pca_col.drop(['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'])
plt.figure(figsize=(15,15))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
corr_pca_hm = sns.heatmap(round(corr_pca_clear, 6), annot=True, fmt='.6f', 
                          cmap=cmap, annot_kws={"size":10}, linewidths=.05)
plt.title('Correlation between PC and variables', fontsize=20)

ind = corr_pca_clear.index
fig, ax = plt.subplots(figsize=(10,10))
x_cir = np.linspace(-1,1,10000)
y_cir = np.sqrt(1-(x_cir**2))
ax.plot(x_cir, y_cir, sns.color_palette().as_hex()[0])
ax.plot(x_cir, -y_cir, sns.color_palette().as_hex()[0])
ax.plot([0,0], [-1,1], "k--", linewidth=3, color='r')
ax.plot([-1,1], [0,0], "k--", linewidth=3, color='r')
ax.set_xlabel('Dimension 1 ({}%)'.format(round(pca_full.explained_variance_ratio_[0]*100, 2)), fontsize=15)
ax.set_ylabel('Dimension 2 ({}%)'.format(round(pca_full.explained_variance_ratio_[1]*100, 2)), fontsize=15)
ax.set_title('Circle of correlation', fontsize=20)
for idx in ind:
    x = corr_pca_clear['PC1'][idx]
    y = corr_pca_clear['PC2'][idx]
    ax.arrow(0,0,x,y,head_width=0.05, head_length=0.01, color='b')
    ax.text(x,y,idx, size=15, ha='left', va='center')


y_target = data_PCA['Sub-population']
data_bitplot = pd.concat([data_PCA_tran, y_target], axis=1)
g = sns.pairplot(data_bitplot, hue="Sub-population", palette="husl")
