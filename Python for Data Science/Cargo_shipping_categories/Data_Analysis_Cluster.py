#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 10:55:35 2019

@author: tungutokyo
"""

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", 60)

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.pylabtools import figsize
import plotly
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.tools as tls
import cufflinks as cf

###############################################################################
######################### Data Cleaning and Formatting ########################
###############################################################################

data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")
data_train.info()

###############################################################################
########################## Data Preprocessing #################################
###############################################################################

data_train = data_train.drop(columns = ["item_id", "item_tag_hash"])    
    

data_train["category_label"] = data_train["category_class"].apply(lambda x: "Cat_0" 
                                                            if x == 0 else "Cat_1"
                                                            if x == 1 else "Cat_2"
                                                            if x == 2 else "Cat_3"
                                                            if x == 3 else "Cat_4")
data_train["category_label"] = pd.Categorical(data_train["category_label"], 
                                              categories=["Cat_0", "Cat_1", "Cat_2", "Cat_3", "Cat_4"])

data_train["time"] = pd.to_datetime(data_train["listing_at"], errors="coerce")
data_train["month"] = data_train["time"].dt.month
data_train["weekofyear"] = data_train["time"].dt.weekofyear
data_train["dayofweek"] = data_train["time"].dt.weekday
data_train["hourofday"] = data_train["time"].dt.hour

data_train["Month"] = data_train["month"].apply(lambda x: "Jan" 
                                                  if x == 1 else "Feb")
data_train["Month"] = pd.Categorical(data_train["Month"],
                                      categories=["Jan", "Feb"])

data_train["WeekofYear"] = data_train["weekofyear"].apply(lambda x: "4th_week" 
                                                      if x == 4 else "5th_week")
data_train["WeekofYear"] = pd.Categorical(data_train["WeekofYear"],
                                              categories=["4th_week", "5th_week"])

data_train["DayofWeek"] = data_train["dayofweek"].apply(lambda x: "Mon"
                                                          if x == 0 else "Tue"
                                                          if x == 1 else "Wed"
                                                          if x == 2 else "Thu"
                                                          if x == 3 else "Fri"
                                                          if x == 4 else "Sat"
                                                          if x == 5 else "Sun")
data_train["DayofWeek"] = pd.Categorical(data_train["DayofWeek"],
                              categories=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

data_train["HourofDay"] = data_train["hourofday"].apply(lambda x: "0-5AM"
                                                          if x <= 5 else "6-12AM"
                                                          if x <= 12 else "1-5PM"
                                                          if x <= 17 else "6-12PM")
data_train["HourofDay"] = pd.Categorical(data_train["HourofDay"],
                              categories=["0-5AM", "6-12AM", "1-5PM", "6-12PM"])

data_train["TypeofPrice"] = data_train["price"].apply(lambda x: "Cheap"
                                                          if x <= 948 else "Middle"
                                                          if x <= 1193.75 else "Expensive")
data_train["TypeofPrice"] = pd.Categorical(data_train["TypeofPrice"], 
                              categories=["Cheap", "Middle", "Expensive"])


data_train["TypeofSize"] = data_train["size"].apply(lambda x: "Small"
                                                  if x <= 4 else "Middle"
                                                  if x <= 12 else "Large")
data_train["TypeofSize"] = pd.Categorical(data_train["TypeofSize"], 
                          categories=["Small", "Middle", "Large"])

data_train_final = data_train.drop(columns = ["category_class", "sold_price", "listing_at",
                                              "time", "month", "weekofyear", "dayofweek", 
                                              "hourofday"])
    
###############################################################################
############################ Cluster Analysis #################################
###############################################################################

"""
Subset data without time varibles
"""
numeric_pre_subset = data_train[["price", "size"]]
categorial_pre_subset = data_train_final[["category_label", "condition", "area_name"]]
categorial_pre_subset = pd.get_dummies(categorial_pre_subset)
features_pre = pd.concat([categorial_pre_subset, numeric_pre_subset], axis=1)

"""
Subset data with time variables
"""
numeric_subset = data_train_final[["price", "size"]]
categorial_subset = data_train_final[["category_label", "condition", "area_name", 
                                      "Month", "WeekofYear", "DayofWeek", "HourofDay"]]
categorial_subset = pd.get_dummies(categorial_subset)
features = pd.concat([categorial_subset, numeric_subset], axis=1)

X = features.drop(columns = ["category_label_Cat_0", "category_label_Cat_1", "category_label_Cat_2",
                             "category_label_Cat_3", "category_label_Cat_4"])

from sklearn import cluster
cluster_pred = cluster.KMeans(n_clusters = 3, max_iter = 1000, random_state=42, n_jobs=-1).fit_predict(X)

"""
Hierarchical clustering - Agglomerative Clustering
"""
cluster_pred = cluster.AgglomerativeClustering(n_clusters = 3, linkage="ward").fit_predict(X)

"""
Gaussian Mixture Model 
"""
from sklearn import mixture
gmm = mixture.GaussianMixture(n_components = 3, covariance_type = "full", 
                                init_params = "kmeans", max_iter = 10000, random_state=42).fit(X)
cluster_pred = gmm.predict(X)



data_train = data_train.assign(Price_Size = cluster_pred)
data_train["Group"] = data_train["Price_Size"].apply(lambda x: "A" 
                                                      if x == 0 else "B"
                                                      if x == 1 else "C"
                                                      )
data_train["Group"] = pd.Categorical(data_train["Group"], 
                      categories=["A", "B", "C"])

data_train_final = data_train.drop(columns = ["category_class", "sold_price", "listing_at",
                                              "time", "month", "weekofyear", "dayofweek", "Price_Size",
                                              "hourofday"])


###############################################################################
################### Lower dimensional visualization ###########################
###############################################################################

from sklearn import decomposition
from datetime import datetime as dt

def standardize(data):
    from sklearn.preprocessing import StandardScaler
    data_std = StandardScaler().fit_transform(data)
    return data_std

def pca_vis(data, labels):
    st = dt.now()
    pca = decomposition.PCA(n_components=6)
    pca_reduced = pca.fit_transform(data)
    
    print("The shape of transformed data", pca_reduced.shape)
    print(pca_reduced[0:6])
    
    pca_data = np.vstack((pca_reduced.T, labels)).T
    print("The shape of data with labels", pca_data.shape)
    
    pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal_component", "2nd_principal_component",
                                                  "3rd_principal_component", "4th_principal_component", 
                                                  "5th_principal_component", "6th_principal_component", "labels"))
    print(pca_df.head())
    
    sns.FacetGrid(pca_df, hue="labels", height=8).map(sns.scatterplot, 
            "1st_principal_component", "2nd_principal_component", edgecolor="w").add_legend()
    plt.xlabel("1st_principal_component ({}%)".format(round(pca.explained_variance_ratio_[0]*100),2), fontsize=15)
    plt.ylabel("2nd_principal_component ({}%)".format(round(pca.explained_variance_ratio_[1]*100),2), fontsize=15)
    
    sns.FacetGrid(pca_df, hue="labels", height=8).map(sns.scatterplot, 
            "3rd_principal_component", "4th_principal_component", edgecolor="w").add_legend()
    plt.xlabel("3rd_principal_component ({}%)".format(round(pca.explained_variance_ratio_[2]*100),2), fontsize=15)
    plt.ylabel("4th_principal_component ({}%)".format(round(pca.explained_variance_ratio_[3]*100),2), fontsize=15)
    
    sns.FacetGrid(pca_df, hue="labels", height=8).map(sns.scatterplot, 
            "5th_principal_component", "6th_principal_component", edgecolor="w").add_legend()
    plt.xlabel("5th_principal_component ({}%)".format(round(pca.explained_variance_ratio_[4]*100),2), fontsize=15)
    plt.ylabel("6th_principal_component ({}%)".format(round(pca.explained_variance_ratio_[5]*100),2), fontsize=15)
    plt.show()
    print("Time taken to perform Principal Component Analysis: {}".format(dt.now()-st))
    return pca_df
    
def pca_redu(data, num_components):
    pca = decomposition.PCA(n_components=num_components)
    pca_data = pca.fit_transform(data)
    plt.figure(figsize=(15,10))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.axis('tight')
    plt.grid()
    plt.axhline(0.95, c='r')
    plt.xlabel("Number of components", fontsize=15)
    plt.ylabel("Cumulative explained variance", fontsize=15)
    plt.legend()
    return pca_data

numeric_subset = data_train_final[["price", "size"]]
categorial_subset = data_train_final[["category_label", "condition", "area_name",
                                      "Month", "WeekofYear", "DayofWeek", "HourofDay"]]
categorial_subset = pd.get_dummies(categorial_subset)
features = pd.concat([categorial_subset, numeric_subset], axis=1)

"""
With time variable 
"""
X = features.drop(columns = ["category_label_Cat_0", "category_label_Cat_1", "category_label_Cat_2",
                             "category_label_Cat_3", "category_label_Cat_4"])
#y = data_train_final["category_label"]
y = data_train_final["Group"]

X = standardize(X)
pca_full_result = pd.DataFrame(pca_redu(X,4))
pca_result = pca_vis(X,y)






































