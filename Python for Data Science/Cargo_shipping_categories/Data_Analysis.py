#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:26:07 2019

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

def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()
    
    # Percentage of missing value 
    mis_val_percent = 100 * df.isnull().sum()/len(df)
    
    # Make a table with results 
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    
    # Rename the columns 
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : "Missing Values",
                                                                1 : "% Total Values"})
    
    mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
            "% Total Values", ascending=False).round(1)
    
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) + 
          " columns that have missing value")
    
    return mis_val_table_ren_columns

missing_values_table = missing_values_table(data_train)
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

data_train["Size"] = data_train["size"].apply(lambda x: "Small"
                                                  if x <= 4 else "Middle"
                                                  if x <= 12 else "Large")
data_train["Size"] = pd.Categorical(data_train["Size"], 
                          categories=["Small", "Middle", "Large"])

data_train_final = data_train.drop(columns = ["category_class", "sold_price", "size", "listing_at",
                                              "time", "month", "weekofyear", "dayofweek", "DayOfWeek"])

###############################################################################
######################### Target variable Analysis ############################
###############################################################################

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))    
ax[0] = data_train["category_label"].value_counts().plot(kind="barh", ax=ax[0])
for i, v in enumerate(data_train["category_label"].value_counts()):
    ax[0].text(v, i, " "+str(v),va="center", color="black", fontweight="bold")
ax[0].set_ylabel("Category class", fontsize=15)
ax[0].set_xlabel("Number item", fontsize=15)
ax[0].set_title("The number of items in categories", fontsize=20)
ax[1] = data_train["category_label"].value_counts().plot(kind="pie", autopct = "%1.1f%%", ax=ax[1])    
ax[1].set_ylabel("")
ax[1].set_title("The percentage of items in categories", fontsize=20)    
fig.subplots_adjust(hspace=1)
plt.show()    
    
data_condition = data_train.groupby(["category_label"])["condition"].value_counts().unstack()  
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,20))
ax[0].pie(data_condition["Fair"], autopct="%1.1f%%", labels=data_condition.index)
ax[0].set_aspect("equal")
ax[0].set_title("Fair", fontsize=15)
ax[1].pie(data_condition["Good"], autopct="%1.1f%%", labels=data_condition.index)
ax[1].set_aspect("equal")
ax[1].set_title("Good", fontsize=15)
ax[2].pie(data_condition["Like New"], autopct="%1.1f%%", labels=data_condition.index)
ax[2].set_aspect("equal")
ax[2].set_title("Like New", fontsize=15)
fig.subplots_adjust(hspace=1)
plt.show() 


plt.figure(figsize=(12,12))
sns.jointplot(x="sold_price", y="price", data=data_train, size=10)
data_special_price = data_train[data_train["price"] > 1.5*data_train["sold_price"]].drop(
        columns= {"item_id", "category_class", "item_tag_hash"})
data_special_price

data_cat_cond = data_train.groupby(["category_class"])["condition"].value_counts().unstack()

###############################################################################
################# Feature Engineering and Selection ###########################
###############################################################################

numeric_subset = data_train_final["price"]

categorial_subset = data_train_final[["category_label", "condition", "area_name", "Size",
                                      "Month", "WeekofYear", "DayofWeek"]]
categorial_subset = pd.get_dummies(categorial_subset)
features = pd.concat([categorial_subset, numeric_subset], axis=1)

plt.figure(figsize=(20,20))
numeric_corr = features.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
numeric_hm = sns.heatmap(round(numeric_corr,2), annot=True, cmap=cmap, 
                         fmt=".2f", linewidth=.05, annot_kws={"size":10})

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
    pca = decomposition.PCA(n_components=4)
    pca_reduced = pca.fit_transform(data)
    
    print("The shape of transformed data", pca_reduced.shape)
    print(pca_reduced[0:4])
    
    pca_data = np.vstack((pca_reduced.T, labels)).T
    print("The shape of data with labels", pca_data.shape)
    
    pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal_component", "2nd_principal_component",
                                                  "3rd_principal_component", "4th_principal_component", "labels"))
    print(pca_df.head())
    
    sns.FacetGrid(pca_df, hue="labels", height=8).map(sns.scatterplot, 
            "1st_principal_component", "2nd_principal_component", edgecolor="w").add_legend()
    plt.xlabel("1st_principal_component ({}%)".format(round(pca.explained_variance_ratio_[0]*100),2), fontsize=15)
    plt.ylabel("2nd_principal_component ({}%)".format(round(pca.explained_variance_ratio_[1]*100),2), fontsize=15)
    
    sns.FacetGrid(pca_df, hue="labels", height=8).map(sns.scatterplot, 
            "3rd_principal_component", "4th_principal_component", edgecolor="w").add_legend()
    plt.xlabel("3rd_principal_component ({}%)".format(round(pca.explained_variance_ratio_[2]*100),2), fontsize=15)
    plt.ylabel("4th_principal_component ({}%)".format(round(pca.explained_variance_ratio_[3]*100),2), fontsize=15)
    plt.show()
    print("Time taken to perform Principal Component Analysis: {}".format(dt.now()-st))
    
def pca_redu(data):
    pca = decomposition.PCA()
    pca_data = pca.fit_transform(data)
    plt.figure(figsize=(15,10))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.axis('tight')
    plt.grid()
    plt.axhline(0.95, c='r')
    plt.xlabel("Number of components", fontsize=15)
    plt.ylabel("Cumulative explained variance", fontsize=15)
    plt.legend()

X = features.drop(columns = ["category_label_Cat_0", "category_label_Cat_1", "category_label_Cat_2",
                             "category_label_Cat_3", "category_label_Cat_4"])
y = data_train_final["category_label"]

X = standardize(X)
pca_redu(X)
pca_vis(X,y)


###############################################################################
##################### Model training and evaluation ###########################
###############################################################################

import itertools
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV

def get_RandSearchCV_RF(X_train, y_train, X_test, y_test, scoring):
    from sklearn.model_selection import TimeSeriesSplit
    from datetime import datetime as dt 
    st_t = dt.now()
    # Numer of trees are used
    n_estimators = [5, 10, 50, 100, 150, 200, 250, 300]
    
    # Maximum depth of each tree
    max_depth = [5, 10, 25, 50, 75, 100]
    
    # Minimum number of samples per leaf 
    min_samples_leaf = [1, 2, 4, 8, 10]
    
    # Minimum number of samples to split a node
    min_samples_split = [2, 4, 6, 10]
    
    # Maximum numeber of features to consider for making splits
    max_features = ["auto", "sqrt", "log2", None]
    
    hyperparameter = {'n_estimators': n_estimators,
                      'max_depth': max_depth,
                      'min_samples_leaf': min_samples_leaf,
                      'min_samples_split': min_samples_split,
                      'max_features': max_features}
    
    cv_timeSeries = TimeSeriesSplit(n_splits=5).split(X_train)
    base_model = RandomForestClassifier(criterion="gini", random_state=42)
    
    # Run randomzed search 
    n_iter_search = 30
    rsearch_cv = RandomizedSearchCV(estimator=base_model, 
                                   random_state=42,
                                   param_distributions=hyperparameter,
                                   n_iter=n_iter_search,
                                   cv=cv_timeSeries,
                                   scoring=scoring,
                                   n_jobs=-1)
    rsearch_cv.fit(X_train, y_train)
    print("Best estimator obtained from CV data: \n", rsearch_cv.best_estimator_)
    print("Best Score: ", rsearch_cv.best_score_)
    return rsearch_cv

def performance_rand(best_clf, X_train, y_train, X_test, y_test, type_search):
    print("-"*100)
    print("~~~~~~~~~~~~~~~~~~ PERFORMANCE EVALUATION ~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Detailed report for the {} algorithm".format(type_search))
    
    y_pred = best_clf.predict(X_test)
    
    test_accuracy = accuracy_score(y_test, y_pred, normalize=True) * 100
    points = accuracy_score(y_test, y_pred, normalize=False)
    print("The number of accurate predictions out of {} data points on unseen data is {}".format(
            X_test.shape[0], points))
    print("Accuracy of the {} model on unseen data is {}".format(
            type_search, np.round(test_accuracy, 2)))
    
    print("Precision of the {} model on unseen data is {}".format(
            type_search, np.round(metrics.precision_score(y_test, y_pred, average="macro"), 4)))
    print("Recall of the {} model on unseen data is {}".format(
           type_search, np.round(metrics.recall_score(y_test, y_pred, average="macro"), 4)))
    print("F1 score of the {} model on unseen data is {}".format(
            type_search, np.round(metrics.f1_score(y_test, y_pred, average="macro"), 4)))
    
    print("\nClassification report for {} model: \n".format(type_search))
    print(metrics.classification_report(y_test, y_pred))
    
    plt.figure(figsize=(10,10))
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    """
    cmap = plt.cm.Blues
    plt.imshow(cnf_matrix, interpolation="nearest", cmap=cmap)
    plt.colorbar()
    fmt = "d"
    thresh = cnf_matrix.max()/2
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j,i,format(cnf_matrix[i,j], fmt), ha="center", va="center", 
                 color="white" if cnf_matrix[i,j] > thresh else "black")
    """
    cmap = plt.cm.Blues
    sns.heatmap(cnf_matrix, annot=True, cmap=cmap, fmt="d", annot_kws={"size":12}, linewidths=.05)
    plt.title("The confusion matrix", fontsize=20)
    plt.ylabel("True label", fontsize=15)
    plt.xlabel("Predicted label", fontsize=15)
    plt.show()
    
    """
    y_pred_prob = best_clf.predict_proba(X_test)[:,1]
    n_classes = 5
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test.values[i], y_pred_prob[i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    """
    
    importances = best_clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    return {"importance": importances, 
            "index": indices}
    
    
def RF_classifier(X_train, y_train, X_test, y_test, scoring, type_search):
    print("*"*100)
    print("Starting {} steps with {} for evaluation rules...".format(type_search, scoring))
    print("*"*100)
    
    rsearch_cv = get_RandSearchCV_RF(X_train, y_train, X_test, y_test, scoring)
    
    best_estimator = rsearch_cv.best_estimator_
    max_depth = rsearch_cv.best_estimator_.max_depth
    n_estimators = rsearch_cv.best_estimator_.n_estimators
    var_imp_rf = performance_rand(best_estimator, X_train, y_train, X_test, y_test, type_search)
    
    print("~~~~~~~~~~~~~ Features ranking and ploting ~~~~~~~~~~~~~~~~~~~~~\n")
    
    importances_rf = var_imp_rf["importance"]
    indices_rf = var_imp_rf["index"]
    
    for f in range(0, indices_rf.shape[0]):
        i = f
        print("{0}. The features '{1}' contribute {2:.5f} to decreasing the weighted impurity".format(
                f+1, X_train.columns[indices_rf[i]], importances_rf[indices_rf[f]]))
    
    index = np.arange(len(X_train.columns))
    importance_desc = sorted(importances_rf)
    feature_space = []
    for i in range(indices_rf.shape[0]-1, -1, -1):
        feature_space.append(X_train.columns[indices_rf[i]])
    
    fig, ax = plt.subplots(figsize=(15,15))
    ax = plt.gca()
    plt.title("Feature importances for Random Forest Model", fontsize=20)
    plt.barh(index, importance_desc, align="center", color="blue", alpha=0.6)
    plt.grid(axis="x", color="white", linestyle="-")
    plt.yticks(index, feature_space)
    plt.xlabel("The Average of Decrease in Impurity", fontsize=15)
    plt.ylabel("Features", fontsize=15)
    ax.tick_params(axis="both", which="both", length=0)
    plt.show()

"""
Random Forest for all data_train_final
"""
X = features.drop(columns = ["category_label_Cat_0", "category_label_Cat_1", "category_label_Cat_2",
                             "category_label_Cat_3", "category_label_Cat_4"])
X["Log_price"] = np.log(features["price"])
y = data_train["category_class"]
y_binary = label_binarize(y, classes=[0,1,2,3,4])
n_classes = y_binary.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 

RF_classifier(X_train, y_train, X_test, y_test, "accuracy", "RandomSearchCV-RF")

"""
Random Forest for second training 
"""

X_2 = features.drop(columns = ["category_label_Cat_0", "category_label_Cat_1", "category_label_Cat_2",
                             "category_label_Cat_3", "category_label_Cat_4", "WeekofYear_4th_week",
                             "WeekofYear_5th_week", "DayofWeek_Fri", "DayofWeek_Mon", 
                             "DayofWeek_Sat", "DayofWeek_Sun", "DayofWeek_Thu", "DayofWeek_Tue",
                             "DayofWeek_Wed"])
y = data_train["category_class"]
X_train_2, X_test_2, y_train, y_test = train_test_split(X_2, y, test_size=0.3, random_state=42)
RF_classifier(X_train_2, y_train, X_test_2, y_test, "accuracy", "RandomSearchCV-RF")














































