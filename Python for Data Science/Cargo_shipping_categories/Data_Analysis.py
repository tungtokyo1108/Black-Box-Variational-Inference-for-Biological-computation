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
numeric_subset = data_train_final["price"]
categorial_subset = data_train_final[["category_label", "condition", "area_name", "TypeofSize",
                                      "Month", "WeekofYear", "DayofWeek", "HourofDay"]]
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

"""
With time variable 
"""
X = features.drop(columns = ["category_label_Cat_0", "category_label_Cat_1", "category_label_Cat_2",
                             "category_label_Cat_3", "category_label_Cat_4"])
y = data_train_final["category_label"]

X = standardize(X)
pca_full_result = pd.DataFrame(pca_redu(X,32))
pca_result = pca_vis(X,y)

"""
Without time variables 
"""
X_pre = features_pre.drop(columns = ["category_label_Cat_0", "category_label_Cat_1", "category_label_Cat_2",
                             "category_label_Cat_3", "category_label_Cat_4"])
y = data_train_final["category_label"]
X_pre = standardize(X_pre)
pca_pre_full = pd.DataFrame(pca_redu(X_pre,6))
pca_pre_result = pca_vis(X_pre,y)


from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.preprocessing import StandardScaler

def tsne(dataset, labels, perplexity):
    model = TSNE(n_components=2, random_state=0, n_jobs=8, perplexity=perplexity, n_iter=5000)
    tsne_data = model.fit_transform(dataset)
    tsne_data = np.vstack((tsne_data.T, labels)).T
    tsne_df = pd.DataFrame(data=tsne_data, columns=("Dimension 1", "Dimension 2", "labels"))
    print("T-SNE plot for perplexity = {}".format(perplexity))
    return tsne_df

def tsne_plot(dataset, labels, perplexity):
    sns.FacetGrid(dataset, hue="labels", size=8).map(
            sns.scatterplot, "Dimension 1", "Dimension 2", edgecolor="w").add_legend()
    plt.title("T-SNE with perplexity = {} and n_iter = 5000".format(perplexity), fontsize=15)
    plt.show()
    
tsne_df = tsne(X, y, 100)
tsne_plot(tsne_df, y, 100)

###############################################################################
##################### Model training and evaluation ###########################
###############################################################################

import itertools
from scipy import interp
from itertools import cycle
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

def get_RandSearchCV(X_train, y_train, X_test, y_test, scoring, type_search, output_file):
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
    base_model_rf = RandomForestClassifier(criterion="gini", random_state=42)
    base_model_gb = GradientBoostingClassifier(criterion="friedman_mse", random_state=42)
    
    # Run randomzed search 
    n_iter_search = 30
    if type_search == "RandomSearchCV-RandomForest":
        rsearch_cv = RandomizedSearchCV(estimator=base_model_rf, 
                                   random_state=42,
                                   param_distributions=hyperparameter,
                                   n_iter=n_iter_search,
                                   cv=cv_timeSeries,
                                   scoring=scoring,
                                   n_jobs=-1)
    else:
        rsearch_cv = RandomizedSearchCV(estimator=base_model_gb, 
                                   random_state=42,
                                   param_distributions=hyperparameter,
                                   n_iter=n_iter_search,
                                   cv=cv_timeSeries,
                                   scoring=scoring,
                                   n_jobs=-1)
    
    rsearch_cv.fit(X_train, y_train)
    #f = open("output.txt", "a")
    print("Best estimator obtained from CV data: \n", rsearch_cv.best_estimator_, file=output_file)
    print("Best Score: ", rsearch_cv.best_score_, file=output_file)
    return rsearch_cv

def performance_rand(best_clf, X_train, y_train, X_test, y_test, type_search, output_file):
    #f = open("output.txt", "a")
    print("-"*100)
    print("~~~~~~~~~~~~~~~~~~ PERFORMANCE EVALUATION ~~~~~~~~~~~~~~~~~~~~~~~~", file=output_file)
    print("Detailed report for the {} algorithm".format(type_search), file=output_file)
    
    y_pred = best_clf.predict(X_test)
    y_pred_prob = best_clf.predict_proba(X_test)
    
    test_accuracy = accuracy_score(y_test, y_pred, normalize=True) * 100
    points = accuracy_score(y_test, y_pred, normalize=False)
    print("The number of accurate predictions out of {} data points on unseen data is {}".format(
            X_test.shape[0], points), file=output_file)
    print("Accuracy of the {} model on unseen data is {}".format(
            type_search, np.round(test_accuracy, 2)), file=output_file)
    
    print("Precision of the {} model on unseen data is {}".format(
            type_search, np.round(metrics.precision_score(y_test, y_pred, average="macro"), 4)), file=output_file)
    print("Recall of the {} model on unseen data is {}".format(
           type_search, np.round(metrics.recall_score(y_test, y_pred, average="macro"), 4)), file=output_file)
    print("F1 score of the {} model on unseen data is {}".format(
            type_search, np.round(metrics.f1_score(y_test, y_pred, average="macro"), 4)), file=output_file)
    
    print("\nClassification report for {} model: \n".format(type_search), file=output_file)
    print(metrics.classification_report(y_test, y_pred), file=output_file)
    
    plt.figure(figsize=(12,12))
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    print("\nThe Confusion Matrix: \n", file=output_file)
    print(cnf_matrix, file=output_file)
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
    sns.heatmap(cnf_matrix_norm, annot=True, cmap=cmap, fmt=".2f", annot_kws={"size":15}, linewidths=.05)
    if type_search == "RandomSearchCV-RandomForest":
        plt.title("The Normalized Confusion Matrix - {}".format("RandomForest"), fontsize=20)
    else:
        plt.title("The Normalized Confusion Matrix - {}".format("GradientBoosting"), fontsize=20)
    
    plt.ylabel("True label", fontsize=15)
    plt.xlabel("Predicted label", fontsize=15)
    plt.show()
    
    print("\nROC curve and AUC")
    y_pred = best_clf.predict(X_test)
    y_pred_prob = best_clf.predict_proba(X_test)
    y_test_cat = np.array(pd.get_dummies(y_test))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(5):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test_cat[:,i], y_pred_prob[:,i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(5)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(5):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        
    mean_tpr /= 5
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(12,12))
    plt.plot(fpr["macro"], tpr["macro"], 
         label = "macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
         color = "navy", linestyle=":", linewidth=4)
    colors = cycle(["red", "orange", "blue", "pink", "green"])
    for i, color in zip(range(5), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label = "ROC curve of class {0} (AUC = {1:0.2f})".format(i, roc_auc[i]))   
    plt.plot([0,1], [0,1], "k--", lw=2)
    plt.title("ROC-AUC for {}".format(type_search), fontsize=20)
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.legend(loc="lower right")
    plt.show()
    
    importances = best_clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    return {"importance": importances, 
            "index": indices,
            "y_pred": y_pred,
            "y_pred_prob": y_pred_prob}

def RF_classifier(X_train, y_train, X_test, y_test, scoring, type_search, output_file):
    #f = open("output.txt", "a")
    print("*"*100)
    print("Starting {} steps with {} for evaluation rules...".format(type_search, scoring))
    print("*"*100)
    
    rsearch_cv = get_RandSearchCV(X_train, y_train, X_test, y_test, scoring, type_search, output_file)
    
    best_estimator = rsearch_cv.best_estimator_
    max_depth = rsearch_cv.best_estimator_.max_depth
    n_estimators = rsearch_cv.best_estimator_.n_estimators
    var_imp_rf = performance_rand(best_estimator, X_train, y_train, X_test, y_test, type_search, output_file)
    
    print("\n~~~~~~~~~~~~~ Features ranking and ploting ~~~~~~~~~~~~~~~~~~~~~\n", file=output_file)
    
    importances_rf = var_imp_rf["importance"]
    indices_rf = var_imp_rf["index"]
    y_pred = var_imp_rf["y_pred"]
        
    feature_tab = pd.DataFrame({"Features" : list(X_train.columns),
                                "Importance": importances_rf})
    feature_tab = feature_tab.sort_values("Importance", ascending = False).reset_index(drop=True)
    print(feature_tab, file=output_file)
    
    index = np.arange(len(X_train.columns))
    importance_desc = sorted(importances_rf)
    feature_space = []
    for i in range(indices_rf.shape[0]-1, -1, -1):
        feature_space.append(X_train.columns[indices_rf[i]])
    
    fig, ax = plt.subplots(figsize=(15,15))
    ax = plt.gca()
    plt.title("Feature importances for {} Model".format(type_search), fontsize=20)
    plt.barh(index, importance_desc, align="center", color="blue", alpha=0.6)
    plt.grid(axis="x", color="white", linestyle="-")
    plt.yticks(index, feature_space)
    plt.xlabel("The Average of Decrease in Impurity", fontsize=15)
    plt.ylabel("Features", fontsize=15)
    ax.tick_params(axis="both", which="both", length=0)
    plt.show()
    
    return var_imp_rf

"""
Random Forest for all data_train_final
"""
X = features.drop(columns = ["category_label_Cat_0", "category_label_Cat_1", "category_label_Cat_2",
                             "category_label_Cat_3", "category_label_Cat_4"])
# X["pca_1"] = pca_result["1st_principal_component"]
# X["pca_2"] = pca_result["2nd_principal_component"]
# X["pca_3"] = pca_result["3rd_principal_component"]
# X["pca_4"] = pca_result["4th_principal_component"]
# X = pd.concat([X, pca_full_result], axis=1)
X["pca_1"] = pca_pre_result["1st_principal_component"]
X["pca_2"] = pca_pre_result["2nd_principal_component"]
X["pca_3"] = pca_pre_result["3rd_principal_component"]
X["pca_4"] = pca_pre_result["4th_principal_component"]
X["pca_5"] = pca_pre_result["5th_principal_component"]
X["pca_6"] = pca_pre_result["6th_principal_component"]
# X = pd.concat([X, pca_pre_full], axis=1)
# X["Log_price"] = np.log(X["price"])
# X["X^2"] = X["price"]^2
# X["X^3"] = X["price"]^3
y = data_train["category_class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 

f = open("output.txt", "a")
result = RF_classifier(X_train, y_train, X_test, y_test, "f1_macro", "RandomSearchCV-RandomForest", f)
f.close()

result = RF_classifier(X_train, y_train, X_test, y_test, "f1_macro", "RandomSearchCV-GradientBoosting")

"""
Random Forest for second training 
"""

X_2 = features.drop(columns = ["category_label_Cat_0", "category_label_Cat_1", "category_label_Cat_2",
                             "category_label_Cat_3", "category_label_Cat_4", "WeekofYear_4th_week",
                             "WeekofYear_5th_week", "DayofWeek_Fri", "DayofWeek_Mon", 
                             "DayofWeek_Sat", "DayofWeek_Sun", "DayofWeek_Thu", "DayofWeek_Tue",
                             "DayofWeek_Wed"])
X_2["pca_1"] = pca_pre_result["1st_principal_component"]
X_2["pca_2"] = pca_pre_result["2nd_principal_component"]
X_2["pca_3"] = pca_pre_result["3rd_principal_component"]
X_2["pca_4"] = pca_pre_result["4th_principal_component"]
y = data_train["category_class"]
X_train_2, X_test_2, y_train, y_test = train_test_split(X_2, y, test_size=0.3, random_state=42)
result = RF_classifier(X_train_2, y_train, X_test_2, y_test, "f1_macro", "RandomSearchCV-RandomForest")
result = RF_classifier(X_train_2, y_train, X_test_2, y_test, "f1_macro", "RandomSearchCV-GradientBoosting")

"""
Random Forest for third training
"""

X_3 = features.drop(columns = ["category_label_Cat_0", "category_label_Cat_1", "category_label_Cat_2",
                             "category_label_Cat_3", "category_label_Cat_4", "area_name_ccc", 
                             "area_name_ddd", "area_name_aaa", "area_name_bbb", "area_name_hhh",
                             "area_name_kkk", "area_name_jjj", "area_name_fff", "area_name_ggg",
                             "area_name_eee"])
X_3["pca_1"] = pca_pre_result["1st_principal_component"]
X_3["pca_2"] = pca_pre_result["2nd_principal_component"]
X_3["pca_3"] = pca_pre_result["3rd_principal_component"]
X_3["pca_4"] = pca_pre_result["4th_principal_component"]
y = data_train["category_class"]
X_train_3, X_test_3, y_train, y_test = train_test_split(X_3, y, test_size=0.3, random_state=42)
result = RF_classifier(X_train_3, y_train, X_test_3, y_test, "f1_macro", "RandomSearchCV-RandomForest")
result = RF_classifier(X_train_3, y_train, X_test_3, y_test, "f1_macro", "RandomSearchCV-GradientBoosting")

###############################################################################
######################### Visualize decision tree #############################
###############################################################################

from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image

final_model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=25, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=4,
            min_weight_fraction_leaf=0.0, n_estimators=250, n_jobs=None,
            oob_score=False, random_state=42, verbose=0, warm_start=False)

final_model.fit(X_train, y_train)
estimator = final_model.estimators_[1]
target_name = pd.DataFrame(pd.get_dummies(data_train_final["category_label"]))

export_graphviz(estimator, out_file="tree.dot", feature_names = X_train.columns, 
                class_names = target_name.columns, rounded = True, proportion = False, 
                precision = 2, filled = True)
call(["dot", "-Tpng", "tree.dot", "-o", "tree.png"])

###############################################################################
######################### Explain model training ##############################
###############################################################################

import lime
import lime.lime_tabular

y_pred = result["y_pred"]
y_pred_prob = result["y_pred_prob"]

clf = []
for clas in range(5):
    clf.append(RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=25, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=4,
            min_weight_fraction_leaf=0.0, n_estimators=250, n_jobs=None,
            oob_score=False, random_state=42, verbose=0, warm_start=False).fit(X_train, y_train == clas))


explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names = X_train.columns,
                                                   random_state=42)

def explain_row (clf, row, num_reasons = 15):
    exp = [
            exp_pair[0] for exp_pair in explainer.explain_instance(
                    row, clf.predict_proba, labels = [1], num_features = num_reasons).as_list() if exp_pair[1] > 0
            ][:num_reasons]
    exp += [""] * (num_reasons - len(exp))
    return exp

def predict_explain(rf, X, score_cat, num_reasons = 15):
    pred_ex = X[[]]
    #pred_ex[score_cat] = rf.predict_proba(X)[:,1]
    cols = zip(
            *X.apply(
                    lambda x: explain_row(rf, x, num_reasons), axis = 1, raw = True))
    for n in range(num_reasons):
        pred_ex["REASON %d" %(n+1)] = next(cols)
    
    return pred_ex

explain_cat0 = predict_explain(clf[0], X_test, "SCORE_CAT_0").assign(
        SCORE_CAT_0 = y_pred_prob[:,0], PRED_CLASS = y_pred, TRUE_CLASS = y_test).sort_values(
                "SCORE_CAT_0", ascending = False)

explain_cat1 = predict_explain(clf[1], X_test, "SCORE_CAT_1").assign(
        SCORE_CAT_1 = y_pred_prob[:,1], PRED_CLASS = y_pred, TRUE_CLASS = y_test).sort_values(
                    "SCORE_CAT_1", ascending = False)

explain_cat2 = predict_explain(clf[2], X_test, "SCORE_CAT_2").assign(
        SCORE_CAT_2 = y_pred_prob[:,2], PRED_CLASS = y_pred, TRUE_CLASS = y_test).sort_values(
                    "SCORE_CAT_2", ascending = False)

explain_cat3 = predict_explain(clf[3], X_test, "SCORE_CAT_3").assign(
        SCORE_CAT_3 = y_pred_prob[:,3], PRED_CLASS = y_pred, TRUE_CLASS = y_test).sort_values(
                    "SCORE_CAT_3", ascending = False)

explain_cat4 = predict_explain(clf[4], X_test, "SCORE_CAT_4").assign(
        SCORE_CAT_4 = y_pred_prob[:,4], PRED_CLASS = y_pred, TRUE_CLASS = y_test).sort_values(
                    "SCORE_CAT_4", ascending = False)
