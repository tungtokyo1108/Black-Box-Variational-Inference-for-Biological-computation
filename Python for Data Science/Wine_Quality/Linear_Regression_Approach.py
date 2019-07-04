#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:35:08 2019

Reference: 
    - https://harvard-iacs.github.io/2018-CS109A/sections/section-3/solutions/
    - https://github.com/WillKoehrsen/Data-Analysis/blob/master/bayesian_lr/Bayesian%20Linear%20Regression%20Project.ipynb

@author: tungutokyo
"""

import numpy as np
import pandas as pd
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
matplotlib.rcParams['figure.figsize'] = (13.0, 6.0)
import itertools 
import warnings
warnings.filterwarnings("ignore")

from IPython.display import display
pd.set_option('display.max_rows', 999)
pd.set_option("display.width", 500)
pd.set_option("display.notebook_repr_html", True)

######################################################################################
############################ Load and merge datasets #################################
######################################################################################

white_wine = pd.read_csv("winequality-white.csv", sep=";")
red_wine = pd.read_csv("winequality-red.csv", sep=";")
red_wine["wine_type"] = "red"
white_wine["wine_type"] = "white"

red_wine["quality_label"] = red_wine["quality"].apply(lambda value: "low" 
                                                            if value <= 5 else "medium"
                                                            if value <= 7 else "high")
red_wine["quality_label"] = pd.Categorical(red_wine["quality_label"], 
                                            categories=["low", "medium", "high"])
white_wine["quality_label"] = white_wine["quality"].apply(lambda value: "low"
                                                            if value <= 5 else "medium"
                                                            if value <= 7 else "high")
white_wine["quality_label"] = pd.Categorical(white_wine["quality_label"],
                                              categories=["low", "medium", "high"])
wines = pd.concat([red_wine, white_wine])
wines = wines.sample(frac=1, random_state=42).reset_index(drop=True)
wines = wines.dropna()
wines.info()
wines.to_csv('wines.csv', index=False)

######################################################################################
########################### Exploring target variable ################################
######################################################################################

wines["alcohol"].describe()
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16,8))
sns.distplot(wines["alcohol"], ax=ax[0,0])
sns.violinplot(x="alcohol", data=wines, ax=ax[0,1])
sns.boxplot(x="alcohol", data=wines, ax=ax[1,0])
sns.barplot(x="alcohol", data=wines, ax=ax[1,1])

######################################################################################
######################## Exploring predictor variables ###############################
######################################################################################

plt.figure(figsize=(10,10))
sns.distplot(wines["fixed acidity"], rug=True, rug_kws={"alpha": .1, "color": "k"})
plt.xlabel("Fixed Acidity", fontsize=15)
plt.title("Histogram of fixed acidity", fontsize=20)

sns.lmplot(x="fixed acidity", y="alcohol", hue="wine_type", hue_order=["white", "red"], data=wines, 
           markers=["o", "x"], size=8)
plt.xlabel("Fixed Acidity", fontsize=15)
plt.ylabel("Alcohol", fontsize=15)
plt.title("Liner relationship Fixed Acidity-Alcohol", fontsize=20)

# How about the wine_type and quality_label
wines["wine_type"].value_counts()
wines["quality_label"].value_counts()

plt.figure(figsize=(10,10))
sns.barplot(x="quality_label", hue="wine_type", y="alcohol", data=wines)
plt.title("The difference of alcohol between quality and type of wines", fontsize=20)
plt.xlabel("Quality Label", fontsize=15)
plt.ylabel("Alcohol", fontsize=15)

qualities = "low medium high".split()
types = "white red".split()
colors = sns.color_palette("Set1", n_colors=len(types), desat=.5)
# sns.palplot(colors)
positions_array = np.arange(len(qualities))
fake_handles = []
plt.figure(figsize=(10,10))
for i, typ in enumerate(types):
    offset = .15 * (-1 if i == 0 else 1)
    violin = plt.violinplot([
            wines["alcohol"][(wines["wine_type"] == typ) & (wines["quality_label"] == quality)].values
            for quality in qualities
            ], positions=positions_array + offset, widths=.25, showmedians=True, showextrema=True)
    
    color = colors[i]
    for part_name, part in violin.items():
        if part_name == "bodies":
            for body in violin["bodies"]:
                body.set_color(color)
        else:
            part.set_color(color)
    fake_handles.append(mpatches.Patch(color=color))

plt.legend(fake_handles, types)
plt.xticks(positions_array, qualities)
plt.xlabel("Quality_labels", fontsize=15)
plt.ylabel("Alcohol", fontsize=15)

######################################################################################
############################### Linear Regression ####################################
######################################################################################

model_1 = sm.OLS(wines["alcohol"], sm.add_constant(wines["fixed acidity"])).fit()
model_1.summary()

# Dummy variables 
wines_cate = wines.copy()
numeric_subset = wines.select_dtypes('number')
categorial_subset = wines[['wine_type', "quality_label"]]
categorial_subset = pd.get_dummies(categorial_subset)
wines_cate = pd.concat([numeric_subset, categorial_subset], axis=1)
wines_cate.shape

model_2 = sm.OLS(wines_cate["alcohol"], sm.add_constant(wines_cate[["fixed acidity", 
                 "wine_type_red", "quality_label_low", 
                 "quality_label_medium"]])).fit()
model_2.summary()

# Interations
wines_cate["fixed_acid_red"] = wines_cate["fixed acidity"] * wines_cate["wine_type_red"]
model_3 = sm.OLS(wines_cate["alcohol"], sm.add_constant(wines_cate[["fixed acidity", 
                "wine_type_red", "quality_label_low", "quality_label_medium", 
                "fixed_acid_red"]])).fit()
model_3.summary()

wines_cate["fixed_acid_low"] = wines_cate["fixed acidity"] * wines_cate["quality_label_low"]
wines_cate["fixed_acid_medium"] = wines_cate["fixed acidity"] * wines_cate["quality_label_medium"]
model_4 = sm.OLS(wines_cate["alcohol"], sm.add_constant(wines_cate[["fixed acidity", 
                "wine_type_red", "quality_label_low", "quality_label_medium", 
                "fixed_acid_red", "fixed_acid_low", "fixed_acid_medium"]])).fit()
model_4.summary()

models = [model_1, model_2, model_3, model_4]
plt.figure(figsize=(15,10))
plt.plot([model.df_model for model in models], [model.rsquared for model in models], "x-")
plt.xlabel("The number of variables in model", fontsize=15)
plt.ylabel("$R^2$", fontsize=15)

######################################################################################
################## Selecting mininal subset of predictor #############################
######################################################################################

predictors = wines_cate.drop(['alcohol'], axis=1)
target = wines_cate["alcohol"]

def find_best_subset_of_size(x, y, num_predictors):
    predictors = x.columns
    best_r_squared = -np.inf
    best_model_data = None
    
    subsets_of_size_k = itertools.combinations(predictors, num_predictors)
    for subset in subsets_of_size_k:
        features = list(subset)
        x_subset = sm.add_constant(x[features])
        model = sm.OLS(y, x_subset).fit()
        r_squared = model.rsquared
        
        if r_squared > best_r_squared:
            best_model_data = {
                    "r_squared": r_squared,
                    "subset": features,
                    "model": model
            }
    return best_model_data

def exhaustive_search_selection(x, y):
    predictors = x.columns
    stats = []
    models = dict()
    for k in range(1, len(predictors)):
        best_size_k_model = find_best_subset_of_size(x, y, num_predictors=k)
        best_subset = best_size_k_model["subset"]
        best_model = best_size_k_model["model"]
        stats.append({
                "k": k,
                "formula": "y ~ {}".format("+".join(best_subset)),
                "BIC": best_model.bic,
                "r_squared": best_model.rsquared})
        models[k] = best_model
    return pd.DataFrame(stats), models

find_best_subset_of_size(predictors, target, 5)
stats, models = exhaustive_search_selection(predictors, target)
stats
stats.plot(x="k", y="BIC", marker="*")
best_stat = stats.iloc[stats["BIC"].idxmin()]
best_k = best_stat["k"]
best_bic = best_stat["BIC"]
best_formula = best_stat["formula"]
best_r2 = best_stat["r_squared"]

print("The best overall model is '{formula}' with bic={bic:.2f} and R^2={r_squared:.3f}".format(
        formula=best_formula, bic=best_bic, r_squared=best_r2))
models[best_k].summary()

######################################################################################
######################## Linear Regression with Sklearn ##############################
######################################################################################

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

predictors = wines_cate.drop(columns='alcohol')
target = wines_cate["alcohol"]
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.2, random_state=42)

regression = LinearRegression()
regression.fit(X_train, y_train)

predicted_y = regression.predict(X_test)
r2 = regression.score(X_test, y_test)
print(r2)

print(regression.score(X_train, y_train))
print(mean_squared_error(predicted_y, y_test))
print(mean_squared_error(y_train, regression.predict(X_train)))
print("Coefficients: \n", regression.coef_, regression.intercept_)

fig, ax = plt.subplots(1,1, figsize=(15,10))
ax.plot(y_test, predicted_y, 'o')
grid = np.linspace(np.min(wines_cate["alcohol"]), np.max(wines_cate["alcohol"]), 100)
ax.plot(grid, grid, color="black")
ax.set_xlabel("actual_y")
ax.set_ylabel("predicted y")

######################################################################################
######################## Linear Regression with PySpark ##############################
######################################################################################

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
sc = SparkContext()
sqlContext = SQLContext(sc)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

wines = sqlContext.read.format('com.databricks.spark.csv').options(header='true', 
                              inferschema='true').load('wines.csv')
wines.cache()
wines.printSchema()

vectorAssembler = VectorAssembler(inputCols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                                               'chlorides'], outputCol = 'features')
vwines_df = vectorAssembler.transform(wines)
vwines_df = vwines_df.select(['features', 'alcohol'])
vwines_df.show(3)

splits = vwines_df.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]

lr = LinearRegression(featuresCol = "features", labelCol = "alcohol", maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))
