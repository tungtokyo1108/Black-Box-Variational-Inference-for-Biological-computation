# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 23:06:30 2019

Reference: Hands on Machine Learning with Scikit Learn and TensorFlow

@author: Tung1108
"""

from __future__ import division, print_function, unicode_literals

import numpy as np
import os 

np.random.seed(42)

# To make the pretty figures

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd 

mpl.rc('axes', labelsize = 14)
mpl.rc('xtick', labelsize = 12)
mpl.rc('ytick', labelsize = 12)

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)

def save_fig(fig_id, tight_layout = True, fig_extension = "png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
housing = load_housing_data()

housing.head()
housing.info()
housing.describe()
housing.hist(bins = 50, figsize=(20,15))
plt.show()

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
train_set, test_set = split_train_test(housing,0.2)
print(len(train_set), "train +", len(test_set), "test")

from zlib import crc32
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32
def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]
import hashlib
def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return hash(np.int64(identifier)).digest()[-1] < 256*test_ratio
housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
test_set.head()


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
test_set.head()

housing["median_income"].hist()
housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
housing["income_cat"].value_counts()
housing["income_cat"].hist() 

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
strat_train_set["income_cat"].value_counts() / len(strat_test_set)

def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
compare_props = pd.DataFrame({
        "Overall": income_cat_proportions(housing),
        "Stratified": income_cat_proportions(strat_test_set),
        "Random": income_cat_proportions(test_set),}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"]/compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"]/compare_props["Overall"] - 100
compare_props

###############################################################################
############### Discover and visualize the data to gain insights ##############
###############################################################################

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import plotly.figure_factory as ff

housing_train_set = strat_train_set.copy()
f, ax = plt.subplots(figsize=(10,6))
corr = housing.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm", fmt='.2f',
                 linewidth = .05)
f.subplots_adjust(top=0.93)
t = f.suptitle('Housing Correlation HeatMap', fontsize=14)

cols = ['total_rooms','total_bedrooms', 'population', 'households']
pp = sns.pairplot(housing_train_set[cols], height=1.8, aspect=1.8, 
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))
fig = pp.fig
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Housing Attributes Pairwise Plots', fontsize=14)

plt.scatter(housing['total_rooms'], housing['total_bedrooms'],
            alpha=0.4, edgecolors='w')
plt.xlabel('total_rooms')
plt.ylabel('total_bedrooms')
plt.title('Room-bedroom', y=1.05)
jp = sns.jointplot(x='total_rooms', y='total_bedrooms', data = housing_train_set, 
                   kind='reg', space=0, height=5, ratio=4)
jp1 = (sns.jointplot(x='total_rooms', y='total_bedrooms', 
                     data = housing_train_set, color='g')
                .plot_joint(sns.kdeplot,zoder=0, n_levels=6))


###############################################################################
######################## Prepare the data for ML algorithm ####################
###############################################################################

housing_train_set = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
sample_incomplete_rows = housing_train_set[housing_train_set.isnull().any(axis=1)].head()
sample_incomplete_rows

############################ Impute the missing value #########################

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

housing_num = housing_train_set.drop('ocean_proximity', axis=1) 
imputer.fit(housing_num)
imputer.statistics_
housing_num.median().values

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
housing_tr.head()
housing_ptr_1 = pd.DataFrame(X, columns=housing_num.columns,
                             index=list(housing_train_set.index.values))
housing_ptr_1.loc[sample_incomplete_rows.index.values]


########################### Categorical input feature #########################

from sklearn.preprocessing import OrdinalEncoder

housing_cat = housing_train_set[['ocean_proximity']]
housing_cat.head(10)
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]


################ Custom transformer to add extra attribute ####################

from sklearn.preprocessing import FunctionTransformer

rooms_ix, bedrooms_ix, population_ix, household_ix = [
        list(housing_train_set.columns).index(col)
        for col in ("total_rooms", "total_bedrooms", "population", "households")]
def add_extra_features(X, add_bedrooms_per_room=True):
    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
    population_per_household = X[:, population_ix] / X[:, household_ix]
    if add_bedrooms_per_room:
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household, population_per_household]
attr_adder = FunctionTransformer(add_extra_features, validate=False, 
                                 kw_args={"add_bedrooms_per_room": False})
housing_extra_attribs = attr_adder.fit_transform(housing_train_set.values)
housing_extra_attribs = pd.DataFrame(
        housing_extra_attribs, 
        columns=list(housing_train_set.columns) + ["rooms_per_household", "population_per_household"])
housing_extra_attribs.head()

##################### Build pipeline for preprocessing ########################

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
imputer = SimpleImputer("median")

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
        ('std_scaler', StandardScaler()),
        ])
housing_num_tr = num_pipeline.fit_transform(housing_num)
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
        ])
housing_prepared = full_pipeline.fit_transform(housing_train_set)


###############################################################################
########################### Select and train a model ##########################
###############################################################################

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing_train_set.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.fit_transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_mae = mean_absolute_error(housing_labels, housing_predictions)
lin_mae

from sklearn.tree import DecisionTreeRegressor 
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


###############################################################################
############################## Fine-tune the model ############################
###############################################################################

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, 
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
display_scores(tree_rmse_scores)    

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse
from sklearn.model_selection import cross_val_score
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
pd.Series(np.sqrt(-forest_scores)).describe()


























