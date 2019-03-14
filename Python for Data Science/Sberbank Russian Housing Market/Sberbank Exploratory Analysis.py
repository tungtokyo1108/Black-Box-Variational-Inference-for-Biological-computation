# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:58:46 2019

Reference: https://www.kaggle.com/c/sberbank-russian-housing-market
R code: https://www.kaggle.com/captcalculator/a-very-extensive-sberbank-exploratory-analysis

@author: Tung1108
"""

import pandas as pd
import numpy as np

# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None

# Display up to 60 columns of dataframe 
pd.set_option('display.max_columns', 60)

import matplotlib as mpl
mpl.rc('axes', labelsize = 14)
mpl.rc('xtick', labelsize = 12)
mpl.rc('ytick', labelsize = 12)

import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 24

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import plotly.figure_factory as ff

from IPython.core.pylabtools import figsize
import seaborn as sns
from sklearn.model_selection import train_test_split

data = pd.read_csv('train.csv')
data.head()
data.info()
data.describe()

###############################################################################
############################### Missing Value #################################
###############################################################################

def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
            columns={0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
            '% of Total Values', ascending=False).round(1)
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n" 
          "There are " + str(mis_val_table_ren_columns.shape[0]) + 
          " columns that have missing values.")
    return mis_val_table_ren_columns

mis_value_tab = missing_values_table(data)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(13,18))
axes.barh(mis_value_tab.index, mis_value_tab['% of Total Values'], align='center')
axes.set_xlabel("%", fontsize=20)
axes.set_title('Percent missing data by feature', fontdict={'size':22})

fig, ax = plt.subplots(figsize=(12,12))
# plt.xticks(rotation='90')
sns.barplot(x=mis_value_tab['% of Total Values'], y=mis_value_tab.index)
ax.set(title='Percent missing data by feature', xlabel='% missing')

###############################################################################
########################### Data Quality Issues ###############################
###############################################################################

data.loc[data['state'] == 33, 'state'] = data['state'].mode().iloc[0]
data.loc[data['build_year'] == 20052009, 'build_year'] = 2007

###############################################################################
###############################################################################
#################### Housing Internal Characteristic ##########################
###############################################################################

internal_chars = ['full_sq', 'life_sq', 'floor', 'max_floor', 'build_year', 
                  'num_room', 'kitch_sq', 'state', 'price_doc']
data_internal_chars = data[internal_chars]
f, ax = plt.subplots(figsize=(10,6))
corr_internal_chars = data_internal_chars.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
hm_internal_chars = sns.heatmap(round(corr_internal_chars,2), annot=True, ax=ax,
                    cmap=cmap, fmt='.2f', annot_kws={"size": 12}, linewidths=.05)
f.subplots_adjust(top=0.93)
t = f.suptitle('Housing Internal Characteristic Correlation Heatmap', fontsize=15)

###############################################################################
#################### Area of Home and Number of Rooms #########################
###############################################################################

f, ax = plt.subplots(figsize=(10,6))
plt.scatter(x=data['full_sq'], y=data['price_doc'], alpha=0.4)
ax.set_xlabel("full_sq", fontsize=15)
ax.set_ylabel("price_doc", fontsize=15)
ax.set_title('Correlation between full_sq and price', fontdict={'size':20})

f, ax = plt.subplots(figsize=(10,6))
ind = data[data['full_sq'] > 2000].index
plt.scatter(x=data.drop(ind)['full_sq'], y=data.drop(ind)['price_doc'], alpha=0.5,
            edgecolors='w')
ax.set_xlabel("full_sq", fontsize=15)
ax.set_ylabel("price_doc", fontsize=15)
ax.set_title('Correlation between full_sq and price', fontdict={'size':20})

f, ax = plt.subplots(figsize=(10,6))
ind = data[data['full_sq'] > 2000].index
sns.stripplot(data.drop(ind)['full_sq'], data.drop(ind)['price_doc'],
              jitter=0.25, size=8, ax=ax, linewidth=.5, alpha=0.5, edgecolor="gray")
plt.title('Jittering with stripplot', fontsize=20)
plt.show()

# To avoid the problem of points overlap is the increase the size of the dot 
# depending on how many points lie in that spot.
# Larger the size of the point more is the concentration of points around that 
ind = data[data['full_sq'] > 2000].index
full_sq_counts = data.groupby([data.drop(ind)['full_sq'], data.drop(ind)['price_doc']]).size().reset_index(name='counts')
f, ax = plt.subplots(figsize=(40,100))
sns.stripplot(full_sq_counts.full_sq, full_sq_counts.price_doc, 
              size=full_sq_counts.counts*2, ax=ax)
plt.title('Counts plot - Size of circle is bigger as more points overlaps', fontsize=20)
plt.show()

ind = data[data['full_sq'] > 2000].index
sns.jointplot(data.drop(ind)['full_sq'], y=data.drop(ind)['price_doc'], 
                                 kind='reg', space=0, height=10, ratio=3)

(data['life_sq'] > data['full_sq']).sum()

f, ax = plt.subplots(figsize=(10,6))
sns.countplot(x=data['num_room'])
ax.set(title='Distribution of room count', xlabel='num_room')

###############################################################################
################################ Sale Types ###################################
###############################################################################

f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,8))
sns.distplot(data.loc[data['product_type'] == 'Investment', "price_doc"], color = "dodgerblue",
             label="Investment", hist_kws={'alpha':.7}, kde_kws={'linewidth':3}, ax=ax1)
ax1.set_ylabel('Density', fontsize=15)
ax1.set_title('Investment', fontsize=15)
sns.distplot(data.loc[data['product_type'] == 'OwnerOccupier', "price_doc"], color = "orange",
             label="OwnerOccupier", hist_kws={'alpha':.7}, kde_kws={'linewidth':3}, ax=ax2)
ax2.set_title('OwnerOccupier', fontsize=15)
plt.show()

data.groupby('product_type')['price_doc'].median()

###############################################################################
################################ Build Year ###################################
###############################################################################

f, ax = plt.subplots(figsize=(12,8))
plt.xticks(rotation='90')
ind = data[(data['build_year'] <= 1691) | (data['build_year'] >= 2018)].index
data_build = data.drop(ind).sort_values(by=['build_year'])
sns.countplot(x=data_build['build_year'])

# Relationship between the build_year and price_doc
# Group the data by year and take the mean of price_doc
f, ax = plt.subplots(figsize=(12,6))
price_year = data_build.groupby('build_year')[['build_year', 'price_doc']].mean()
sns.regplot(x="build_year", y="price_doc", data=price_year, scatter=False, 
            order=3, truncate=True)
plt.plot(price_year['build_year'], price_year['price_doc'], color='r')
ax.set_title('Mean Price by year of build', fontsize=14)

# The relationship appears somewhat steady over time, especially after 1960. 
# There is some volatility in the earlier years.
# This is not a real effect but simple due to the sparseness of observations around 1950

###############################################################################
################################ Timestamp ####################################
###############################################################################

# Question: How does the price vary over the time horizon of the data set 

f, ax = plt.subplots(figsize=(12,6))
# data['timestamp'] = data['timestamp'].astype(float)
price_ts = data.groupby('timestamp')[['price_doc']].mean()
#sns.regplot(x=price_ts.index, y="price_doc", data=price_ts, scatter=False, 
#            order=3, truncate=True)
plt.plot(price_ts['price_doc'], color='r')

import datetime
import matplotlib.dates as mdates
years = mdates.YearLocator()
yearsFmt = mdates.DateFormatter('%Y')
ts_vc = data['timestamp'].value_counts().sort_index()
f, ax = plt.subplots(figsize=(12,6))
sns.barplot(x=ts_vc.index, y=ts_vc, ax=ax)
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
#ax.set_xticklabels(labels=ts_vc.index, rotation=45, ha='right')
ax.set_xlabel("Time", fontsize=14)
ax.set_ylabel("Number of transaction", fontsize=14)
ax.set_title("Sales Volumnn over time", fontsize=20)

# Question: Is there a seasonal component to home prices in the course of a year 
f, ax = plt.subplots(figsize=(12,8))
data['timestamp'] = pd.to_datetime(data['timestamp'])
ts_season = data.groupby(by=[data.timestamp.dt.month])[['price_doc']].median()
plt.plot(ts_season.index, ts_season, color='g')
ax.set_xlabel("Months", fontsize=14)
ax.set_ylabel("Price", fontsize=14)
ax.set_title("A seasonal component to home price", fontsize=20)

###############################################################################
########################### Home State/Material ###############################
###############################################################################

f, ax = plt.subplots(figsize=(12,8))
ind = data[data['state'].isnull()].index
data['price_doc_log'] = np.log10(data['price_doc'])
ax = sns.violinplot(x="state", y="price_doc_log", data=data.drop(ind), inner=None, alpha=0.6)
ax = sns.stripplot(x="state", y="price_doc_log", data=data.drop(ind), jitter=True, 
                   color=".8", alpha=0.1)
ax.set_xlabel("State", fontsize=14)
ax.set_ylabel("Log10(price)", fontsize=14)
ax.set_title("Log10 of median price by state of home", fontsize=20)
ind = data[data['state'].isnull()].index
aver_price_state = data.drop(ind).groupby(by='state')[['price_doc']].mean()
aver_price_state

f, ax = plt.subplots(figsize=(12,8))
ind = data[data['material'].isnull()].index
data['price_doc_log'] = np.log10(data['price_doc'])
ax = sns.violinplot(x="material", y = "price_doc_log", data=data.drop(ind), inner=None)
ax = sns.stripplot(x="material", y = "price_doc_log", data=data.drop(ind), jitter=True,
                   color=".8", alpha=0.1)
ax.set_xlabel("Material", fontsize=14)
ax.set_ylabel("Log10(price)", fontsize=14)
ax.set_title("Distribution of price by build material", fontsize=20)
ind = data[data['material'].isnull()].index
aver_price_material = data.drop(ind).groupby(by='material')[['price_doc']].median()
aver_price_material

###############################################################################
############################### Floor of Home #################################
###############################################################################

# Question: How does the floor feature compare with price? 

col_floor = ['floor', 'max_floor', 'price_doc_log']

def corr_func(x, y, **kwargs):
    r = np.corrcoef(x,y)[0][1]
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r), 
                xy=(.2, .8), xycoords=ax.transAxes, size=20)
grid = sns.PairGrid(data=data[col_floor].dropna(), height=3)
grid.map_upper(plt.scatter, color='green', alpha = 0.25)
grid.map_diag(plt.hist, color='blue', edgecolor='black')
grid.map_lower(corr_func)
grid.map_lower(sns.kdeplot, cmap=plt.cm.Reds)

pp = sns.pairplot(data[col_floor], height=1.8, aspect=1.8,
             plot_kws=dict(edgecolor="k", linewidth=0.5),
             diag_kind="kde", diag_kws=dict(shade=True))
fig = pp.fig
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Correlation among floor features and Price', fontsize=14)

f, ax = plt.subplots(figsize=(12,8))
plt.scatter(x=data['floor'], y=data['price_doc_log'], c = 'r', alpha=0.15)
sns.regplot(x="floor", y="price_doc_log", data=data, scatter=False, truncate=True)
ax.set_xlabel("Floor", fontsize=14)
ax.set_ylabel("Log price", fontsize=14)
ax.set_title("Price by floor of home", fontsize=20)

f, ax = plt.subplots(figsize=(12,8))
plt.scatter(x=data['max_floor'], y=data['price_doc_log'], c = 'r', alpha=0.15)
sns.regplot(x="max_floor", y="price_doc_log", data=data, scatter=False, truncate=True)
ax.set_xlabel("Max Floor", fontsize=14)
ax.set_ylabel("Log Price", fontsize=14)
ax.set_title("Price by max floor", fontsize=20)

###############################################################################
###############################################################################

###############################################################################
####################### Demographic Characteristics ###########################
###############################################################################

demo_var = ['area_m', 'raion_popul', 'full_all', 'male_f', 'female_f', 'young_all',
            'young_female', 'work_all', 'work_male', 'work_female', 'price_doc']

data_demo_var = data[demo_var]
f, ax = plt.subplots(figsize=(10,8))
corr_demo_var= data_demo_var.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
hm_demo_var = sns.heatmap(round(corr_demo_var,2), annot=True, ax=ax, square=True,
                 cmap=cmap, fmt='.2f', annot_kws={'size':12}, linewidth=.05)
f.subplots_adjust(top=0.93)
t = f.suptitle('Housing Price & Demographic Correlation Heatmap', fontsize=15)

# How many unique districts are there? 
data['sub_area'].unique().shape[0]

# Calculate the population density and check to see the correlation of density and price
data['area_km'] = data['area_m'] / 1000000
data['density'] = data['raion_popul'] / data['area_km']
f, ax = plt.subplots(figsize=(12,8))
demo_price = data.groupby(by='area_km')[['density','price_doc']].median()
sns.regplot(x="density", y="price_doc", data=demo_price, scatter=True, 
            order=5, truncate=True)
ax.set_xlabel('Density', fontsize=14)
ax.set_ylabel('Price', fontsize=14)
ax.set_title('Median home price by raion population density', fontsize=20)

'''
These density numbers seem to make sense given that the population density of Moscow as
whole is 8537/sq km. There are a few raions that seem to have a density of near zero,
which seems odd. Home price does seem to increase with population density.

'''
# Question: how many sales transactions are in each district. 
f, ax = plt.subplots(figsize=(10,20))
trans_dis_raw = data['sub_area'].value_counts()
trans_dis = pd.DataFrame({'sub_area': trans_dis_raw.index, 'count': trans_dis_raw.values})
ax = sns.barplot(x="count", y="sub_area", data=trans_dis, orient="h")
ax.set_title('Number of Transaction by District', fontsize=20)
f.tight_layout()

trans_dis_sort = list(trans_dis_raw[trans_dis_raw.values > 900].index)
f, ax = plt.subplots(figsize=(12,8))
for trans_type in trans_dis_sort:
    subset = data[data['sub_area'] == trans_type]
    sns.kdeplot(subset['price_doc'].dropna(), 
                label=trans_type, shade=True, alpha=.25)
ax.set_xlabel('Price',fontsize=14)
ax.set_ylabel('Density', fontsize=14)
ax.set_title('Density of Price by sub areas', fontsize=20)

# Relationship between the share of the population that is working age and price
f, ax = plt.subplots(figsize=(12,8))
data['work_share'] = data['work_all']/data['raion_popul']
share_price = data.groupby(by='sub_area')[['work_share', 'price_doc']].mean()
sns.regplot(x="work_share", y="price_doc", data=share_price, scatter=True,
            order=5, truncate=True)
ax.set_xlabel('Work Shre', fontsize=14)
ax.set_ylabel('Price', fontsize=14)
ax.set_title('District mean home price by share of working age populaton', fontsize=20)

###############################################################################
###############################################################################

###############################################################################
########################### School Charateristics #############################
###############################################################################

school_var = ['children_preschool', 'preschool_quota', 'preschool_education_centers_raion',
              'children_school', 'school_quota', 'school_education_centers_raion',
              'school_education_centers_top_20_raion', 'university_top_20_raion',
              'additional_education_raion', 'additional_education_km', 'university_km',
              'price_doc']

data_school_var = data[school_var]
f, ax = plt.subplots(figsize=(12,8))
corr_school_var = data_school_var.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
hm_school_var = sns.heatmap(round(corr_school_var,2), annot = True, ax=ax, square=True,
                            cmap=cmap, fmt='.2f', annot_kws={'size':12}, linewidth=.05)
f.subplots_adjust(top=0.93)
t = f.suptitle('Housing Price & School Characteristics Heatmap', fontsize=20)

f, ax = plt.subplots(figsize=(12,8))
sns.stripplot(x="university_top_20_raion", y="price_doc", data=data, jitter=True,
              alpha=.15, color=".8")
sns.boxplot(x="university_top_20_raion", y="price_doc", data=data)
ax.set_xlabel('University top 20 raion', fontsize=14)
ax.set_ylabel('Price', fontsize=14)
ax.set_title('Distribution of home price of top university in Raion', fontsize=20)

'''
Homes in a raion with 3 top 20 universities have the highest median home price, however, 
it is fairly close among 0, 1, and 2. There are very few home with 3 top universities 

'''

# Question: How many districts there are with 3 universities 
data.loc[data['university_top_20_raion'] == 3, "sub_area"].unique()


###############################################################################
###############################################################################

###############################################################################
################### Cultural/Recretinal Characteristics #######################
###############################################################################

cult_vars = ['sport_objects_raion', 'culture_objects_top_25_raion', 'shopping_centers_raion',
             'park_km', 'fitness_km', 'swim_pool_km', 'ice_rink_km', 'stadium_km',
             'basketball_km', 'shopping_centers_km', 'big_church_km', 'mosque_km',
             'church_synagogue_km', 'theater_km', 'museum_km', 'exhibition_km',
             'catering_km', 'price_doc']
data_cult_vars = data[cult_vars]
f, ax = plt.subplots(figsize=(20,15))
corr_cult_vars = data_cult_vars.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
hm_cult_vars = sns.heatmap(round(corr_cult_vars,2), annot=True, ax=ax, square=True,
                           cmap=cmap, fmt='.2f', annot_kws={'size':12}, linewidth=.05)
f.subplots_adjust(top=0.93)
t = f.suptitle('Housing Price & Cultural Characteristics', fontsize=25)

f, ax = plt.subplots(figsize=(12,8))
sport_price = data.groupby(by='sub_area')[['sport_objects_raion','price_doc']].median()
sns.regplot(x="sport_objects_raion", y="price_doc", data=sport_price, scatter=True,
            truncate=True)
ax.set_xlabel('Sport Objects Raion', fontsize=14)
ax.set_ylabel('Price', fontsize=14)
ax.set_title('Median Raion home price of sports objects in Raion', fontsize=20)

f, ax = plt.subplots(figsize=(12,8))
cult_price_top = data.groupby(by='sub_area')[['culture_objects_top_25_raion', 'price_doc']].median()
sns.regplot(x="culture_objects_top_25_raion", y="price_doc", data=cult_price_top, 
            scatter=True, truncate=True)
ax.set_xlabel('Cultre Objects top 25', fontsize=14)
ax.set_ylabel('Price', fontsize=14)
ax.set_title('Median Raion home price of culture objects in Raion', fontsize=20)
data.groupby(by='culture_objects_top_25_raion')['price_doc'].median()

f, ax = plt.subplots(figsize=(12,6))
sns.stripplot(x="park_km", y="price_doc", data=data, jitter=0.25, 
              size=8, ax=ax, alpha=0.25, linewidth=.5)
ax.set_xlabel('Park', fontsize=14)
ax.set_ylabel('Price', fontsize=14)
ax.set_title('Home price by distance to nearest park', fontsize=20)

f, ax = plt.subplots(figsize=(12,8))
sns.regplot(x="park_km", y="price_doc", data=data, scatter=True, truncate=True,
            scatter_kws={'color':'r', 'alpha': .15})
ax.set_xlabel('Park', fontsize=14)
ax.set_ylabel('Price', fontsize=14)
ax.set_title('Home price by distance to nearest park', fontsize=20)


###############################################################################
###############################################################################

###############################################################################
########################## Infrastructure Features ############################
###############################################################################

infras_vars = ['nuclear_reactor_km', 'thermal_power_plant_km', 'power_transmission_line_km',
               'incineration_km','water_treatment_km', 'incineration_km', 'railroad_station_walk_km',   
               'railroad_station_walk_min', 'railroad_station_avto_km', 'railroad_station_avto_min',  
               'public_transport_station_km', 'public_transport_station_min_walk', 'water_km',
               'mkad_km', 'ttk_km', 'sadovoe_km','bulvar_ring_km', 'kremlin_km', 'price_doc']
data_infras_vars = data[infras_vars]
f, ax = plt.subplots(figsize=(15,10))
corr_infras_vars = data_infras_vars.corr()
cmap = sns.diverging_palette(220,10,as_cmap=True)
hm_infras_vars = sns.heatmap(round(corr_infras_vars,2), annot=True, ax=ax, square=True,
                             cmap=cmap, fmt='.2f', annot_kws={'size':10}, linewidth=.05)
f.subplots_adjust(top=0.93)
t = f.suptitle('Housing Price & Infrastructure Features', fontsize=20)

f, ax = plt.subplots(figsize=(12,8))
sns.regplot(x="kremlin_km", y="price_doc", data = data, scatter=True, truncate=True,
            scatter_kws={'color':'r', 'alpha':0.15})
ax.set_xlabel('Kremlin (km)', fontsize=14)
ax.set_ylabel('Price', fontsize=14)
ax.set_title('Home price by distance to Kremlin', fontsize=20)
