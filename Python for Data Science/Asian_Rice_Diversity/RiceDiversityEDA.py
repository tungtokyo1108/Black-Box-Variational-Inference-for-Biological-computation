# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:41:27 2019

@author: Tung1108
"""

import pandas as pd
import numpy as np 

pd.options.mode.chained_assignment = None

pd.set_option('display.max_columns', 60)
import matplotlib as mpl
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 24

from IPython.core.pylabtools import figsize
import seaborn as sns

pheno = pd.read_csv('RiceDiversityPheno.csv')
pheno.info()
pheno_des = pheno.describe()

line = pd.read_csv('RiceDiversityLine.csv')
data = pd.concat([pheno, line], axis=1, sort=False)

# Missing Value 
def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
            columns={0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
            '% of Total Values', ascending=True).round(1)
    print("Your selected dataframe has " + str(df.shape[1]) + " columns. \n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) + 
          " columns that have missing values")
    return mis_val_table_ren_columns

mis_val_tab = missing_values_table(data)
plt.figure(figsize=(12,12))
ax = plt.gca()
features = mis_val_tab.index
per_miss = mis_val_tab['% of Total Values'].values
nums = np.arange(len(mis_val_tab))
plt.barh(nums, per_miss)
for p, c, ch in zip(nums, features, per_miss):
    plt.annotate(str(ch), xy=(ch + 1, p), va='center', fontsize=10)
ticks = plt.yticks(nums, features)
xt = plt.xticks()[0]
plt.xticks(xt, [' '] * len(xt))
plt.grid(axis = 'x', color='white', linestyle='-')
plt.xlabel('% Missing Value', fontsize=15)
plt.title('Percent missing data by features', fontsize=20)
ax.tick_params(axis='both', which='both', length=0)
sns.despine(left=True, bottom=True)

# Canonical correlation analysis 
pheno_data = pheno.drop(columns=['HybID', 'NSFTVID'])
plt.figure(figsize=(20,20))
corr_pheno = pheno_data.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
hm_pheno = sns.heatmap(round(corr_pheno,2), annot=True,
                       cmap="coolwarm", fmt='.1f', annot_kws={"size":10}, linewidths=.05)
# plt.title("Pairwise correlations of phenotypes across accessions", fontsize=20)

# Flowering time 
flowering_time = ['Flowering time at Arkansas', 'Flowering time at Faridpur', 
                  'Flowering time at Aberdeen', 'FT ratio of Arkansas/Aberdeen',
                  'FT ratio of Faridpur/Aberdeen']
data_flower = pheno[flowering_time]
data_flower.rename(columns={'Flowering time at Arkansas': 'Arkansas', 
                            'Flowering time at Faridpur': 'Faridpur',
                            'Flowering time at Aberdeen': 'Aberdeen',
                            'FT ratio of Arkansas/Aberdeen': 'Arkansas/Aberdeen', 
                            'FT ratio of Faridpur/Aberdeen': 'Faridpur/Aberdeen'}, inplace=True)

def corr_func(x, y, **kwargs):
    r = np.corrcoef(x,y)[0][1]
    ax = plt.gca()
    ax.annotate("corr = {:.2f}".format(r), 
                xy={.2, .8}, xycoords=ax.transAxes, size=20)
grid = sns.PairGrid(data=data_flower.dropna(), height=3)
grid.map_upper(sns.regplot, color='green')
grid.map_diag(plt.hist, color='blue', edgecolor='black', alpha=0.25)
grid.map_lower(corr_func)
grid.map_lower(sns.kdeplot, cmap=plt.cm.Reds)    

data_flower['SubPop'] = data['Sub-population']

## Arkansas
fig, ax = plt.subplots(figsize=(10,10), sharey=True)
ax = sns.boxplot(x="SubPop", y="Arkansas", data=data_flower, color="white", width=.5)
plt.setp(ax.artists, edgecolor='black', facecolor='w', alpha=1)
plt.setp(ax.lines, color='black', alpha=1)
ax = sns.stripplot(x="SubPop", y="Arkansas", data=data_flower, jitter=True,
                   alpha=0.8)
plt.show()
plt.figure(figsize=(10,10))
sns.distplot(data_flower['Arkansas'].dropna(), bins=50,
             hist_kws={"linewidth": 3, "alpha": 0.3, "color": "blue"},
             kde_kws={"color":"r", "lw": 2, "alpha": 0.5})

## Faridpur
plt.figure(figsize=(10,10))
ax = sns.boxplot(x="SubPop", y="Faridpur", data=data_flower, color="white", width=.5)
plt.setp(ax.artists, edgecolor='black', facecolor='w', alpha=1)
plt.setp(ax.lines, color='black', alpha=1)
ax = sns.stripplot(x="SubPop", y="Faridpur", data=data_flower, jitter=True, 
                   alpha=0.8)
plt.figure(figsize=(10,10))
sns.distplot(data_flower['Faridpur'].dropna(), bins=50,
             hist_kws={"linewidth": 3, "alpha": 0.3, "color": "blue"},
             kde_kws={"color": "r", "lw": 2, "alpha": 0.5})

# Morphology 
morphology = ['Culm habit', 'Leaf pubescence', 'Flag leaf length', 
              'Flag leaf width', 'Awn presence']
data_morphology = pheno[morphology]
data_morphology.rename(columns={'Culm habit':'Habit', 
                                'Leaf pubescence': 'Pubescence',
                                'Flag leaf length': 'Length',
                                'Flag leaf width': 'Width', 
                                'Awn presence': 'Awn'}, inplace=True)
def corr_func(x, y, **kwargs):
    r = np.corrcoef(x,y)[0][1]
    ax = plt.gca()
    ax.annotate("corr = {:.2f}".format(r),
                xy={.2, .8}, xycoords=ax.transAxes, size=20)
grid = sns.PairGrid(data=data_morphology.dropna(), height=3)
grid.map_upper(sns.regplot, color='green')
grid.map_diag(plt.hist, color='blue', edgecolor='black', alpha=0.25)
grid.map_lower(corr_func)
grid.map_lower(sns.kdeplot, cmap=plt.cm.Reds)
    
data_morphology['SubPop'] = data['Sub-population']

## Culm habit
plt.figure(figsize=(10,10))
sns.boxplot(x="SubPop", y="Habit", data=data_morphology, color="white", width=.5)
plt.setp(ax.artists, edgecolor='black', facecolor='w', alpha=1)
plt.setp(ax.lines, color='black', alpha=1)
ax = sns.stripplot(x="SubPop", y="Habit", data=data_morphology, jitter=True, alpha=0.8)
plt.figure(figsize=(10,10))
sns.distplot(data_morphology['Habit'].dropna(), bins=50, 
             hist_kws={"linewidth": 3, "alpha": 0.3, "color": "blue"}, 
             kde_kws={"color": "r", "lw": 2, "alpha": 0.5})






























