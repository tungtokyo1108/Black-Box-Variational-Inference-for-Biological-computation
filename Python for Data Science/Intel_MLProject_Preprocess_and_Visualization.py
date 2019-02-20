# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 09:15:41 2019

Reference: https://github.com/WillKoehrsen/machine-learning-project-walkthrough/blob/master/Machine%20Learning%20Project%20Part%201.ipynb

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

###############################################################################
######################### Data Cleaning and Formatting ########################
###############################################################################

data = pd.read_csv('Energy_and_Water_Data_Disclosure_for_Local_Law_84_2017__Data_for_Calendar_Year_2016_.csv')
data.info()

# Convert Data to Correct Types
data = data.replace({'Not Available': np.nan})
# Iterate through the columns 
for col in list(data.columns):
    if ('ft²' in col or 'kBtu' in col or 'Metric Tons CO2e' in col or 'kWh' in col 
        or 'therms' in col or 'gal' in col or 'Score' in col):
        data[col] = data[col].astype(float)
data.describe()

###############################################################################
############################### Missing Value #################################
###############################################################################

def missing_values_table(df):
    # Total missing values 
    mis_val = df.isnull().sum()
    
    # Percentage of missing values 
    mis_val_percent = 100 * df.isnull().sum()/len(df)
    
    # Make a table with the results 
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    
    # Rename the columns 
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 
                                                                1 : '% Total Values'})
    
    mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
            '% Total Values', ascending=False).round(1)
    
    print("Your selected dataframe has" + str(df.shape[1]) + "columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) + 
          "columns that have missing values")
    return mis_val_table_ren_columns

missing_values_table = missing_values_table(data)
missing_columns = list(missing_values_table[missing_values_table['% Total Values']>50].index)
print('We will remove %d columns. ' %len(missing_columns))
data = data.drop(columns = list(missing_columns))


###############################################################################
######################## Exploratory Data Analysis ############################
###############################################################################

figsize(8,8)
data = data.rename(columns = {'ENERGY STAR Score' : 'score'})
plt.style.use('fivethirtyeight')
plt.hist(data['score'].dropna(), bins=100, edgecolor='k')
plt.xlabel('Score')
plt.ylabel('Number of Buildings')
plt.title('Energy Star Score Distributions')

# Interact plot with plotly
trace1 = go.Histogram(x=data['score'].dropna())
data_iplot = [trace1]
layout = go.Layout(title='Energy Star Score Distributions', 
                   xaxis=dict(title='Score'),
                   yaxis=dict(title='Number of Buildings'),
                   bargap=0.2, bargroupgap=0.1)
fig = go.Figure(data=data_iplot, layout=layout)
plot(fig, filename='Energy Star Score Distributions')

figsize(8,8)
plt.hist(data['Site EUI (kBtu/ft²)'].dropna(), bins=20, edgecolor = 'black')
plt.xlabel('Site EUI')
plt.ylabel('Count')
plt.title('Site EUI Distribution')
data['Site EUI (kBtu/ft²)'].describe()
data['Site EUI (kBtu/ft²)'].dropna().sort_values().tail(10)

first_quartile = data['Site EUI (kBtu/ft²)'].describe()['25%']
third_quartile = data['Site EUI (kBtu/ft²)'].describe()['75%']
iqr = third_quartile - first_quartile
data = data[(data['Site EUI (kBtu/ft²)'] > (first_quartile - 3 * iqr)) &
            (data['Site EUI (kBtu/ft²)'] < (third_quartile + 3 * iqr))]
figsize(8,8)
plt.hist(data['Site EUI (kBtu/ft²)'].dropna(), bins=100, edgecolor = 'black')
plt.xlabel('Site EUI')
plt.ylabel('Count')
plt.title('Site EUI Distribution')

# Interact plot with plotly
trace1 = go.Histogram(x=data['Site EUI (kBtu/ft²)'].dropna())
data_iplot = [trace1]
layout = go.Layout(title='Site EUI Distribution', 
                   xaxis=dict(title='Site EUI'),
                   yaxis=dict(title='Count'),
                   bargap=0.2, bargroupgap=0.1)
fig = go.Figure(data=data_iplot, layout=layout)
plot(fig, filename='Site EUI Distribution')


###############################################################################
######################## Looking for relationships ############################
###############################################################################

types = data.dropna(subset=['score'])
types = types['Largest Property Use Type'].value_counts()
types = list(types[types.values > 100].index)
figsize(12,10)
for b_type in types:
    subset = data[data['Largest Property Use Type'] == b_type]
    sns.kdeplot(subset['score'].dropna(), 
               label = b_type, shade = False, alpha = 0.8)
plt.xlabel('Energy Star Score', size=20)
plt.ylabel('Density', size=20)
plt.title('Density Plot of Energy Star Score by Building Type', size=28)


boroughs = data.dropna(subset=['score'])
boroughs = boroughs['Borough'].value_counts()
boroughs = list(boroughs[boroughs.values > 100].index)
for borough in boroughs:
    subset = data[data['Borough'] == borough]
    sns.kdeplot(subset['score'].dropna(),
                label = borough)
plt.xlabel('Energy Star Score', size = 20)
plt.ylabel('Density', size = 20)
plt.title('Density Plot of Energy Star Scores by Borough', size=28)


###############################################################################
############# Correlations between Features and Target ########################
###############################################################################

correlations_data = data.corr()['score'].sort_values()
print(correlations_data.head(15),'\n')
print(correlations_data.tail(15))

numeric_subset = data.select_dtypes('number')
for col in numeric_subset:
    if col == 'score':
        next
    else:
        numeric_subset['sqrt_' + col] = np.sqrt(numeric_subset[col])
        numeric_subset['log_' + col] = np.log(numeric_subset[col])
categorial_subset = data[['Borough', 'Largest Property Use Type']]
categorial_subset = pd.get_dummies(categorial_subset)
features = pd.concat([numeric_subset, categorial_subset], axis=1)
features = features.dropna(subset=['score'])
correlations = features.corr()['score'].dropna().sort_values()
print(correlations.head(15), '\n')
print(correlations.tail(15))

ss = sns.jointplot(x='Site EUI (kBtu/ft²)', y='score', data = features,
                    kind="kde", height=8, space=0, ratio=3)

# Interact plot with plotly
colorscale = ['#7A4579', '#D56073', 'rgb(236,158,105)', (1,1,0.2), (0.98,0.98,0.98)]
fig = ff.create_2d_density(features['Site EUI (kBtu/ft²)'], features['score'],colorscale=colorscale,
                           hist_color='rgb(230,158,105)', point_size=3)
plot(fig, filename='Energy Star Score and Site EUI')


figsize(12,10)
features['Largest Property Use Type'] = data.dropna(subset=['score'])['Largest Property Use Type']
features = features[features['Largest Property Use Type'].isin(types)]
sns.lmplot('Site EUI (kBtu/ft²)', 'score', 
           hue = 'Largest Property Use Type', data = features,
           scatter_kws = {'alpha': 0.8, 's':60}, fit_reg = False,
           size = 12, aspect=1.2)
plt.xlabel("Site EUI", size = 28)
plt.ylabel("Energy Star Score", size = 28)
plt.title('Energy Star Score and Site EUI', size = 36)

# Interact plot with plotly
fig = {
       'data': [
               {'x': features[features['Largest Property Use Type']==stype]['Site EUI (kBtu/ft²)'],
                'y': features[features['Largest Property Use Type']==stype]['score'],
                'name': stype, 'mode':'markers',
                } for stype in types 
               ],
        'layout': {
                'title':'Energy Star Score and Site EUI',
                'xaxis':{'title': 'Site EUI'},
                'yaxis':{'title': 'Energy Star Score'}}
       }
plot(fig, filename='Energy Star Score and Site EUI')


###############################################################################
################################# Pairs Plot ##################################
###############################################################################

plot_data = features[['score', 'Site EUI (kBtu/ft²)', 
                      'Weather Normalized Site EUI (kBtu/ft²)',
                      'sqrt_Weather Normalized Source EUI (kBtu/ft²)']]
plot_data = plot_data.replace({np.inf: np.nan, -np.inf: np.nan})
plot_data = plot_data.rename(columns = {'Site EUI (kBtu/ft²)': 'Site EUI',
                                        'Weather Normalized Site EUI (kBtu/ft²)': 'Weather Norm EUI',
                                        'sqrt_Weather Normalized Source EUI (kBtu/ft²)': 'Sqrt Weather Norm EUI'})
plot_data = plot_data.dropna()
def corr_func(x, y, **kwargs):
    r = np.corrcoef(x,y)[0][1]
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r), 
                xy=(.2, .8), xycoords=ax.transAxes, size=20)
grid = sns.PairGrid(data=plot_data, height=3)
grid.map_upper(plt.scatter, color = 'green', alpha = 0.6)
grid.map_diag(plt.hist, color = 'blue', edgecolor = 'black')
grid.map_lower(corr_func);
grid.map_lower(sns.kdeplot, cmap=plt.cm.Reds)
plt.suptitle('Pair Plot of Energy Data', height=36, y=1.02)


# Interact Violin plot with plotly
iplot_data = []
for col in plot_data :
    trace = {
            "type": 'violin',
            "y": plot_data[col],
            "name": col,
            "box": {"visible": True},
            "meanline": {"visible":True}
            }
    iplot_data.append(trace)
fig = {
       "data": iplot_data,
       "layout": {
               "title": "Violin Plot of Energy Data",
               "yaxis": {
                      "zeroline":False, 
                }
               }
       }
plot(fig, filename='Violin Plot of Energy Data', validate=False)


plot_data_d = features[['score', 
                        'Site EUI (kBtu/ft²)', 
                      'Weather Normalized Site EUI (kBtu/ft²)',
                      'sqrt_Weather Normalized Source EUI (kBtu/ft²)',
                      'Largest Property Use Type']]
plot_data_d = plot_data_d.replace({np.inf: np.nan, -np.inf: np.nan})
plot_data_d = plot_data_d.rename(columns = {'Site EUI (kBtu/ft²)': 'Site EUI',
                                        'Weather Normalized Site EUI (kBtu/ft²)': 'Weather Norm EUI',
                                        'sqrt_Weather Normalized Source EUI (kBtu/ft²)': 'Sqrt Weather Norm EUI',
                                        })
plot_data_d = plot_data_d.dropna()

# Interact Split Violin plot with plotly
iplot_data_d = []
for col in plot_data:
    plot_data_d['new_'+col] = col
    Multifamily = {
            "type": 'violin',
            "x": plot_data_d['new_'+col][plot_data_d['Largest Property Use Type'] == 'Multifamily Housing'],
            "y": plot_data_d[col][plot_data_d['Largest Property Use Type'] == 'Multifamily Housing'] ,
            "legendgroup": 'Multifamily Housing',
            "scalegroup": 'Multifamily Housing',
            "name":'Multifamily Housing',
            "side" : 'negative',
            "box": {"visible": True},
            "meanline": {"visible":True},
            "line": {"color":'#8dd3c7'},
            "marker": {"line":{"with":2, "color": '#8dd3c7'}}
            }
    iplot_data_d.append(Multifamily)
    Office = {
            "type": 'violin',
            "x": plot_data_d['new_'+col][plot_data_d['Largest Property Use Type'] == 'Office'],
            "y": plot_data_d[col][plot_data_d['Largest Property Use Type'] == 'Office'],
            "legendgroup": 'Office',
            "scalegroup": 'Office',
            "name": 'Office',
            "side" : 'positive',
            "box": {"visible": True},
            "meanline": {"visible":True},
            "line": {"color":'#bebada'},
            "marker": {"line":{"with":2, "color": '#bebada'}}
            }
    iplot_data_d.append(Office)
    
fig = {
        "data": iplot_data_d,
        "layout": {
               "title": "Split Violin Plot of Energy Data",
               "yaxis": {
                      "zeroline":False, 
                },
               "violingap": 0,
               "violingroupgap": 0,
               "violinmode": "overlay"
        }
}
plot(fig, filename='Split Violin Plot of Energy Data', validate=False)


###############################################################################
###################### Feature Engineering and Selection ######################
###############################################################################

features = data.copy()
numeric_subset = data.select_dtypes('number')
for col in numeric_subset.columns:
    if col == 'score':
        next
    else:
        numeric_subset['log_' + col] = np.log(numeric_subset[col])
categorial_subset = data[['Borough', 'Largest Property Use Type']]
categorial_subset = pd.get_dummies(categorial_subset)
features = pd.concat([numeric_subset, categorial_subset], axis = 1)
features.shape

# Remove Collinear Features
plot_data = data[['Weather Normalized Site EUI (kBtu/ft²)','Site EUI (kBtu/ft²)']].dropna()
plt.plot(plot_data['Site EUI (kBtu/ft²)'], 
                   plot_data['Weather Normalized Site EUI (kBtu/ft²)'], 'bo')
plt.xlabel('Site EUI (kBtu/ft²)')
plt.ylabel('Weather Normalized Site EUI (kBtu/ft²)')
plt.title('Weather Norm EUI vs Site EUI, R = %0.4f' % np.corrcoef(data[[
        'Weather Normalized Site EUI (kBtu/ft²)', 'Site EUI (kBtu/ft²)']].dropna(),
        rowvar=False)[0][1])
