# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 11:10:30 2019

@author: Tung1108
@Reference: https://github.com/dipanjanS/practical-machine-learning-with-python/blob/master/bonus%20content/effective%20data%20visualization/Bonus%20-%20Effective%20Multi-dimensional%20Data%20Visualization.ipynb
https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/

"""
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
import seaborn as sns 
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import plotly.figure_factory as ff
import joypy

######################################################################################
############################ Load and merge datasets #################################
######################################################################################

white_wine = pd.read_csv('winequality-white.csv', sep=';')
red_wine = pd.read_csv('winequality-red.csv', sep=';') 

red_wine['wine_type'] = 'red'
white_wine['wine_type'] = 'white'

red_wine['quality_label'] = red_wine['quality'].apply(lambda value: 'low'
                                                      if value <= 5 else 'medium'
                                                      if value <= 7 else 'high')
red_wine['quality_label'] = pd.Categorical(red_wine['quality_label'],
                                            categories=['low', 'medium', 'high'])
white_wine['quality_label'] = white_wine['quality'].apply(lambda value: 'low'
                                                          if value <= 5 else 'medium'
                                                          if value <= 7 else 'high')
white_wine['quality_label'] = pd.Categorical(white_wine['quality_label'],
                                              categories=['low','medium','high'])

# Merge datasets 
wines = pd.concat([red_wine, white_wine])

# Re-shuffle records just to randomize data points 
wines = wines.sample(frac=1, random_state=42).reset_index(drop=True)
wines.head()


######################################################################################
################# Exploratory Data Analysis and Visualization ########################
######################################################################################

########################### Descriptive Statistics ###################################

subset_attributes = ['residual sugar', 'total sulfur dioxide', 'sulphates', 'alcohol', 
                     'volatile acidity', 'quality']
rs = round(red_wine[subset_attributes].describe(),2)
ws = round(white_wine[subset_attributes].describe(),2)
rs_ws = pd.concat([rs,ws], axis=1, keys=['Red Wine Statistics', 'White Wine Statistics'])

subset_attributes_total = ['alcohol', 'volatile acidity', 'pH', 'quality']
ls = round(wines[wines['quality_label'] == 'low'][subset_attributes_total].describe(),2)
ms = round(wines[wines['quality_label'] == 'medium'][subset_attributes_total].describe(),2)
hs = round(wines[wines['quality_label'] == 'high'][subset_attributes_total].describe(),2)
ls_ms_hs = pd.concat([ls,ms,hs], axis=1, keys=['Low Quality Wine', 'Medium Quality Wine', 
                                                 'High Quality Wine'])

for col in wines :
    red_average = wines[wines['wine_type']=='red'].mean()
    white_average = wines[wines['wine_type']=='white'].mean()
wine_average = pd.concat([red_average, white_average], axis=1, 
                         keys=['Red Wine Average', 'White Wine Average'])
wine_average = wine_average.T
n_rows = len(wine_average)
index = np.arange(len(wine_average.columns)) + 0.3
bar_width = 0.4
y_offset = np.zeros(len(wine_average.columns))
p1 = plt.bar(red_average.index, red_average, bar_width)
p2 = plt.bar(white_average.index, white_average, bar_width, bottom=red_average)
    
########################### Visualizing one dimension ################################

wines.hist(bins=15, color='steelblue', edgecolor='black',linewidth=1.0,
           xlabelsize=8, ylabelsize=8, grid=False)
plt.tight_layout(rect=(0,0,1.2,1.2))

fig = plt.figure(figsize = (6,4))
title = fig.suptitle("Sulphates Content in Wine", fontsize = 14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1,1)
ax.set_xlabel("Sulphates")
ax.set_ylabel("Frequency")
ax.text(1.2, 800, r'$\mu$=' + str(round(wines['sulphates'].mean(),2)), fontsize=12)
freq, bins, patches = ax.hist(wines['sulphates'], color='steelblue', bins=50,
                              edgecolor='black', linewidth=1)

fig = plt.figure(figsize = (6,4))
title = fig.suptitle("Sulphates Content in Wine", fontsize = 14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel("Sulphates")
ax1.set_ylabel("Density")
sns.kdeplot(wines['sulphates'], ax=ax1, shade=True, color='steelblue')

wines_types = wines['wine_type'].value_counts() 
red_wine_quality_types = red_wine['quality_label'].value_counts()
white_wine_quality_types = white_wine['quality_label'].value_counts()
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(23,10))
axes[0].pie(wines_types,shadow=True, autopct='%1.1f%%', labels=wines_types.index)
axes[0].set_title('Percentage of Wine Read and White')
axes[1].pie(red_wine_quality_types,shadow=True, autopct='%1.1f%%', labels=red_wine_quality_types.index)
axes[1].set_title('Percentage of Quality of Red Wine')
axes[2].pie(white_wine_quality_types,shadow=True, autopct='%1.1f%%', labels=white_wine_quality_types.index)
axes[2].set_title('Percentage of Quality of White Wine')

########################### Visualizing two dimensions ###############################

f, ax = plt.subplots(figsize=(10,6))
corr = wines.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm", fmt='.2f',
                 linewidths = .05)
f.subplots_adjust(top=0.93)
t = f.suptitle('Wine Attributes Correlation Heatmap', fontsize = 14)

cols = ['density', 'residual sugar', 'total sulfur dioxide', 'fixed acidity']
pp = sns.pairplot(wines[cols], size=1.8, aspect=1.8, 
                  plot_kws=dict(edgecolor="k", linewidth = 0.5), 
                  diag_kind="kde", diag_kws=dict(shade=True))
fig = pp.fig
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Wine Attributes Pairwise Plots', fontsize=14)

from sklearn.preprocessing import StandardScaler
subset_df = wines[cols]
ss = StandardScaler()
scaled_df = ss.fit_transform(subset_df)
scaled_df = pd.DataFrame(scaled_df, columns=cols)
final_df = pd.concat([scaled_df, wines['wine_type']], axis=1)
final_df.head()

plt.scatter(wines['sulphates'], wines['alcohol'], 
            alpha=0.4, edgecolors='w')
plt.xlabel('Sulphates')
plt.ylabel('Alcohol')
plt.title('Wine Sulphates - Alcohol Content', y=1.05)

# Jittering with stripplot 
# Use jittered plots to avoid overlapping of points 
fig, ax = plt.subplots(figsize=(20,8), dpi=80)
sns.stripplot(wines['sulphates'], wines['alcohol'], jitter=0.25, size=8, ax=ax, linewidth=.5)

# To avoid the problem of points overlap is the increase the size of the dot 
# depending on how many points lie in that spot.
# Larger the size of the point more is the concentration of points around that 
wines_counts = wines.groupby(['sulphates', 'alcohol']).size().reset_index(name='counts')
fig, ax = plt.subplots(figsize=(16,10), dpi=80)
sns.stripplot(wines_counts.sulphates, wines_counts.alcohol, size=wines_counts.counts*2, ax=ax)
plt.title('Counts plot - Size of circle is bigger as more points overlaps', fontsize=22)
plt.show()

jp = sns.jointplot(x='sulphates', y='alcohol', data = wines, 
                   kind='reg', space=0, size=5, ratio=4)

cp = sns.countplot(x="quality", hue="wine_type", data=wines, 
                   palette={"red": "#FF9999", "white": "#FFE888"})

# Interact plotly 
x=wines['sulphates']
y=wines['alcohol']
data = [
    go.Histogram2dContour(
            x=x,
            y=y,
            colorscale = 'Blues',
            reversescale = True,
            xaxis = 'x',
            yaxis = 'y'),
    go.Scatter(
            x=x,
            y=y,
            xaxis = 'x',
            yaxis = 'y',
            mode = 'markers',
            marker = dict(color='rgba(0,0,0,0.3)', size = 3)
            ),
    go.Histogram(
        y = y,
        xaxis = 'x2',
        marker = dict(
            color = 'rgba(0,0,0,1)'
        )
    ),
    go.Histogram(
        x = x,
        yaxis = 'y2',
        marker = dict(
        color = 'rgba(0,0,0,1)'
        )
    )
]
layout = go.Layout(
    title = 'Wine Sulphates - Alcohol Content',
    autosize = False,
    xaxis = dict(
        title='sulphates',
        zeroline = False,
        domain = [0,0.85],
        showgrid = False
    ),
    yaxis = dict(
        title='alcohol',    
        zeroline = False,
        domain = [0,0.85],
        showgrid = False
    ),
    xaxis2 = dict(
        zeroline = False,
        domain = [0.85,1],
        showgrid = False
    ),
    yaxis2 = dict(
        zeroline = False,
        domain = [0.85,1],
        showgrid = False
    ),
    height = 600,
    width = 600,
    bargap = 0,
    hovermode = 'closest',
    showlegend = False
)
fig = go.Figure(data=data, layout=layout)
plot(fig,filename='Wine Sulphates - Alcohol Content')

colorscale = ['#7A4579', '#D56073', 'rgb(236,158,105)', (1,1,0.2), (0.98,0.98,0.98)]
fig = ff.create_2d_density(wines['sulphates'], wines['alcohol'],colorscale=colorscale,
                           hist_color='rgb(230,158,105)', point_size=3)
plot(fig, filename='Wine Sulphates - Alcohol Content')


# The distribution of two variabels                    
fig = plt.figure(figsize = (6,4))
title = fig.suptitle("Sulphates Content in Wine", fontsize = 14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1,1)
ax.set_xlabel("Sulphates")
ax.set_ylabel("Frequency")
g = sns.FacetGrid(wines, hue='wine_type', palette={"red": "r", "white": "y"})
g.map(sns.distplot, 'sulphates', kde=False, bins=35, ax=ax)
ax.legend(title='Wine Type')
plt.close(2)

# Density Curves with Histogram 
plt.figure(figsize=(8,5), dpi=80)
sns.distplot(wines.loc[wines['wine_type'] == 'red', "sulphates"], color = "dodgerblue", 
             label="red", hist_kws={'alpha':.7}, kde_kws={'linewidth':3})
sns.distplot(wines.loc[wines['wine_type'] == 'white', "sulphates"], color = "orange", 
             label="white", hist_kws={'alpha':.7}, kde_kws={'linewidth':3})
plt.title('Density Plot of Wines by the Wine types', fontsize=12)
plt.legend()
plt.show()

# Interact plot for the distribution of two variables 
hist_data = [red_wine['sulphates'], white_wine['sulphates']]
group_labels = ['red_wine', 'white_wine']
fig = ff.create_distplot(hist_data, group_labels, bin_size=.01, curve_type='kde')
fig['layout'].update(title='Displot with Red-White wine', xaxis=dict(title='Sulphates'))
plot(fig, filename='Displot with Red-White wine')

# The bax plot of two variables
f, (ax) = plt.subplots(1,1,figsize=(12,4))
f.suptitle('Wine Quality - Alcohol Content', fontsize=14)
sns.boxplot(x="quality", y="alcohol", data=wines, ax=ax)
ax.set_xlabel("Wine Quality", size = 12, alpha=0.8)
ax.set_ylabel("Wine Alcohol", size = 12, alpha=0.8)

# The Violin plot of two variables
f, (ax) = plt.subplots(1,1,figsize=(12,4))
f.suptitle('Wine Quality - Sulphates Content', fontsize=14)
sns.violinplot(x="quality", y="sulphates", data=wines, ax=ax)
ax.set_xlabel("Wine Quality", size=12, alpha=0.8)
ax.set_ylabel("Wine Sulphates", size=12, alpha=0.8)

########################## Visualizing three dimensions #######################

cols = ['density', 'residual sugar', 'total sulfur dioxide', 'fixed acidity', 'wine_type']
pp =  sns.pairplot(wines[cols], hue='wine_type', height=1.8, aspect=1.8,
                   palette="husl", markers=["o","D"],
                   plot_kws=dict(edgecolor="black", linewidth=0.5))

plt.figure(figsize=(10,8), dpi=80)
pp1 = sns.pairplot(wines[cols], hue='wine_type', kind="scatter", plot_kws=dict(s=80, 
                   edgecolor="white",linewidth=2.5))
# pp2 = sns.pairplot(wines[cols], hue='wine_type', kind="reg")
plt.show()

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
xs = wines['residual sugar']
ys = wines['fixed acidity']
zs = wines['alcohol']
ax.scatter(xs,ys,zs, s=50, alpha=0.6, edgecolors='w')
ax.set_xlabel('Residual Sugar')
ax.set_ylabel('Fixed Acidity')
ax.set_zlabel('Alcohol')

# Interac plot for three dimensions visualization
xs = wines['residual sugar']
ys = wines['fixed acidity']
zs = wines['alcohol']
trace1 = go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode = 'markers',
            marker=dict(
                    size=5,
                    color=xs,   # set color to an array/list of desired values
                    colorscale='Viridis',
                    #color='rgb(227,227,227)',
                    #symbol='circle',
                    #line=dict(
                    #        color='rgb(204,204,204)',
                    #        width=1
                    #),
                    opacity=0.8
            )
        )
data = [trace1]
fig = go.Figure(data=data)
plot(fig, filename='R-F-A')


fc = sns.factorplot(x="quality", hue="wine_type", col="quality_label",
                    data=wines, kind="count",
                    palette={"red": "#FF9999", "white":"#FFE888"})

jp = sns.pairplot(wines, x_vars=["sulphates"], y_vars=["alcohol"], size=4.5,
                  hue="wine_type", palette="husl", markers=["o", "X"], height=5,aspect=1.6,
                  plot_kws=dict(edgecolor="k", linewidth=0.7, s=60))

# Joy plot allows the density curves of different group to overlap
plt.figure(figsize=(8,6), dpi=80)
fig, axes = joypy.joyplot(wines, column=['sulphates','alcohol'], by="wine_type", ylim='own',
                          figsize=(8,6))
plt.title('Joy Plot of Sulphates and alcohol by Wine types', fontsize = 12)
plt.show()

# Scatter plot with linear regression line of best fit
lp = sns.lmplot(x='sulphates', y='alcohol', hue='wine_type', 
                palette="husl", height=5, aspect=1.6, data=wines, fit_reg=True, legend=True,
                scatter_kws=dict(edgecolor="k", linewidth=0.7, s=60))

ax=sns.kdeplot(white_wine['sulphates'], white_wine['alcohol'], cmap="YlOrBr",
               shade=True, shade_lowest=False)
ax=sns.kdeplot(red_wine['sulphates'], red_wine['alcohol'], cmap="Reds", 
               shade=True, shade_lowest=False)

f, (ax1, ax2) = plt.subplots(1,2, figsize = (14,4))
f.suptitle('Wine Type - Quality - Alcohol Content', fontsize = 14)
sns.boxplot(x="quality", y="alcohol", hue="wine_type", 
            data=wines, palette="husl", ax=ax1)
ax1.set_xlabel("Wine Quality", size=12, alpha=0.8)
ax1.set_ylabel("Wine Alcohol %", size=12, alpha=0.8)
sns.boxplot(x="quality_label", y="alcohol", hue="wine_type",
            data=wines, palette="husl", ax=ax2)
ax2.set_xlabel("Wine Quality Class", size=12, alpha=0.8)
ax2.set_ylabel("Wine Alcohol %", size=12, alpha=0.8)
l = plt.legend(loc='best', title='Wine Type')


############################### Visualizing four dimensions ##############################

cols = ['density', 'residual sugar', 'total sulfur dioxide', 'fixed acidity', 'quality_label']
pp =  sns.pairplot(wines[cols], hue='quality_label', height=1.8, aspect=1.8,
                   palette="husl", markers=["o","D","P"],
                   plot_kws=dict(edgecolor="black", linewidth=0.5))

fig = plt.figure(figsize=(8,6))
t = fig.suptitle('Wine Residual Sugar - Alcohol Content - Acidity - Type', fontsize=14)
ax = fig.add_subplot(111, projection='3d')
xs = list(wines['residual sugar'])
ys = list(wines['alcohol'])
zs = list(wines['fixed acidity'])
data_points = [(x,y,z) for x,y,z in zip(xs,ys,zs)]
colors = ['red' if wt == 'red' else 'green' for wt in list(wines['wine_type'])]
for data, color in zip(data_points, colors) : 
    x,y,z = data
    ax.scatter(x,y,z, alpha=0.4, c=color, edgecolors='none', s=30)
ax.set_xlabel('Residual Sugar')
ax.set_ylabel('Alcohol')
ax.set_zlabel('Fixed Acidity')

g = sns.FacetGrid(wines, col="wine_type", hue='quality_label', 
                  col_order=['red','white'], hue_order=['low','medium','high'],
                  aspect=1.2, height=3.5, palette=sns.light_palette('navy',4)[1:])
g.map(plt.scatter, "volatile acidity", "alcohol", alpha=0.9,
      edgecolor='white', linewidth=0.5, s=100)
fig = g.fig
fig.subplots_adjust(top=0.8, wspace=0.3)
fig.suptitle('Wine Type - Acohol - Quality - Acidity', fontsize=14)
l = g.add_legend(title='Wine Quality Class')

g = sns.FacetGrid(wines, col="wine_type", hue='quality_label', 
                  col_order=['red','white'], hue_order=['low','medium','high'],
                  aspect=1.2, height=3.5, palette=sns.light_palette('green', 4)[1:])
g.map(plt.scatter, "alcohol", "total sulfur dioxide", alpha=0.9,
      edgecolor='white', linewidth=0.5, s=100)
fig = g.fig
fig.subplots_adjust(top=0.8, wspace=0.3)
fig.suptitle('Wine Type - Sulfur Dioxide - Quality - alcohol', fontsize=14)
l = g.add_legend(title='Wine Quality Class')
