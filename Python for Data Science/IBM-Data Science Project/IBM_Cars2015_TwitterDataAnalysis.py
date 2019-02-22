# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 09:47:07 2019

Reference: https://github.com/ibm-watson-data-lab/spark.samples/blob/master/notebook/DashDB%20Twitter%20Car%202015%20Python%20Notebook.ipynb

@author: Tung1108
"""

from __future__ import division 

import nltk
nltk.download("stopwords")

from pyspark.sql import SQLContext
from pyspark.sql import functions as F
from pyspark import SparkConf, SparkContext
import time
from datetime import date
from dateutil import parser
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from IPython.core.display import Javascript

sc = SparkContext.getOrCreate(SparkConf())
sqlContext = SQLContext(sc)


car_makers_list = [['bmw'], ['daimler', 'mercedes'], ['gm', 'general motors'], ['tesla'], ['toyota'], ['vw', 'volkswagen']]

car_makers_name_list = []
for car_maker in car_makers_list:
    car_makers_name_list.append(car_maker[0].upper())

#plotting variables
ind = np.arange(len(car_makers_list)) #index list for plotting
width = 0.8       # the width of the bars in the bar plots

num_car_makers = len(car_makers_list)

##car features #support English, Deutsch, french, Spanish
electric_vehicle_terms = ['electric car', 'electric vehicle', 'electric motor', 'hybrid vehicle', 'Hybrid car', 'elektroauto', 'elektrofahrzeug', 
                          'hybridautos', 'voiture hyprid', 'coche híbrido', 'Auto Hibrido', 'vehículo híbrido', 'elektrovehikel', 'voiture électrique', 'coche eléctrico']
auto_driver_terms = ['auto drive', 'autodrive', 'autonomous', 'driverless', 'self driving', 'robotic', 'autonomes', 'selbstfahrendes', 'autonome', 'autónomo']

SCHEMA="DASH7504."
PREFIX="PYCON_"

def GeoChart(data_string, element):
    return Javascript("""
        //container.show();
        function draw() {{
          var chart = new google.visualization.GeoChart(document.getElementById(""" + element + """));
          chart.draw(google.visualization.arrayToDataTable(""" + data_string + """));
        }}
        google.load('visualization', '1.0', {'callback': draw, 'packages':['geochart']});
        """, lib="https://www.google.com/jsapi")

def addMissingDates(baseDates, checkedDates):
    temp = checkedDates.copy()
    checkedDatesValues = checkedDates['POSTING_TIME']
    for index, row in baseDates.iterrows():
        if (not row['POSTING_TIME'] in checkedDatesValues.tolist()):
            row['NUM_TWEETS'] = 0
            temp = temp.append(row)
    return temp.sort('POSTING_TIME')

props = {}
props['user'] = 'dash7504'
props['password'] = 'vtl2yZi1LX9n'

jdbcurl='jdbc:db2://dashdb-entry-yp-dal09-07.services.dal.bluemix.net:50000/BLUDB'

#get the data frame
df_TWEETS = sqlContext.read.jdbc(jdbcurl, SCHEMA + PREFIX+'TWEETS', properties=props)
df_TWEETS.printSchema()

df_SENTIMENTS = sqlContext.read.jdbc(jdbcurl, SCHEMA + PREFIX + 'SENTIMENTS', properties=props)
df_SENTIMENTS.printSchema()

print("Number of Tweets: " + str(df_TWEETS.count()))
print("Number of Sentiment Records: " + str(df_SENTIMENTS.count()))
