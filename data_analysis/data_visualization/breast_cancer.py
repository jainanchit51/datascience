#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 00:03:49 2018

@author: ist
"""

import  pandas as pd
dataframe = pd.read_csv('breast-cancer-wisconsin.csv', engine='python')
dataframe = dataframe.drop(['bare_nucleoli'], axis=1)
print(dataframe.describe())




## To check the correlativity between attributes   
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='ticks', color_codes=True)
plt.figure(figsize=(14, 12))
sns.heatmap(dataframe.astype(float).corr(), 
            linewidths=0.1, 
            square=True, 
            linecolor='white', 
            annot=True)
plt.show()


##Bar plot 1 
fig = plt.figure()
ax = sns.countplot(x='bland_chromatin', 
                   hue='class', 
                   #palette={0:'#EB434A', 1:'#61A98F'}, 
                   data=dataframe)
ax.set(xlabel='Bland Chromatin', ylabel='No of cases')
fig.suptitle("Bland Chromatin w.r.t. Class", y=0.96);

##Bar plot 2 
fig = plt.figure()
ax = sns.countplot(x='clump_thickness', 
                   hue='class', 
                   #palette={0:'#EB434A', 1:'#61A98F'}, 
                   data=dataframe)
ax.set(xlabel='Thickness of Clump', ylabel='Total')
fig.suptitle("clump_thickness w.r.t. Class", y=0.96);