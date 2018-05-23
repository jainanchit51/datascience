#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 00:01:30 2018

@author: ist
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

dataset = pd.read_csv("train.csv")


x_train = dataset.iloc[0:21000,1:]
labels  =dataset.iloc[0:21000,0]
print(labels)