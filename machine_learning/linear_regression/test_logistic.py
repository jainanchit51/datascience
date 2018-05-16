
from oneoone import *

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import datasets



iris = datasets.load_iris()
X = iris.data[:, :]
y = (iris.target != 0) *1

print(type(X))
# lr=0.001
# num_iter=10000
# logr = LogisticRegression(lr,num_iter, 0.5)
#
# logr.train(X,y)
#
# print(logr.evaluate(X,y))
#
# print(logr.predict(np.array([[1,1,1,1],[0.1,0.1,0.1,0.1]])))
