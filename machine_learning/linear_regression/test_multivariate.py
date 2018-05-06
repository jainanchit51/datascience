from oneoone import *
import pandas as pd
import numpy as np

# reading data
dataframe  = pd.read_csv('home.txt',names=["size","bedroom","price"])


# normalizing data

dataframe = (dataframe - dataframe.mean())/dataframe.std()
print(dataframe.head())
train=dataframe.sample(frac=0.8,random_state=200)

training_x = dataframe.iloc[0:38,0:2]
training_y = dataframe.iloc[0:38,2:3]
print((training_y))

test=dataframe.drop(train.index)






#training data
test_x = dataframe.iloc[38:48,0:2]
test_y =  dataframe.iloc[38:48,2:3]
print("Testdata",test_x)

mlr = MultiLinearRegression(training_x,training_y)

#set hyper parameters
alpha = 0.01
iteration = 1000
mlr.train(alpha,iteration) #returns nothing
print("training done")
rmse = mlr.evaluate(test_x,test_y)
print("rmse :",rmse)

x1 = 15
x2 = 30
y_predicted = mlr.predict(x1,x2) #returns y

print("Y-Predicted:",y_predicted)
