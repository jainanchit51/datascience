from statistics import mean
import math
from matplotlib import style
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def cost_calution(data):
    
    data['difference'] = (data['price_predicted'] - data['price'])
    data['differece_sqr']= data['difference'] * data['difference']
    sum_of_difference = data['differece_sqr'].sum()
    rmse = math.sqrt(sum_of_difference)
    print(rmse)
    return rmse

def predict_y(m,b,data):
    col1 = data['rooms']
    col2 = data['price']
    x_original = np.array(col1, dtype=np.float64)
    y_original = np.array(col2, dtype=np.float64)
    y_predicted = [(m*x)+b for x in x_original]
    data['price_predicted'] = y_predicted
    
    style.use('ggplot')
    plt.scatter(x_original,y_original,color='orange')
    plt.plot(x_original, y_predicted)
    plt.show()
    return m,b
    
    
def tune_model(data):
    #if var_rmse <= var_rmse:
    #    print("Anchit")
    global m
    global b
    global theta
    
    X = data['rooms'].values.reshape(-1,1)
    ones = np.ones([X.shape[0], 1])
    X = np.concatenate([ones, X],1)
    y = data['price'].values.reshape(-1,1)
    ones = np.ones([y.shape[0],1])
    y = np.concatenate([ones,y],1)
    theta = theta - (learning_rate/len(X)) * np.sum((X @ theta.T - y) * X, axis=0)
    m = theta[0][0]
    b = theta[0][1]
    
       


#Reading the dataset
data = pd.read_csv("train.csv")
#Renaming the rm to rooms
data["rooms"]= data["rm"]
#renaming price and sclaing by 1000
data["price"] = data["medv"] *1

#parameter initialization
theta = np.array([[1.0, 1.0]])
m = theta[0][0]
b = theta[0][1]

#Hyperparameters
learning_rate = 0.0001
iterations = np.arange(0,100)

#tracking costs per iteration
costs_per_iteration = []
final_m = 0
final_b = 0



for i in iterations:
    #predict Y with existing parameters(m,b)
    predict_y(m,b,data)
      
    #caculate the error with the loss function
    print(m)
    var_rmse = cost_calution(data)
    costs_per_iteration.append(var_rmse)
    
    #finding lowest loss m,b
    if var_rmse <= np.array(costs_per_iteration).min():
        final_m = m
        final_b = b
    
    #now change the model paramerters to reduce loss
    tune_model(data)
    
plt.plot(iterations, costs_per_iteration,color='blue')
plt.show()

print("Lowest Loss :", np.array(costs_per_iteration).min())
print("Lowest m :", final_m)
print("Lowest b :", final_b)



    
    
    
    
    
    
    
    
    