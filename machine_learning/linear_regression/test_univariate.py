from oneoone import *

#data
data_x = [1,2,3,4,5,6,7,8,9,10,11,12,13]
data_y = [2,4,5,6,8,13,19,25,26,29,30,32,35]

#training data
taining_x = data_x[0:11]
training_y = data_y[0:11]

#test data
test_x = data_x[11:14]
test_y =  data_y[11:14]

ulr = UniLinearRegression(taining_x,training_y)

alpha = 0.07
iteration = 2
ulr.train(alpha,iteration) #returns nothing

print("Accuracy : ",ulr.evaluate(test_x,test_y))

x = 15
y_predicted = ulr.predict(x) #returns y


print(y_predicted)
