
import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
#dataframe = pd.read_csv("train.csv")
#dataframe["rooms"] = dataframe["rm"]
#dataframe["price"] = dataframe["medv"]

class MultiLinearRegression:

    def __init__ (self, x_input, y_input):
        self.x_input = x_input
        self.y_input = y_input


    def train(self,learning_rate,iter):
        #write the training logic


        alpha = learning_rate
        iterations = range(0,iter)

        # here x is columns

        X = self.x_input
        ones = np.ones([X.shape[0],1])
        X = np.concatenate((ones,X),axis=1)
        y = self.y_input.values
        theta = np.zeros([1,3])



        g, cost = self._gradient_descent(X, y, theta, alpha, iterations)
        # print("This is Theta [m,c]->",g,"This is cost->", cost)
        slope = g[0][0]
        slope1 = g[0][1]
        constant = g[0][2]
        data = {"slope" : slope,"slope1":slope1, "constant": constant}
        filehandler = open("multivariate_model.pickle",'wb')
        pickle.dump(data,filehandler)
        filehandler.close()



        #find optimal m,b and save it to a picke
        return True

    def predict(self,x1,x2):
       #read the picke and put it in y = mx+c
       file = open("multivariate_model.pickle",'rb')
       object_file = pickle.load(file)
       y = object_file['slope']* x1 + object_file['slope1']* x2  + object_file['constant']

       return y

    def evaluate(self,test_x, test_y):
        print(test_x)
        print(test_y)
        test_predicted_y = []
        for index, row in test_x.iterrows():
            test_predicted_y.append(self.predict(row[0],row[1]))
        sum_of_squares = 0
        for i,y_dash in enumerate(test_predicted_y):
            diff = y_dash - test_y['price'].iloc[i]
            diff_sq = diff * diff
            sum_of_squares = sum_of_squares + diff_sq
        rmse = math.sqrt(sum_of_squares / float(2*len(test_x)))
        return rmse



    def cost_calculation(self,X,y,theta):
        inner = np.power(((X @ theta.T) - y), 2) # @ means matrix multiplication of arrays. If we want to use * for multiplication we will have to convert all arrays to matrices
        cst =  np.sum(inner) / (2 * len(X))
        return cst
    def _gradient_descent(self,X, y, theta, alpha, iterations):

        cost_array = []
        for i in (iterations):
            theta = theta - (alpha/len(X)) * np.sum((X @ theta.T - y) * X, axis=0)
            cost = self.cost_calculation(X, y, theta)
            cost_array.append(cost)

        # print("Theta ---->",theta)
        # data1 = np.array(iterations).to_list()
        # print(data1)
        # plt.scatter(cost,iterations)
        # plt.show()
        return (theta, cost)
