
import numpy as np
import pickle
import math

#dataframe = pd.read_csv("train.csv")
#dataframe["rooms"] = dataframe["rm"]
#dataframe["price"] = dataframe["medv"]

class UniLinearRegression:

    def __init__ (self, x_input, y_input):
        self.x_input = x_input
        self.y_input = y_input


    def train(self,learning_rate,iter):
        #write the training logic


        alpha = learning_rate
        iterations = range(0,iter)
        theta = np.array([[1.0, 1.0]])
        # here x is columns

        X = (np.array(self.x_input).reshape(-1,1)) # -1 tells numpy to figure out the dimension by itself
        ones = np.ones([X.shape[0], 1])
        X = np.concatenate([ones, X],1)

        # y is a columns vector
        y = (np.array(self.y_input).reshape(-1,1))

        g, cost = self._gradient_descent(X, y, theta, alpha, iterations)
        
        # print("This is Theta [m,c]->",g,"This is cost->", cost)
        slope = g[0][0]
        constant = g[0][1]
        data = {"slope" : slope, "constant": constant}
        filehandler = open("model.pickle",'wb')
        pickle.dump(data,filehandler)
        filehandler.close()



        #find optimal m,b and save it to a picke
        return True

    def predict(self,x):
       #read the picke and put it in y = mx+c
       file = open("model.pickle",'rb')
       object_file = pickle.load(file)
       y = object_file['slope']* x + object_file['constant']

       return y

    def evaluate(self,test_x, test_y):
        test_predicted_y = []
        for x in test_x:
            test_predicted_y.append(self.predict(x))
        sum_of_squares = 0
        for i,y_dash in enumerate(test_predicted_y):
            diff = y_dash - test_y[i]
            diff_sq = diff * diff
            sum_of_squares = sum_of_squares + diff_sq
        accuracy = math.sqrt(sum_of_squares / float(2*len(test_x)))
        return accuracy



    def cost_calculation(self,X,y,theta):
        inner = np.power(((X @ theta.T) - y), 2) # @ means matrix multiplication of arrays. If we want to use * for multiplication we will have to convert all arrays to matrices
        return np.sum(inner) / (2 * len(X))

    def _gradient_descent(self,X, y, theta, alpha, iterations):
        for i in (iterations):
            theta = theta - (alpha/len(X)) * np.sum((X @ theta.T - y) * X, axis=0)
            cost = self.cost_calculation(X, y, theta)
            #cost_array.append(cost)

            return (theta, cost)
