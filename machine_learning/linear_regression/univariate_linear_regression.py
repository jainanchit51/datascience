#Reading the dataset
data = pd.read_csv("train.csv")
#Renaming the rm to rooms
data["rooms"]= data["rm"]
#renaming price and sclaing by 1000
data["price"] = data["medv"] *1

#parameter initialization
m = 1
b = -33.5

#Hyperparameters
learning_rate = 0.1
iterations = np.arange(0,100)

#tracking costs per iteration
costs_per_iteration = []
final_m = 0
final_b = 0
training_mode = False

if training_mode == True:
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
        tune_model(var_rmse)
    
    plt.plot(iterations, costs_per_iteration,color='blue')
    plt.show()

    print("Lowest Loss :", np.array(costs_per_iteration).min())
    print("Lowest m :", final_m)
    print("Lowest b :", final_b)

    #saving the model in a picle file
    model_parameters = {}
    model_parameters['m']  = final_m
    model_parameters['b']  = final_b
    f = open('model.pickle', 'wb')
    cPickle.dump(model_parameters,f)

inp = input("Enter the room number to predict the price")
print("Price is : ", model(inp))

    
    
