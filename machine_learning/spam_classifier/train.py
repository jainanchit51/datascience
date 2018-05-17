import numpy as np
from numpy import array
import pandas as pd
from oneoone import LogisticRegression
import pickle

## This method will read the data and appends data to a list
def read(filename):
    all_messages = []
    all_categories = []
    df = pd.read_csv(filename)
    messsages = (df.iloc[:,1:2])
    categories = (df.iloc[:,0:1])
    messsages_to_list = (messsages.values.tolist())
    categories_to_list = (categories.values.tolist())

    for message in messsages_to_list:
        all_messages.append(message[0])
    for category in categories_to_list:
        if category[0] == "spam":
            all_categories.append(1)
        else:
            all_categories.append(0)


    return (all_messages, all_categories)


## This method will read data and convert string to vectors
def create_unique_dict(text_arr):
    dictt = {}
    count = -1
    x = np.array(text_arr)
    text1 = (np.unique(x))

    for item in text1:
        data = (item.lower().strip().split(" "))
        for item1 in data:
            if item1 not in dictt:
                count = count +1
                dictt[item1] = count
    print((dictt))
    return dictt

def vectrorize(text_arr,dictt):
    vectors  = []
    for sentence in text_arr:
        vector = [0] * len(dictt)
        words = (sentence.lower().strip().split(" "))
        for word in words:
            index = dictt[word]
            vector[index] += 1
        vectors.append(vector)
    print(array(vectors))
    return vectors


#loading total text
total_x, total_y = read("sms_spam.csv")
dictt = create_unique_dict(total_x)

#creating dictionary from total text to pickle
filehandler = open("dictionary.pickle",'wb')
pickle.dump(dictt,filehandler)
filehandler.close()


#reading train data and vectorizing it
x, y = read("train.csv")
x = np.array(vectrorize(x, dictt))
y = np.array(y)


#reading test data and vectorizing it
test_x, test_y = read("test.csv")
test_x = np.array(vectrorize(test_x, dictt))
test_y = np.array(test_y)


lr=0.0001
num_iter=10000
logr = LogisticRegression(lr,num_iter, 0.5)

logr.train(x,y)

print(logr.evaluate(test_x,test_y))
