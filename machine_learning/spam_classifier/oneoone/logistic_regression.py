#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 22:59:32 2018

@author: Anchit Jain
"""
import numpy as np
import pickle

class LogisticRegression:
    def __init__(self, lr, num_iter, threshold = 0.5):
        self.lr = lr
        self.num_iter = num_iter
        self.threshold = threshold


    def __add_intercept(self,X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def _sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def train(self,X,y):
        X = self.__add_intercept(X)

        # weights initialization
        self.theta = np.zeros(X.shape[1])
        print(self.theta)

        for i in range(self.num_iter):
           z = np.dot(X, self.theta)
           h = self._sigmoid(z)
           gradient = np.dot(X.T, (h - y)) / y.size
           self.theta -= self.lr * gradient

           z = np.dot(X, self.theta)
           h = self._sigmoid(z)
           loss = self.__loss(h, y)

           if( i % 100 == 0):
               print(f'loss: {loss} \t')
        filehandler = open("logistic_regression_model.pickle",'wb')
        pickle.dump(self.theta,filehandler)
        filehandler.close()

        return True

    def predict(self, X):
        file = open("logistic_regression_model.pickle",'rb')
        self.theta = pickle.load(file)
        X = self.__add_intercept(X)
        prob = self._sigmoid(np.dot(X, self.theta))
        return prob >= self.threshold
    def evaluate(self, test_x, test_y):
        y_predicted = self.predict(test_x)
        correct = 0
        for i,y in enumerate(test_y):
            if y == 0:
                y = False
            else:
                y = True
            if y == y_predicted[i]:
                correct = correct + 1
        total = y_predicted.size

        return (correct/total)*100
