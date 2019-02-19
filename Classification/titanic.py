#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 23:55:03 2018

@author: Huilin
"""

from sklearn.svm import SVC

class titanicClassifier(SVC):
        
    def score(self,X,y):
#        for i in range(9):
#            a = max(X[:, i])
#            b = min(X[:, i])
#            X[:, i] = (X[:, i] - b)/(a-b) # Normalization       
#       
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        X = min_max_scaler.fit_transform(X)
        return super().score(X, y)

