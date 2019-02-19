#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 18:06:12 2018

@author: Huilin
"""

import titanic
import pickle

import pandas as pd
data_train = pd.read_csv("titanic_filled.csv",header=None)
train_np = data_train.as_matrix()
y = train_np[:, 0]
X = train_np[:, 1:]
#X = data_train.iloc[:, 1:].values
#y = data_train.iloc[:, 0].values

new = titanic.titanicClassifier(C=100)
new.fit(X,y)

classifier = {'nickname': '+V$%⎖℃', 'classifier': new}
pickle.dump(classifier, open('titanic_classifier.pkl', 'wb'))
#%%
import pickle
import mnist
import pandas as pd
data_train = pd.read_csv("mnist_binary.csv", index_col=False)
train_np = data_train.as_matrix()
y = train_np[:, 0]
X = train_np[:, 1:]


new = mnist.mnistClassifier(n_estimators=50)

new.fit(X,y)
classifier = {'nickname': '+V$%⎖℃', 'classifier': new}
pickle.dump(classifier, open('mnist_classifier.pkl', 'wb'))
