#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 14:28:58 2018

@author: Huilin
"""
#import time
#start = time.clock()

import pandas as pd
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

data_train = pd.read_csv("mnist_binary.csv", index_col=False)
train_np = data_train.as_matrix()
y = train_np[:, 0]
X = train_np[:, 1:]

data = X
label = y

#%%
from sklearn.tree import DecisionTreeClassifier
n = np.array([1,4,10,20])  # different parameters
s = []; S = []
print('For DT classifier: ')
for i in range(4):
    neigh2 = DecisionTreeClassifier(max_depth=n[i])
    neigh2.fit(data,label) 
    scores = cross_val_score(neigh2,data, label, cv=10)    # Cross Validation 
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    s.append(1-scores.mean())
    S_terror = neigh2.score(data,label)
    S.append(1-S_terror)
plt.figure()
plt.plot(n,s)
plt.plot(n,S,'r')
plt.xlabel('depth')
plt.ylabel('error')
plt.title('decision tree classifier')
plt.legend(['cross validation','training error'])
plt.show()

#%%
from sklearn.neighbors import KNeighborsClassifier  # KNN classifier
n = np.array([1,5,15,100])  # different parameters
s = []; S = []
print('For KNN classifier: ')
for i in range(4):
    neigh = KNeighborsClassifier(n_neighbors=n[i])
    neigh.fit(data,label) 
    scores = cross_val_score(neigh,data, label, cv=10)    # Cross Validation 
    S_terror = neigh.score(data,label)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    s.append(1-scores.mean())
    S.append(1-S_terror)
plt.figure()
plt.plot(n,s)
plt.plot(n,S,'r')
plt.xlabel('Neighbours')
plt.ylabel('error')
plt.title('KNN classifier')
plt.legend(['cross validation','training error'])
plt.show()

#%% 

from sklearn.ensemble import RandomForestClassifier
n = np.array([1,5,10,50])  # different parameters
s = []; S = []
print('For Random Forest classifier: ')
for i in range(4):
    clf7 = RandomForestClassifier(n_estimators=n[i], max_depth=None, min_samples_split=2, random_state=0)
    clf7.fit(data,label)
    scores = cross_val_score(clf7,data, label, cv=10)    # Cross Validation 
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    s.append(1-scores.mean())
    S_terror = clf7.score(data,label)
    S.append(1-S_terror)
plt.figure()
plt.plot(n,s) 
plt.plot(n,S,'r')
plt.xlabel('number of estimators')
plt.ylabel('error')
plt.title('Random forest classifier')
plt.legend(['cross validation','training error'])
plt.show()

#end = time.clock()
#print ("Total time: %f s" % (end - start))


