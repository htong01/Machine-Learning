#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 14:01:54 2018

@author: Huilin
"""

import pandas as pd
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

data_train = pd.read_csv("titanic_filled.csv")
train_np = data_train.as_matrix()
y = train_np[:, 0]
X = train_np[:, 1:]

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(X)
#for i in range(9):
#    a = max(X[:, i])
#    b = min(X[:, i])
#    X[:, i] = (X[:, i] - b)/(a-b) # Normalization
#    
#data = X
label = y

# Feature selection

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X_new = SelectKBest(chi2, k=9).fit_transform(data, label)  # select the most relevant features


data = X_new


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


#%% SVM
from sklearn.svm import SVC
n = np.array([0.1,2,10,100])  # different parameters
s = []; S = []
print('For SVM: ')
for i in range(len(n)):
    clf5 = SVC(C=n[i])
    clf5.fit(data,label)
    scores = cross_val_score(clf5,data, label, cv=10)    # Cross Validation 
    S_terror = clf5.score(data,label)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    s.append(1-scores.mean())
    S.append(1-S_terror)
plt.figure()
plt.plot(n,s)
plt.plot(n,S,'r')
plt.xlabel('C')
plt.ylabel('error')
plt.title('SVM classifier')
plt.legend(['cross validation','training error'])
plt.show()

#%% RandomForest
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



