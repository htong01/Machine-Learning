#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 15:20:52 2018

@author: Huilin
"""

import pandas as pd
import numpy as np

header = ['user_id', 'item_id', 'rating', 'timestamp']
data = pd.read_csv('u.data', sep='\t', names=header)
header2 = ['user id', 'age', 'gender', 'occupation', 'zip code']
user = pd.read_csv('u.user', sep='|', names=header2)
item = pd.read_csv('u.item', sep='|' , header=None,encoding='latin-1')
item = item.drop(item.columns[[0,3]], axis=1)
item.iat[266,1] = '16-Jul-1989'



#%%
from surprise import SVD
from surprise.model_selection import cross_validate
from surprise import Dataset, Reader

reader=Reader(line_format='user item rating timestamp',sep='\t')
data=Dataset.load_from_file('u.data',reader=reader)
algo = SVD()
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


from collections import defaultdict
def get_top_n(predictions, n=10):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

trainset = data.build_full_trainset()
algo = SVD(n_factors=25) # and the defult reg_all is 0.02
algo.fit(trainset)
# Than predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
predictions = algo.test(testset)
top_n = get_top_n(predictions, n=5)

#for uid, user_ratings in top_n.items():
#    print(uid, [iid for (iid, _) in user_ratings])

with open("Recommendation.txt", "w") as text_file:
    for uid, user_ratings in top_n.items():
        print(uid, [iid for (iid, _) in user_ratings], file=text_file)

#%% get the user and movie vector
from surprise import SVD
from surprise.model_selection import cross_validate
from surprise import Dataset, Reader
from surprise.model_selection import GridSearchCV

reader=Reader(line_format='user item rating timestamp',sep='\t')
data=Dataset.load_from_file('u.data',reader=reader)
trainset = data.build_full_trainset()
algo = SVD(n_factors=25)
algo.fit(trainset)
U = algo.pu  # user vector
V = algo.qi  # movie vector
#%%
# Grid search for the best parameter
param_grid = {'n_factors':[5,10,25,50],'reg_all':[0.02,0.4,1] }
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)
print(gs.best_params['rmse'])
#%% User age
user_age = user.iloc[:,1]
user_age = user_age.as_matrix()
age = np.zeros(len(U))
for i in range(len(U)):
    a = int(trainset.to_raw_uid(i))
    age[i] = user_age[a-1]

from scipy import stats
K = 25
corref_U = []
for i in range(K):
    a = stats.pearsonr(U[:,i], age)
    corref_U.append(a)
y = []
for i in range(K):
    y.append(abs(corref_U[i][0]))
 
m = max(y) # max correlatin coefficient
print(m)
ind = y.index(max(y)) # index of the max element



import matplotlib.pyplot as plt
plt.figure()
plt.plot(age,U[:,ind],'.')
plt.xlabel('age')
#plt.ylabel('correlation coefficient')
plt.title('user age and U')
plt.show()


#%% user gender
from scipy import stats

user_gender = user.iloc[:,2]
user_gender = user_gender.as_matrix() 
for i in range(len(user_gender)):
    if user_gender[i] == 'M':
        user_gender[i] = int(0)
    else:
        user_gender[i] = int(1)
      

gender = np.zeros(len(U))
for i in range(len(U)):
    a = int(trainset.to_raw_uid(i))
    gender[i] = user_gender[a-1]


corref_G = []
for i in range(K):
    a = stats.pearsonr(U[:,i], gender)
    corref_G.append(a)
X=[i for i in range(K)]
y = []
for i in range(K):
    y.append(abs(corref_G[i][0]))
    
    
m = max(y) # max correlatin coefficient
print(m)
ind = y.index(max(y)) # index of the max element

    
import matplotlib.pyplot as plt
plt.figure()
plt.plot(gender,U[:,ind],'.')
plt.xlabel('gender: 0 stands for male and 1 stands for female')
#plt.ylabel('correlation coefficient')
plt.title('user gender and U')
plt.show()


#%% release year

release_year = item.iloc[:,1]
release_year = release_year.str.split('-')
release_year = release_year.as_matrix()
release = np.zeros(len(release_year))
for i in range(len(release_year)):
    release[i] = int(release_year[i][2])


year = np.zeros(len(V))
for i in range(len(V)):
    a = int(trainset.to_raw_iid(i))
    year[i] = release[a-1]



corref_V = []
for i in range(K):
    a = stats.pearsonr(V[:,i], year)
    corref_V.append(a)

y = []
for i in range(K):
    y.append(corref_V[i][0])
    
m = max(y) 
print(m)
ind = y.index(max(y)) 
    
import matplotlib.pyplot as plt
plt.figure()
plt.plot(year,V[:,ind],'.')
plt.xlabel('year')
#plt.ylabel('correlation coefficient')
plt.title('release year and V')
plt.show()



