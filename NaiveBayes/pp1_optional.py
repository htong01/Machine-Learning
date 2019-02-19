
def train(X_train, Y_train, train_opt):
    m = train_opt
    p0Num = []; p1Num = []
    l = len(X_train)
    Y_train = list(map(int, Y_train))
    for i in range(l):  
        if Y_train[i] == 0:  
            p0Num += X_train[i]   # list of words belong to negative label
        else:
            p1Num += X_train[i]
    p = []
    for i in range(len(X_train)):
        p += X_train[i]
    list1 = []
    for i in p:
        if not i in list1:
            list1.append(i)
#    list1.remove('the'); list1.remove('it'); list1.remove('i'); list1.remove('and'); list1.remove('is')
    d = len(list1) 
    d1 = d #Vocabulary Size
    list1 = list1[0:d1]
    dir0 = Counter(p0Num) # count how many times one word occur in the file
    dir1 = Counter(p1Num)
    p0 = []; p1 = []; p = 0
    
    for i in list1:
        p11 = (dir1[i] + m)/(m * d1 + np.sum(Y_train) )
        p1.append(p11)
        p00 = (dir0[i] + m)/(m * d1 + len(Y_train)-np.sum(Y_train) )
        p0.append(p00) # Probability one word occur given condition that label is 0

    
    p_y1 = np.sum(Y_train)/float(l) # probability of label is 1
    return p_y1, list1, p0, p1
#%%
def test(X_test , p_y1, list1, p0, p1):
    l = len(X_test)
    pt0 = 0; pt1 = 0; p = np.zeros(l); y_pred = []; g = np.zeros(l)
    a0 = np.log(p_y1) - np.log(1-p_y1)
    for i in range(l):
        for n in range(len(X_test[i])):
            for k in range(len(list1)):
                if X_test[i][n] == list1[k]:
                    pt0 = np.log(p0[k])
                    pt1 = np.log(p1[k])
                    p[i] += pt1 - pt0
                    g[i] = p[i] + a0
        if g[i] > 0:
            lab = 1
        else:
            lab = 0
        y_pred.append(lab)
    return y_pred
#%%
def evaluate(y_test, y_pred):
    y_test = list(map(int, y_test))
    l = len(y_test)
    error_rate = np.ones(l)
    for i in range(l):
        error_rate[i] = np.abs(y_pred[i] - y_test[i])
    errorrate = np.sum(error_rate)/l
    return errorrate


#%%
def main(f):
    x = []; y = []
    for line in f.readlines():
            line = line.lower()
            line = line.strip()# remove space
            st = line.split("\t")
            data = st[0]
            data = data.strip("!.,?%$&-)(")# remove ! . ?
            remove_digits = str.maketrans('', '', digits)
            data = data.translate(remove_digits)
            data = data.split()
            label = st[1]
            x.append(data)
            y.append(label)
    k = ([0.1, 0.5, 0.8, 0.95])
    
    a = []; a2 = []
    for n in range(4):
        error = []; error2 = []; 
        for i in range(10):
            X_train , X_test , Y_train , Y_test = train_test_split(x, y, test_size=0.1)
            X_vald , X_train , Y_vald , Y_train = train_test_split(X_train, Y_train, test_size=k[n])
            
            p_y1, list1, p0, p1 = train(X_train, Y_train, 0.1)
            y_pred = test(X_vald , p_y1, list1, p0, p1)
            errorrate = evaluate(Y_vald, y_pred)
            y_pred2 = test(X_test , p_y1, list1, p0, p1)
            errorrate2 = evaluate(Y_test, y_pred2)
    #            print('The error rate of my classifier is ' + str(errorrate))
            error.append(errorrate)
            error2.append(errorrate2)
        er = np.sum(error)/10
        er2 = np.sum(error2)/10
        a.append(er)
        a2.append(er2)
        
    plt.figure()
    plt.plot(k,a,label='validation')
    plt.plot(k,a2,label='real test',color='red')
    plt.xlabel('Ration')
    plt.ylabel('Error')
    plt.show()
    
#%%
import numpy as np
from sklearn.model_selection import train_test_split
from string import digits
import matplotlib.pyplot as plt
from collections import Counter

f = open("amazon_cells_labelled.txt")
main(f)
v = open("yelp_labelled.txt")
main(v)
p = open("imdb_labelled.txt")
main(p)

