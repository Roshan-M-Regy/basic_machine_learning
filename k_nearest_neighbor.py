# Roshan Mammen Regy, CSE 633 HW1 Q2
# K-Nearest Neighbor
# Load required libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

# Read training data set 
columns = ['Age','Year of Operation','# of lymph nodes', 'Class']
dtrain = pd.read_csv('data_train.csv',names = columns)

# Calculate l2norm 
def calc_l2norm(x1, x2):
    l2norm = 0
    for i,val in enumerate(x1):
        l2norm += np.power(val-x2[i],2)
    return l2norm

# Define K-NN classifier
def knn_classifier(K, query, data):
    l2arr = np.zeros((data.shape[0],2))
    for i in range(data.shape[0]):
        l2arr[i,0] = calc_l2norm(query,data.iloc[i,:len(query)])
        l2arr[i,1] = data.iloc[i,-1]
    sortl2arr = l2arr[np.argsort(l2arr[:,0])]
    classes, counts = np.unique(sortl2arr[:K,1], return_counts=True)
    for i,val in enumerate(counts):
        if val==max(counts):
            queryclass = classes[i]
            break
    return (queryclass)
        

# Read testing data set
columns = ['Age','Year of Operation','# of lymph nodes', 'Class']
dtest = pd.read_csv('data_dev.csv',names = columns)
uniqclass, uniqcounts = np.unique(dtest.iloc[:,-1],return_counts=True)
# Get distance matrix 
distmat = np.zeros((dtest.shape[0],dtrain.shape[0]))
for i in range(


Klist = [1,3,5,7,9,11,13]
Acc = np.zeros(len(Klist))
Bacc = np.zeros(len(Klist))
for i,K in enumerate(Klist):
    print ('K = %s'%K)
    knnclass = np.zeros(dtest.shape[0])
    acc = 0
    bacc = np.zeros(uniqclass.shape)
    for j in range(dtest.shape[0]):
        knnclass[j] = knn_classifier(K, dtest.iloc[j,:3], dtrain)
        if knnclass[j] == dtest.iloc[j,-1]:
            print ('%s == %s'%(knnclass[j],dtest.iloc[j,-1]))
            acc += 1
            for l,clas in enumerate(uniqclass):
                if knnclass[j] == clas:
                    bacc[l] += 1
        else:
            print ('%s != %s'%(knnclass[j],dtest.iloc[j,-1]))
    Acc[i] = acc /dtest.shape[0]
    for l,val in enumerate(bacc):
        Bacc[i] += 1/(len(uniqclass))*val/uniqcounts[l]

# Create Accuracy metric dataframe
#accdf = pd.DataFrame(data, columns = ['Name', 'Age'])
print (Klist)
print (Acc)
print (Bacc)
