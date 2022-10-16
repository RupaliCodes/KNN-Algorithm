# this is a programme to implement k nearest neighbours algorithm

from math import sqrt
import pandas as pd
import os
import numpy as np
  
dataset = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '/dataset/iris.csv')
dataset.drop (["Id"], axis=1, inplace = True)
print (dataset)

def eucl_dist(row1, row2):
    distance = 0.0
    for i in range (len(row1)-1): # -1 because it does not want label column
        distance += (row1[i]-row2[i])**2
    return sqrt (distance)

def get_neighbours (dataset, myInputRow, num):
    
    distan = list()

    for j in range(dataset.shape[0]):
        dist = eucl_dist(dataset.iloc[j], myInputRow)
        distan.append((dataset.iloc[j], dist))

    distan.sort(key= lambda n : n [1])

    neighbours = list()

    for i in range (num):
        neighbours.append(distan[i][0])
    return neighbours

def predict (dataset,myInputRow, num):
    neighbours = get_neighbours (dataset,myInputRow, num)
    output = [i[-1] for i  in neighbours]
    prediction = max(set (output), key= output.count)
    return prediction 

print ('enter the the sepal length, sepal width, petal length and petal width respectively : ')
myInputRow = np.zeros(4, float)
for i in range (4):
    myInputRow[i] = float (input ('Input the data point:'))

print(myInputRow)
predict = predict( dataset, myInputRow, 13)

print('This', str(myInputRow),' sample may belong to the variety: ', predict)