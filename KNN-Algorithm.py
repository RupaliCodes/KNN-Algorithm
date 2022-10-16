# this is a programme to implement k nearest neighbours algorithm

from math import sqrt
import pandas as pd
import os
import numpy as np
#from sklearn.preprocessing import LabelEncoder 
  
dataset = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '/dataset/iris2.csv')
dataset.drop (["Id"], axis=1, inplace = True)
print (dataset)



def eucl_dist(row1, row2):
    distance = 0.0
    for i in range (len(row1)-1): # why the -1 ? because it does not want label column
        distance += (row1[i]-row2[i])**2
    return sqrt (distance)

#import the dataset here
# test row specify
"""
for i in range(dataset.shape[0]):
    distance = eucl_dist(myrow, dataset.iloc[i])
    print (distance)
"""
# locate the nearest neighbours
# row is the normal training dataset row 
# myrow is the test row/ new row
# num is the k in the knn, i.e. the numbers of nearest neighbour you want

def get_neighbours (dataset, myrow, num):
    
    distan = list()

    for j in range(dataset.shape[0]):
        dist = eucl_dist(dataset.iloc[j], myrow)
        distan.append((dataset.iloc[j], dist))

    distan.sort(key= lambda n : n [1]) #how does the n/tup work?

    neighbours = list()

    for i in range (num):
        neighbours.append(distan[i][0])
    return neighbours

#neighbours  = get_neighbours( dataset, myrow, 3)

#for neighbour in neighbours: 
#    print (neighbour)
# specify the neighbours parameters to get the result 

def predict (dataset,myrow, num):
    neighbours = get_neighbours (dataset,myrow, num)
    output = [i[-1] for i  in neighbours] # what does this give in a form of list?
    prediction = max(set (output), key= output.count)  # what is the role of the set function here?
    return prediction 

# specify the predict parameters here
#myrow = dataset.iloc[19]
print ('enter the the sepal length and petal length respectively : ')
myrow = np.zeros(4, float)
for i in range (4):
    myrow[i] = float (input ('Input the data point:'))

print(myrow)
predict = predict( dataset, myrow, 13)

print('Expected %d, Got %d' % (myrow[-1], predict)) 
# Instead of a dataset row, as in the example, if we had a single row , how would the last element of the row be represented? x[-1] ?

