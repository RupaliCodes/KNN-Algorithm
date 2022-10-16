# KNN-Algorithm

## This programme demostrates the functioning of K Nearest Neighbours Algorithm. 
It does so from scratch, using modules like numpy, panda, math. It does not make use of the in-built knn classifier and related methods of python modules.

## USAGE
Currently it is working by taking data from the Iris Dataset. It is required to classify the input sample into the 3 varieties of Iris flower (Setosa , Versicolor and Virginica), by considering the various parameters like-

    1. sepal width 
    2. sepal length 
    3. petal width 
    4. petal length 

## METHODOLOGY
The following is a step by step explanation of what the program does:

### 1: Reading the iris.csv file
It reads the .csv file and stores it into the dataframe called dataset
It drops the ID column as it is redundant, and prints the remaining column.
 ```py
dataset = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '/dataset/iris.csv')
dataset.drop (["Id"], axis=1, inplace = True)
print (dataset)
```

### 2: Asking for Input sample 
```py
print ('enter the the sepal length, sepal width, petal length and petal width respectively : ')
myInputRow = np.zeros(4, float)
for i in range (4):
    myInputRow[i] = float (input ('Input the data point:'))
```

### 3: Kickstarting the knn algorithm process
```py
print(myInputRow)
predict = predict( dataset, myInputRow, 13)
```



