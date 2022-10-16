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

### 3: Knn process works by calculating the euclidean distance between any two rows
Here, it will be used to calculate the distance between the input sample row and each of the different rows in the given dataset respectively.
```py 
def eucl_dist(row1, row2):
    distance = 0.0
    for i in range (len(row1)-1): # -1 because it does not want label column (Iris variety)
        distance += (row1[i]-row2[i])**2
    return sqrt (distance)
```

### 4: It calculates the k nearest neighbours next
It calculates the k nearest neighbours next of the input sample by comparing and sorting its euclidean distance to each row of the available dataset.
Here num variable takes the value of k (k - nearest neighbours)
```py
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
```
### 5: It gives the predicted value of Iris variety as per the algorithm
It selects the k nearest neighbours to the sample row and among them, selects the maximum occuring parameter/variety and returns it.
```py 
def predict (dataset, myInputRow, num):
    neighbours = get_neighbours (dataset, myInputRow, num)
    output = [i[-1] for i  in neighbours]
    prediction = max(set (output), key = output.count)
    return prediction 
```

### 6: Kickstarting the knn algorithm process
```py
print(myInputRow)
predict = predict(dataset, myInputRow, 13)
```

### What's next ?
Next plan of action is testing this model for its accuracy, precision, over-fitting etc.
