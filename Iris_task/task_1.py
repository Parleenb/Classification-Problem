import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix

#------ helper functions ------
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def add_bias(X):
    return np.hstac((X, np.ones((X.shape[0], 1))))

def one_hot(y, num_classes=3):
    T = np.zeros((len(y), num_classes))
    T[np.arange(len(y)), y] = 1
    return T

def predict(Xb, W):
    G = sigmoid(Xb @ W.T)
    
    

iris = load_iris()

# class_1 = Iris setosa
# class_2 = Iris versicolor
# class_3 = Iris virginica

print(iris.keys())
#print(iris.data)
#print(iris.target_names)
#print(iris.target)

X = iris.data
y = iris.target

X_train = np.vstack((X[0:30], X[50:80], X[100:130]))
X_test  = np.vstack((X[30:50], X[80:100], X[130:150]))

y_train = np.hstack((y[0:30], y[50:80], y[100:130]))
y_test  = np.hstack((y[30:50], y[80:100], y[130:150]))

# Når vi printer y = iris.target, er klassenavnene representert som tall (0,1,2). I MSE vil f.eks. class 3 være større enn class 2, og MSE vil ikke være riktig, fordi modellen vil alltid tenke at en classe er større enn den andre, basert på tallverdien. Derfor gjør vi om klassenavnene (0,1,2) til one-hot encoding, slik at alle klasser er likeverdige og ikke har noen rangering.
#one-hot encoding: https://medium.com/data-science/building-a-one-hot-encoding-layer-with-tensorflow-f907d686bf39 
T_train = np.eye(3)[y_train]
T_test  = np.eye(3)[y_test]

# Add bias term - test med og uten for å se forskjell i resultatet
# y = w*x + b, vi legger til bias direkte til vektoren w, slik at y = w'*x, der w' = [b, w1, w2, w3], og x = [1, x1, x2, x3]. På denne måten kan vi bruke samme formel for både vekter og bias, og det gjør det enklere å implementere modellen.
# Without bias: Line must go through (0,0). With bias: Line can go through any point.
X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))

)
