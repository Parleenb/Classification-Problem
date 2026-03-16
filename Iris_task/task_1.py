import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix

iris = load_iris()
print(iris.keys())
#print(iris.data)
print(iris.target_names)


x = iris.data
y = iris.target