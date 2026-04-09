import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#------ helper functions ------
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def add_bias(X):
    return np.hstack((X, np.ones((X.shape[0], 1))))

def one_hot(y, num_classes=3):
    T = np.zeros((len(y), num_classes))
    T[np.arange(len(y)), y] = 1
    return T

def predict(Xb, W):
    G = sigmoid(Xb @ W.T)
    return np.argmax(G, axis=1)

def error_rate(y_true, y_pred):
    return np.mean(y_true != y_pred)


# ---------- load data ----------
iris = load_iris()
X = iris.data
y = iris.target

# first 30 of each class for training, last 20 of each class for testing
#1)a)
# case 1
X_train = np.vstack((X[0:30], X[50:80], X[100:130]))
X_test  = np.vstack((X[30:50], X[80:100], X[130:150]))

y_train = np.hstack((y[0:30], y[50:80], y[100:130]))
y_test  = np.hstack((y[30:50], y[80:100], y[130:150]))

"""
# case 2 (part d)
X_train = np.vstack((X[20:50], X[70:100], X[120:150]))
X_test  = np.vstack((X[0:20], X[50:70], X[100:120]))

y_train = np.hstack((y[20:50], y[70:100], y[120:150]))
y_test  = np.hstack((y[0:20], y[50:70], y[100:120]))
"""

#1)b)
# add bias term
X_train_b = add_bias(X_train)
X_test_b = add_bias(X_test)

# one-hot targets for MSE training
T_train = one_hot(y_train, num_classes=3)

# ---------- initialize weights ----------
np.random.seed(0)
W = 0.01 * np.random.randn(3, X_train_b.shape[1])   # shape (3, 5)

# ---------- training settings ----------
alpha = 0.001       # 0.01, 0.001, 0.0001
max_iters = 20000
tol = 1e-7

mse_history = []

# ---------- batch gradient descent ----------
for it in range(max_iters):
    Z = X_train_b @ W.T                  # shape (N, 3)
    G = sigmoid(Z)                       # shape (N, 3)

    mse = 0.5 * np.sum((G - T_train) ** 2)
    mse_history.append(mse)

    # Eq. (22): gradient of MSE
    grad = ((G - T_train) * G * (1 - G)).T @ X_train_b   # shape (3, 5)

    # Eq. (23): weight update
    W = W - alpha * grad

    if it > 0 and abs(mse_history[-1] - mse_history[-2]) < tol:
        print(f"Converged after {it} iterations")
        break

# ---------- evaluate ----------
#1)c)
y_pred_train = predict(X_train_b, W)
y_pred_test = predict(X_test_b, W)

print("\nTraining confusion matrix:")
print(confusion_matrix(y_train, y_pred_train))
print("Training error rate:", error_rate(y_train, y_pred_train))

print("\nTest confusion matrix:")
print(confusion_matrix(y_test, y_pred_test))
print("Test error rate:", error_rate(y_test, y_pred_test))

#--------------------------------

#1)d)
#Trained and tested already.

"""
1)a) with alpha = 0.01
Training confusion matrix:
[[30  0  0]
 [ 0 29  1]
 [ 0  0 30]]
Training error rate: 0.011111111111111112

Test confusion matrix:
[[20  0  0]
 [ 0 18  2]
 [ 0  0 20]]
Test error rate: 0.03333333333333333

1)a) with alpha = 0.001
Training confusion matrix:
[[30  0  0]
 [ 0 29  1]
 [ 0  0 30]]
Training error rate: 0.03333333333333333

Test confusion matrix:
[[20  0  0]
 [ 0 18  2]
 [ 0  0 20]]
Test error rate: 0.03333333333333333
"""

"""
1)d) with alpha = 0.01
Training confusion matrix:
[[30  0  0]
 [ 0 28  2]
 [ 0  3 27]]
Training error rate: 0.05555555555555555

Test confusion matrix:
[[20  0  0]
 [ 0 20  0]
 [ 0  0 20]]
Test error rate: 0.0

1)d) with alpha = 0.001
Training confusion matrix:
[[30  0  0]
 [ 0 28  2]
 [ 0  3 27]]
Training error rate: 0.05555555555555555

Test confusion matrix:
[[20  0  0]
 [ 0 20  0]
 [ 0  0 20]]
Test error rate: 0.0
"""

#1)e)
#The results for the two splits are similar overall, but some differences can be observed. 
# In (a), the training error is very low and the test error is small but non-zero. 
# In case (d), the training error is slightly higher, while the test error is zero. 
# This does not necessarily mean the classifier is better in case (d), but rather that the chosen test samples are easier to classify. 
# The variation in error rates is due to the different data splits, and this goes to show that performance depends on which samples are used for training and testing. 
# Overall, the classifier performs well in both cases, which confirms that the Iris dataset is close to linearly separable.


# ------------------------------------------
# 2(a) and 2(b)
# Use first 30 per class for training, last 20 per class for testing

feature_names = iris.feature_names
class_names = iris.target_names

# ---------- histograms ----------
for f in range(X.shape[1]):
    plt.figure(figsize=(7, 4))
    for c in range(3):
        plt.hist(
            X_train[y_train == c, f],
            bins=8,
            alpha=0.5,
            label=class_names[c]
        )
    plt.title(f"Histogram of {feature_names[f]}")
    plt.xlabel(feature_names[f])
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------- helper functions for part 2 ----------
def train_model(X_train, y_train, alpha=0.001, max_iters=20000, tol=1e-7):
    X_train_b = add_bias(X_train)
    T_train = one_hot(y_train, num_classes=3)

    np.random.seed(0)
    W = 0.01 * np.random.randn(3, X_train_b.shape[1])

    mse_history = []

    for it in range(max_iters):
        Z = X_train_b @ W.T
        G = sigmoid(Z)

        mse = 0.5 * np.sum((G - T_train) ** 2)
        mse_history.append(mse)

        grad = ((G - T_train) * G * (1 - G)).T @ X_train_b
        W = W - alpha * grad

        if it > 0 and abs(mse_history[-1] - mse_history[-2]) < tol:
            break

    return W

def test_model(X_train, y_train, X_test, y_test, alpha=0.001):
    W = train_model(X_train, y_train, alpha=alpha)

    X_train_b = add_bias(X_train)
    X_test_b = add_bias(X_test)

    y_pred_train = predict(X_train_b, W)
    y_pred_test = predict(X_test_b, W)

    print("Training confusion matrix:")
    print(confusion_matrix(y_train, y_pred_train))
    print("Training error rate:", error_rate(y_train, y_pred_train))

    print("\nTest confusion matrix:")
    print(confusion_matrix(y_test, y_pred_test))
    print("Test error rate:", error_rate(y_test, y_pred_test))

def overlap_score(feature_column, y_train, bins=8):
    histograms = []
    for c in range(3):
        hist, _ = np.histogram(feature_column[y_train == c], bins=bins)
        histograms.append(hist)

    overlap = 0
    for i in range(3):
        for j in range(i + 1, 3):
            overlap += np.sum(np.minimum(histograms[i], histograms[j]))
    return overlap

# ---------- rank features by overlap ----------
scores = []
for f in range(X_train.shape[1]):
    score = overlap_score(X_train[:, f], y_train)
    scores.append(score)

scores = np.array(scores)

print("\nFeature overlap scores (higher means more overlap / worse feature):")
for i in range(len(feature_names)):
    print(feature_names[i], "->", scores[i])

# worst feature first
remove_order = np.argsort(-scores)

print("\nFeatures removed in this order:")
for idx in remove_order:
    print(feature_names[idx])

# ---------- 4 features ----------
print("\n====================")
print("Using 4 features")
print("====================")
test_model(X_train, y_train, X_test, y_test, alpha=0.001)

# ---------- 3 features ----------
keep3 = [i for i in range(4) if i != remove_order[0]]

print("\n====================")
print("Using 3 features")
print("Kept features:")
for idx in keep3:
    print(feature_names[idx])
print("====================")

test_model(X_train[:, keep3], y_train, X_test[:, keep3], y_test, alpha=0.001)

# ---------- 2 features ----------
keep2 = [i for i in keep3 if i != remove_order[1]]

print("\n====================")
print("Using 2 features")
print("Kept features:")
for idx in keep2:
    print(feature_names[idx])
print("====================")

test_model(X_train[:, keep2], y_train, X_test[:, keep2], y_test, alpha=0.001)

# ---------- 1 feature ----------
keep1 = [keep2[0]]

print("\n====================")
print("Using 1 feature")
print("Kept feature:")
print(feature_names[keep1[0]])
print("====================")

test_model(X_train[:, keep1], y_train, X_test[:, keep1], y_test, alpha=0.001)

#---------------------------------

#1)c)
"""
Training confusion matrix:
[[30  0  0]
 [ 0 28  2]
 [ 0  1 29]]
Training error rate: 0.03333333333333333

Test confusion matrix:
[[20  0  0]
 [ 0 18  2]
 [ 0  0 20]]
Test error rate: 0.03333333333333333

Feature overlap scores (higher means more overlap / worse feature):
sepal length (cm) -> 77
sepal width (cm) -> 59
petal length (cm) -> 58
petal width (cm) -> 44

Features removed in this order:
sepal length (cm)
sepal width (cm)
petal length (cm)
petal width (cm)

====================
Using 4 features
====================
Training confusion matrix:
[[30  0  0]
 [ 0 28  2]
 [ 0  1 29]]
Training error rate: 0.03333333333333333

Test confusion matrix:
[[20  0  0]
 [ 0 18  2]
 [ 0  0 20]]
Test error rate: 0.03333333333333333

====================
Using 3 features
Kept features:
sepal width (cm)
petal length (cm)
petal width (cm)
====================
Training confusion matrix:
[[30  0  0]
 [ 0 26  4]
 [ 0  0 30]]
Training error rate: 0.044444444444444446

Test confusion matrix:
[[20  0  0]
 [ 0 19  1]
 [ 0  2 18]]
Test error rate: 0.05

====================
Using 2 features
Kept features:
petal length (cm)
petal width (cm)
====================
Training confusion matrix:
[[30  0  0]
 [ 0 26  4]
 [ 0  2 28]]
Training error rate: 0.06666666666666667

Test confusion matrix:
[[20  0  0]
 [ 0 19  1]
 [ 0  2 18]]
Test error rate: 0.05

====================
Using 1 feature
Kept feature:
petal length (cm)
====================
Training confusion matrix:
[[30  0  0]
 [ 0 21  9]
 [ 0  1 29]]
Training error rate: 0.1111111111111111

Test confusion matrix:
[[20  0  0]
 [ 0 18  2]
 [ 0  0 20]]
Test error rate: 0.03333333333333333
"""

#------------------------------------

#2)c)
#Using all four features gives the lowest error, but performance remains similar when using only petal length and petal width, showing that these features contain most of the discriminative information. 
#The Setosa class is always perfectly classified, while most errors occur between Versicolor and Virginica, indicating some overlap and limited linear separability between these two classes.

#------------------------------------