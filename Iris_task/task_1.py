import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix

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
"""
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