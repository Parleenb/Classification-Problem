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
    return np.argmax(G, axis=1)

def error_rate(y_true, y_pred):
    return np.mean(y_true != y_pred)


# ---------- load data ----------
iris = load_iris()
X = iris.data
y = iris.target

# first 30 of each class for training, last 20 of each class for testing
#1)a)
X_train = np.vstack((X[0:30], X[50:80], X[100:130]))
X_test  = np.vstack((X[30:50], X[80:100], X[130:150]))

y_train = np.hstack((y[0:30], y[50:80], y[100:130]))
y_test  = np.hstack((y[30:50], y[80:100], y[130:150]))

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
y_pred_train = predict(X_train_b, W)
y_pred_test = predict(X_test_b, W)

print("\nTraining confusion matrix:")
print(confusion_matrix(y_train, y_pred_train))
print("Training error rate:", error_rate(y_train, y_pred_train))

print("\nTest confusion matrix:")
print(confusion_matrix(y_test, y_pred_test))
print("Test error rate:", error_rate(y_test, y_pred_test))

#--------------------------------

