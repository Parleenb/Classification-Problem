import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

"""
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


#1)b)
# add bias term
X_train_b = add_bias(X_train)
X_test_b = add_bias(X_test)

# one-hot targets for MSE training
T_train = one_hot(y_train, num_classes=3)

# ---------- initialize weights ----------
np.random.seed(0)
W = 0.01 * np.random.randn(3, X_train_b.shape[1])  

# ---------- training settings ----------
alpha = 0.01      
max_iters = 20000
tol = 1e-7

mse_history = []

# ---------- training ----------
for it in range(max_iters):
    Z = X_train_b @ W.T                  
    G = sigmoid(Z)                       

    mse = 0.5 * np.sum((G - T_train) ** 2)
    mse_history.append(mse)

    # Eq. (22): gradient of MSE
    grad = ((G - T_train) * G * (1 - G)).T @ X_train_b   

    # Eq. (23): weight update
    W = W - alpha * grad

    if it > 0 and abs(mse_history[-1] - mse_history[-2]) < tol:
        print(f"Converged after {it} iterations")
        break

# ---------- final weights ----------

print("Weights after training:")
print(np.round(W, 3))

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

# ------------------------------------------
# 2(a) and 2(b)
# Use first 30 per class for training, last 20 per class for testing

feature_names = iris.feature_names
class_names = iris.target_names

# ---------- histograms ----------
for f in range(X.shape[1]):
    plt.figure(figsize=(7, 4))
    
    for c in range(3):
        sns.histplot(
            X_train[y_train == c, f],
            bins=8,
            kde=True,
            label=class_names[c],
            alpha=0.5,
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

    print("Weights:")
    print(np.round(W, 3))

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

print("\nFeature overlap scores:")
for i in range(len(feature_names)):
    print(feature_names[i], "->", scores[i])
    
# feature indices: [sepal length, sepal width, petal length, petal width]
remove_order = [1, 0, 2, 3] 

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