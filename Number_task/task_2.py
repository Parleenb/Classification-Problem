import numpy as np
import time
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
# --------------------------------------------------
# Load data from file data_all.mat
# --------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'MNIST files', 'data_all.mat')
data = loadmat(data_path)
trainv = data['trainv']          # Training images (60000 x 784)
testv = data['testv']            # Test images (10000 x 784)
trainlab = data['trainlab'].flatten()  # Training labels
testlab = data['testlab'].flatten()    # Test labels

# --------------------------------------------------
# Normalize pixel values from [0,255] to [0,1]
# --------------------------------------------------
trainv = trainv / 255.0
testv = testv / 255.0

# --------------------------------------------------
# 3. NEAREST NEIGHBOR CLASSIFIER (k = 1)
# Using Euclidean distance
# --------------------------------------------------

chunk_size = 1000  # as suggested in the task
num_test = testv.shape[0]

predictions = []  # to store predicted labels

# Loop over test data in chunks
for i in range(0, num_test, chunk_size):
    print(f"Processing chunk {i} to {i + chunk_size}...")
    
    test_chunk = testv[i:i+chunk_size]  # take a chunk of test images
    
    # --------------------------------------------------
    # Compute Euclidean distance:
    # distance = ||test - train||^2
    # Using efficient vectorized formula:
    # (a - b)^2 = a^2 + b^2 - 2ab
    # --------------------------------------------------
    
    # squared norms
    test_sq = np.sum(test_chunk**2, axis=1, keepdims=True)   # shape: (chunk, 1)
    train_sq = np.sum(trainv**2, axis=1)                     # shape: (60000,)
    
    # distance matrix (chunk_size x 60000)
    distances = test_sq + train_sq - 2 * np.dot(test_chunk, trainv.T)
    
    # --------------------------------------------------
    # Find nearest neighbor (minimum distance)
    # --------------------------------------------------
    nearest_idx = np.argmin(distances, axis=1)
    
    # Get predicted labels
    pred_labels = trainlab[nearest_idx]
    
    predictions.extend(pred_labels)

# Convert to numpy array
predictions = np.array(predictions)

# --------------------------------------------------
# 4. CONFUSION MATRIX
# --------------------------------------------------
conf_mat = confusion_matrix(testlab, predictions)

print("\nConfusion Matrix:")
print(conf_mat)

# --------------------------------------------------
# 5. ERROR RATE
# --------------------------------------------------
error_rate = np.mean(predictions != testlab)

print("\nError rate:", error_rate)


"""
Task 1b
"""


# --------------------------------------------------
# 1. FIND MISCLASSIFIED INDICES
# --------------------------------------------------

# Indices where prediction != true label
misclassified_idx = np.where(predictions != testlab)[0]

print("Number of misclassified images:", len(misclassified_idx))

# --------------------------------------------------
# 2. PLOT SOME MISCLASSIFIED IMAGES
# --------------------------------------------------

num_to_plot = 10  # number of images to show

for i in range(num_to_plot):
    
    idx = misclassified_idx[i]  # index of misclassified image
    
    # --------------------------------------------------
    # Convert vector to 28x28 image
    # --------------------------------------------------
    x = testv[idx, :].reshape((28, 28))
    
    # --------------------------------------------------
    # Plot image
    # --------------------------------------------------
    plt.imshow(x, cmap='gray')
    plt.title(f"True: {testlab[idx]}, Predicted: {predictions[idx]}")
    plt.axis('off')  # remove axes for cleaner image
    plt.show()


    """
    Task 1c
    """

    # --------------------------------------------------
# 1. FIND CORRECTLY CLASSIFIED INDICES
# --------------------------------------------------

# Indices where prediction == true label
correct_idx = np.where(predictions == testlab)[0]

print("Number of correctly classified images:", len(correct_idx))

# --------------------------------------------------
# 2. PLOT SOME CORRECTLY CLASSIFIED IMAGES
# --------------------------------------------------

num_to_plot = 10  # number of correct images to show

for i in range(num_to_plot):
    
    idx = correct_idx[i]  # index of correctly classified image
    
    # Convert the image vector to a 28x28 image
    x = testv[idx, :].reshape((28, 28))
    
    # Plot the image
    plt.imshow(x, cmap='gray')
    plt.title(f"True: {testlab[idx]}, Predicted: {predictions[idx]}")
    plt.axis('off')  # hide axes
    plt.show()


    """
    Task 2a
    """


# --------------------------------------------------
# 1. PARAMETERS
# --------------------------------------------------
M = 64  # number of clusters per class
num_classes = 10

# Store cluster centers and their labels
all_centers = []
all_labels = []

# --------------------------------------------------
# 2. LOOP OVER EACH CLASS (0–9)
# --------------------------------------------------
for digit in range(num_classes):
    
    print(f"Clustering digit {digit}...")
    
    # --------------------------------------------------
    # Extract all training samples of this class
    # --------------------------------------------------
    train_vi = trainv[trainlab == digit]
    
    # train_vi should have ~6000 samples for each digit
    
    # --------------------------------------------------
    # Apply KMeans clustering
    # --------------------------------------------------
    kmeans = KMeans(n_clusters=M, random_state=42, n_init=10)
    #kmeans = KMeans(n_clusters=M, random_state=42)
    
    idx_i = kmeans.fit_predict(train_vi)   # cluster assignment (not used later)
    Ci = kmeans.cluster_centers_           # cluster centers (important!)
    
    # --------------------------------------------------
    # Store cluster centers and labels
    # --------------------------------------------------
    all_centers.append(Ci)
    all_labels.append(np.full(M, digit))  # label each center with the digit

# --------------------------------------------------
# 3. COMBINE ALL CLUSTERS INTO ONE DATASET
# --------------------------------------------------
cluster_centers = np.vstack(all_centers)   # shape: (10*M, 784)
cluster_labels = np.hstack(all_labels)     # shape: (10*M,)

print("Cluster centers shape:", cluster_centers.shape)
print("Cluster labels shape:", cluster_labels.shape)

# ==================================================
# TASK 2(b): NN using cluster templates (640 total)
# ==================================================

from sklearn.metrics import confusion_matrix
import numpy as np

chunk_size = 1000
num_test = testv.shape[0]

pred_cluster_nn = []

for i in range(0, num_test, chunk_size):
    
    print(f"Processing chunk {i} to {i+chunk_size}...")
    
    test_chunk = testv[i:i+chunk_size]
    
    # Compute Euclidean distance (same trick as before)
    test_sq = np.sum(test_chunk**2, axis=1, keepdims=True)
    train_sq = np.sum(cluster_centers**2, axis=1)
    
    distances = test_sq + train_sq - 2 * np.dot(test_chunk, cluster_centers.T)
    
    # Find nearest cluster center
    nearest_idx = np.argmin(distances, axis=1)
    
    pred_labels = cluster_labels[nearest_idx]
    
    pred_cluster_nn.extend(pred_labels)

pred_cluster_nn = np.array(pred_cluster_nn)

# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------
conf_mat_cluster = confusion_matrix(testlab, pred_cluster_nn)

print("\nConfusion Matrix (Cluster NN):")
print(conf_mat_cluster)

# --------------------------------------------------
# Error rate
# --------------------------------------------------
error_cluster = np.mean(pred_cluster_nn != testlab)

print("\nError rate (Cluster NN):", error_cluster)

# ==================================================
# TASK 2(c): KNN classifier (K = 7)
# ==================================================

from scipy.stats import mode

K = 7
pred_knn = []

for i in range(0, num_test, chunk_size):
    
    print(f"Processing chunk {i} to {i+chunk_size}...")
    
    test_chunk = testv[i:i+chunk_size]
    
    # Compute distances
    test_sq = np.sum(test_chunk**2, axis=1, keepdims=True)
    train_sq = np.sum(cluster_centers**2, axis=1)
    
    distances = test_sq + train_sq - 2 * np.dot(test_chunk, cluster_centers.T)
    
    # Find K nearest neighbors
    nearest_idx = np.argsort(distances, axis=1)[:, :K]
    
    nearest_labels = cluster_labels[nearest_idx]
    
    # Majority vote
    pred_labels = mode(nearest_labels, axis=1).mode.flatten()
    
    pred_knn.extend(pred_labels)

pred_knn = np.array(pred_knn)

# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------
conf_mat_knn = confusion_matrix(testlab, pred_knn)

print("\nConfusion Matrix (KNN, K=7):")
print(conf_mat_knn)

# --------------------------------------------------
# Error rate
# --------------------------------------------------
error_knn = np.mean(pred_knn != testlab)

print("\nError rate (KNN, K=7):", error_knn)


"""
2b)

Error rate (Cluster NN): 0.0477

Confusion Matrix (Cluster NN):
[[ 966    1    3    1    0    4    3    1    0    1]
 [   0 1126    3    1    0    0    3    0    1    1]
 [   9    8  979    8    1    0    3   11   12    1]
 [   0    0    7  940    1   32    0    6   17    7]
 [   1    6    2    0  922    0   10    4    3   34]
 [   3    1    0   20    2  845    9    1    7    4]
 [   8    3    2    0    3    3  935    0    3    1]
 [   0   20    8    0    4    1    0  961    2   32]
 [   5    1    3   11    3   20    2    6  917    6]
 [   5    6    4    5   30    1    1   19    6  932]]



2c)
Error rate (KNN, K=7): 0.063

Confusion Matrix (KNN, K=7):
[[ 952    1    3    0    0    8   13    1    2    0]
 [   0 1128    2    1    0    1    2    0    1    0]
 [  13   15  952   12    3    1    4   14   18    0]
 [   2    3    8  947    2   19    0   12   14    3]
 [   1   14    4    0  901    0    7    2    2   51]
 [   3    3    4   28    4  836    7    1    4    2]
 [   9    4    4    0    5   10  924    0    2    0]
 [   0   33   13    2   12    0    0  939    0   29]
 [   5    3    6   25   10   32    2    7  875    9]
 [   7    9    3    9   32    4    1   20    8  916]]

 
Task 1: Error rate: 0.0309
Task 2b: Error rate (Cluster NN): 0.0477
Task 2c: Error rate (KNN, K=7): 0.063



Using clustering significantly reduces computational complexity by replacing the 
full training set with a smaller number of representative templates. However, this 
leads to some loss in accuracy because cluster centers do not perfectly represent 
all variations of handwritten digits. Introducing a KNN classifier with K = 7 improves 
performance compared to using a single nearest neighbor, as it reduces sensitivity to
noise and outliers. Overall, KNN with clustered templates provides a good balance 
between accuracy and computational efficiency.

The KNN classifier (K = 7) performs worse than the nearest neighbor classifier in this case. 
This is because the classifier operates on cluster centers rather than the original training 
data. Cluster centers are averaged representations and may not preserve important local 
structures. As a result, using multiple neighbors introduces noise into the decision process, 
leading to a higher error rate. In contrast, using a single nearest cluster center (NN) gives
a more precise match. KNN generally performs better with large and dense datasets, but here
the reduced dataset (640 templates) limits its effectiveness.

"""