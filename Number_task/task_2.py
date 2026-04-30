import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix


# Load data
script_direction = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_direction, "MNIST files", "data_all.mat")

data = loadmat(data_path)

trainv = data["trainv"]
testv = data["testv"]

trainlab = data["trainlab"].flatten()
testlab = data["testlab"].flatten()

trainlab = trainlab.astype(int)
testlab = testlab.astype(int)

# Normalize from 0-255 to 0-1
trainv = trainv / 255.0
testv = testv / 255.0


# NN (Task 1)
def compute_nn_predictions(train_data, train_labels, test_data, chunk_size=1000):
    predictions = []
    num_test = test_data.shape[0]

    # Precompute squared lengths for training data
    train_sq = np.sum(train_data**2, axis=1)

    for i in range(0, num_test, chunk_size):
        end = min(i + chunk_size, num_test)
        print(f"Processing chunk {i} to {end}")

        test_chunk = test_data[i:end]

        # Compute squared lengths for test chunk
        test_sq = np.sum(test_chunk**2, axis=1).reshape(-1, 1)

        #  Squared Euclidean distance, Formula (9) from report
        distances = test_sq + train_sq - 2 * np.dot(test_chunk, train_data.T)

        # Avoid negative values
        distances[distances < 0] = 0

        # Find NN
        nearest_idx = np.argmin(distances, axis=1)

        for idx in nearest_idx:
            predictions.append(train_labels[idx])

    return np.array(predictions)


# KNN with K=7, change K as needed (Task 2c)
def compute_knn_predictions(train_data, train_labels, test_data, K=7, chunk_size=1000):
    predictions = []
    num_test = test_data.shape[0]

    # Precompute squared lengths for training data
    train_sq = np.sum(train_data**2, axis=1)

    for i in range(0, num_test, chunk_size):
        end = min(i + chunk_size, num_test)
        print(f"Processing chunk {i} to {end}...")

        test_chunk = test_data[i:end]

        # Squared lengths for test data
        test_sq = np.sum(test_chunk**2, axis=1).reshape(-1, 1)

        # Sqyared euclidean distance
        distances = test_sq + train_sq - 2 * np.dot(test_chunk, train_data.T)
        distances[distances < 0] = 0

        # Find closest K neighbors
        nearest_idx = np.argpartition(distances, K, axis=1)[:, :K]

        # Labels of closest K neighbors
        nearest_labels = train_labels[nearest_idx]

        # Majority voting -> predict the most common label among neighbors
        for row in nearest_labels:
            counts = np.bincount(row.astype(int))
            pred_label = np.argmax(counts)
            predictions.append(pred_label)

    return np.array(predictions)


# Clustering templates (Task 2a)
def build_cluster_templates(train_data, train_labels, M=64, num_classes=10):
    centers_list = []
    labels_list = []

    for i in range(num_classes):
        print("Clustering number", i)

        # Find samples for the class
        idx = np.where(train_labels == i)[0]
        class_data = train_data[idx]

        # KMeans
        kmeans = KMeans(n_clusters=M, random_state=42, n_init=10)
        kmeans.fit(class_data)

        # Store centers/centroids
        centers_list.append(kmeans.cluster_centers_)

        # Store corresponding labels for the centers
        class_labels = []
        for _ in range(M):
            class_labels.append(i)

        labels_list.append(np.array(class_labels))

    # Combine results
    cluster_centers = np.concatenate(centers_list, axis=0)
    cluster_labels = np.concatenate(labels_list, axis=0)

    return cluster_centers, cluster_labels


# Evaluation of classifiers: confusion matrix and error rate (Task 2b and 2c)
def evaluate_classifier(true_labels, pred_labels, name="Classifier"):
    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_labels, pred_labels)

    # Compute error rate 
    num_wrong = np.sum(pred_labels != true_labels)
    error_rate = num_wrong / len(true_labels)

    print("\n" + name + " confusion matrix:")
    print(conf_matrix)

    print("\n" + name + " error rate:", round(error_rate, 4))

    return conf_matrix, error_rate


# Plotting examples (Task 1b and 1c)
def plot_examples(images, true_labels, pred_labels, indices, title_prefix, num_to_plot=10):
    plt.figure(figsize=(12, 5))

    count = min(num_to_plot, len(indices))

    for j in range(count):
        idx = indices[j]
        plt.subplot(2, 5, j + 1)

        # Reshape image
        img = images[idx].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        title_text = "T:" + str(true_labels[idx]) + " P:" + str(pred_labels[idx])
        plt.title(title_text)
        plt.axis('off')

    plt.suptitle(title_prefix)
    plt.tight_layout()
    plt.show()



#Task 1a: NN without clustering
start = time.time() # timer start
prediction_NN = compute_nn_predictions(trainv, trainlab, testv, chunk_size=1000)
time_NN = time.time() - start

conf_NN, err_NN = evaluate_classifier(testlab, prediction_NN, "NN")
print(f"NN processing time: {time_NN:.2f} seconds")

# Task 1b: misclassified images
misclassified_images = np.where(prediction_NN != testlab)[0]
print("Number of misclassified images:", len(misclassified_images))
plot_examples(testv, testlab, prediction_NN, misclassified_images, "Misclassified images")


# Task 1c: correctly classified images
correct_images = np.where(prediction_NN == testlab)[0]
print("Number of correctly classified images:", len(correct_images))
plot_examples(testv, testlab, prediction_NN, correct_images, "Correctly classified images")


# Task 2a: clustering the data
cluster_centers, cluster_labels = build_cluster_templates(trainv, trainlab, M=64, num_classes=10)

# Use for debugging, supposed to be 640 clusters and 784 pxls
#print("Cluster centers shape:", cluster_centers.shape)
#print("Cluster labels shape:", cluster_labels.shape)


# Task 2b: NN with clustering
start = time.time()
prediction_cluster_NN = compute_nn_predictions(cluster_centers, cluster_labels, testv, chunk_size=1000)
time_cluster_NN = time.time() - start

conf_cluster, err_cluster = evaluate_classifier(testlab, prediction_cluster_NN, "Cluster NN")
print(f"Cluster NN processing time: {time_cluster_NN:.2f} seconds")


# Task 2c: KNN with clustering, K = 7
start = time.time()
prediction_KNN = compute_knn_predictions(cluster_centers, cluster_labels, testv, K=7, chunk_size=1000)
time_KNN = time.time() - start

conf_KNN, err_KNN = evaluate_classifier(testlab, prediction_KNN, "Cluster KNN (K=7)")
print(f"Cluster KNN (K=7) processing time: {time_KNN:.2f} seconds")

