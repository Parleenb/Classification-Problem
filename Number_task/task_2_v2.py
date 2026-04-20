import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'MNIST files', 'data_all.mat')
data = loadmat(data_path)

trainv = data['trainv']
testv = data['testv']
trainlab = data['trainlab'].flatten().astype(int)
testlab = data['testlab'].flatten().astype(int)

# Normalize pixel values to [0,1]
trainv = trainv / 255.0
testv = testv / 255.0


# --------------------------------------------------
# FUNCTIONS
# --------------------------------------------------
def compute_1nn_predictions(train_data, train_labels, test_data, chunk_size=1000):
    predictions = []
    num_test = test_data.shape[0]

    train_sq = np.sum(train_data**2, axis=1)

    for i in range(0, num_test, chunk_size):
        print(f"Processing chunk {i} to {min(i + chunk_size, num_test)}...")
        test_chunk = test_data[i:i + chunk_size]

        test_sq = np.sum(test_chunk**2, axis=1, keepdims=True)
        distances = test_sq + train_sq - 2 * np.dot(test_chunk, train_data.T)
        distances = np.maximum(distances, 0)

        nearest_idx = np.argmin(distances, axis=1)
        pred_labels = train_labels[nearest_idx]
        predictions.extend(pred_labels)

    return np.array(predictions)


def compute_knn_predictions(train_data, train_labels, test_data, K=7, chunk_size=1000):
    predictions = []
    num_test = test_data.shape[0]

    train_sq = np.sum(train_data**2, axis=1)

    for i in range(0, num_test, chunk_size):
        print(f"Processing chunk {i} to {min(i + chunk_size, num_test)}...")
        test_chunk = test_data[i:i + chunk_size]

        test_sq = np.sum(test_chunk**2, axis=1, keepdims=True)
        distances = test_sq + train_sq - 2 * np.dot(test_chunk, train_data.T)
        distances = np.maximum(distances, 0)

        nearest_idx = np.argpartition(distances, K, axis=1)[:, :K]
        nearest_labels = train_labels[nearest_idx]

        pred_labels = np.array([
            np.bincount(row.astype(int)).argmax()
            for row in nearest_labels
        ])
        predictions.extend(pred_labels)

    return np.array(predictions)


def build_cluster_templates(train_data, train_labels, M=64, num_classes=10):
    all_centers = []
    all_labels = []

    for digit in range(num_classes):
        print(f"Clustering digit {digit}...")
        class_data = train_data[train_labels == digit]

        kmeans = KMeans(n_clusters=M, random_state=42, n_init=10)
        kmeans.fit(class_data)

        centers = kmeans.cluster_centers_
        labels = np.full(M, digit, dtype=int)

        all_centers.append(centers)
        all_labels.append(labels)

    cluster_centers = np.vstack(all_centers)
    cluster_labels = np.hstack(all_labels)

    return cluster_centers, cluster_labels


def evaluate_classifier(true_labels, pred_labels, name="Classifier"):
    conf_mat = confusion_matrix(true_labels, pred_labels)
    error_rate = np.mean(pred_labels != true_labels)

    print(f"\n{name} confusion matrix:")
    print(conf_mat)
    print(f"\n{name} error rate: {error_rate:.4f}")

    return conf_mat, error_rate


def plot_examples(images, true_labels, pred_labels, indices, title_prefix, num_to_plot=10):
    plt.figure(figsize=(12, 5))
    for j, idx in enumerate(indices[:num_to_plot]):
        plt.subplot(2, 5, j + 1)
        x = images[idx].reshape(28, 28)
        plt.imshow(x, cmap='gray')
        plt.title(f"T:{true_labels[idx]} P:{pred_labels[idx]}")
        plt.axis('off')

    plt.suptitle(title_prefix)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# TASK 1(a): FULL 1-NN
# --------------------------------------------------
start = time.time()
pred_full_nn = compute_1nn_predictions(trainv, trainlab, testv, chunk_size=1000)
time_full_nn = time.time() - start

conf_full, err_full = evaluate_classifier(testlab, pred_full_nn, "Full 1-NN")
print(f"Full 1-NN processing time: {time_full_nn:.2f} seconds")


# --------------------------------------------------
# TASK 1(b): MISCLASSIFIED IMAGES
# --------------------------------------------------
misclassified_idx = np.where(pred_full_nn != testlab)[0]
print("Number of misclassified images:", len(misclassified_idx))
plot_examples(testv, testlab, pred_full_nn, misclassified_idx, "Misclassified images")


# --------------------------------------------------
# TASK 1(c): CORRECTLY CLASSIFIED IMAGES
# --------------------------------------------------
correct_idx = np.where(pred_full_nn == testlab)[0]
print("Number of correctly classified images:", len(correct_idx))
plot_examples(testv, testlab, pred_full_nn, correct_idx, "Correctly classified images")


# --------------------------------------------------
# TASK 2(a): CLUSTERING
# --------------------------------------------------
cluster_centers, cluster_labels = build_cluster_templates(trainv, trainlab, M=64, num_classes=10)
print("Cluster centers shape:", cluster_centers.shape)
print("Cluster labels shape:", cluster_labels.shape)


# --------------------------------------------------
# TASK 2(b): 1-NN WITH CLUSTER TEMPLATES
# --------------------------------------------------
start = time.time()
pred_cluster_nn = compute_1nn_predictions(cluster_centers, cluster_labels, testv, chunk_size=1000)
time_cluster_nn = time.time() - start

conf_cluster, err_cluster = evaluate_classifier(testlab, pred_cluster_nn, "Cluster 1-NN")
print(f"Cluster 1-NN processing time: {time_cluster_nn:.2f} seconds")


# --------------------------------------------------
# TASK 2(c): KNN WITH K=7
# --------------------------------------------------
start = time.time()
pred_knn = compute_knn_predictions(cluster_centers, cluster_labels, testv, K=7, chunk_size=1000)
time_knn = time.time() - start

conf_knn, err_knn = evaluate_classifier(testlab, pred_knn, "Cluster KNN (K=7)")
print(f"Cluster KNN (K=7) processing time: {time_knn:.2f} seconds")

