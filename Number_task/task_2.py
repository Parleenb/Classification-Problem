import numpy as np
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# --------------------------------------------------
# 1. LOAD DATA FROM data_all.mat
# --------------------------------------------------
data = loadmat(r'MNIST files\data_all.mat')
trainv = data['trainv']          # Training images (60000 x 784)
testv = data['testv']            # Test images (10000 x 784)
trainlab = data['trainlab'].flatten()  # Training labels
testlab = data['testlab'].flatten()    # Test labels

# --------------------------------------------------
# 2. NORMALIZE DATA (scale pixel values to 0-1)
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