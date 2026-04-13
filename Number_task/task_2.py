import numpy as np
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix

# --------------------------------------------------
# 1. LOAD DATA FROM data_all.mat
# --------------------------------------------------
data = loadmat('data_all.mat')

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