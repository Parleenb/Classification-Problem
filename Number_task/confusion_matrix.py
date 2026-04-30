import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Confusion matrix
cm = np.array([[20, 0, 0],
                 [0, 18, 2],
                 [0, 0, 20]])

# Class labels
labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

plt.figure(figsize=(12, 10))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")

plt.xticks(rotation=30, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()