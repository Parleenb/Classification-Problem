import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

cm = np.array([
    [20, 0, 0],
    [0, 14, 6],
    [0, 1, 19]
])


# Correct labels (3 classes)
labels = ["Setosa", "Versicolor", "Virginica"]

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.tight_layout()
plt.show()