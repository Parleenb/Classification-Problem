import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Your confusion matrix
cm = np.array([
[955, 1, 3, 1, 0, 7, 10, 1, 2, 0],
[0, 1127, 2, 1, 0, 1, 3, 0, 1, 0],
[15, 11, 955, 14, 2, 0, 3, 14, 18, 0],
[1, 4, 8, 942, 1, 26, 0, 9, 14, 5],
[1, 13, 3, 0, 903, 0, 7, 3, 4, 48],
[3, 3, 2, 35, 4, 826, 8, 1, 8, 2],
[11, 4, 4, 0, 5, 7, 925, 0, 2, 0],
[0, 34, 14, 2, 7, 0, 0, 939, 1, 31],
[8, 2, 6, 24, 5, 43, 4, 6, 871, 5],
[6, 9, 5, 12, 32, 6, 2, 28, 3, 906]
])

# Labels (adjust if you have actual class names like digits)
labels = [str(i) for i in range(10)]

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)

#sns.heatmap(cm, annot=True, fmt="d", cmap="plasma")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")

plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()