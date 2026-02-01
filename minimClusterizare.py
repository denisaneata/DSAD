# ===== CLUSTER IERARHIC CRIZA =====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score

df = pd.read_csv("date.csv", index_col=0)  # <-- schimbi
X = df.values
X_std = (X - X.mean(axis=0)) / X.std(axis=0)

Z = linkage(X_std, method="ward")

plt.figure(figsize=(10,5))
dendrogram(Z, labels=df.index.tolist())
plt.title("Dendrograma")
plt.show()

k = 3  # <-- daca e dat / alegi repede
cl = fcluster(Z, k, criterion="maxclust")
pd.DataFrame({"Cluster": cl}, index=df.index).to_csv("Partitie.csv")

print("Silhouette:", silhouette_score(X_std, cl))
