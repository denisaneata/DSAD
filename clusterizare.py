# =========================
# CLUSTERIZARE IERARHICA
# =========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score, silhouette_samples

# 1) citire date
df = pd.read_csv("date.csv", index_col=0)
X = df.values

# 2) standardizare (aproape mereu recomandat)
X_std = (X - X.mean(axis=0)) / X.std(axis=0)

# 3) ierarhie (matrice ierarhie)
Z = linkage(X_std, method="ward")      # ward ca la seminar

# 4) dendrograma
plt.figure(figsize=(10, 6))
dendrogram(Z, labels=df.index.tolist())
plt.title("Dendrograma")
plt.show()

# 5) partiție k (daca k e dat / citit)
k = 3  # <-- schimbi / citesti
clusteri = fcluster(Z, k, criterion="maxclust")
pd.DataFrame({"Cluster": clusteri}, index=df.index).to_csv("Partitie_k.csv")

# 6) Silhouette (partitie)
sil = silhouette_score(X_std, clusteri)
sil_inst = silhouette_samples(X_std, clusteri)

print("Silhouette partitie:", sil)
pd.DataFrame({"Silhouette": sil_inst}, index=df.index).to_csv("Silhouette_inst.csv")

# 7) Plot silhouette simplu (optional)
plt.figure(figsize=(10, 5))
plt.hist(sil_inst, bins=20, edgecolor="black")
plt.title("Distributie Silhouette (instante)")
plt.show()
