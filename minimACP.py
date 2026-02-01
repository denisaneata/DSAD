
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv("date.csv", index_col=0)   # <-- schimbi
X = df.values
variabile = df.columns.tolist()

# standardizare (obligatoriu)
X_std = (X - X.mean(axis=0)) / X.std(axis=0)

pca = PCA()
pca.fit(X_std)

# 1) varianta componentelor
alpha = pca.explained_variance_
ponderi = alpha / alpha.sum()
cum = np.cumsum(ponderi)
t_var = pd.DataFrame({"alpha": alpha, "procent": ponderi, "cumulat": cum},
                     index=[f"C{i+1}" for i in range(len(alpha))])
t_var.to_csv("Varianta.csv")
print(t_var)

# 2) scoruri (componente)
C = pca.transform(X_std)
t_C = pd.DataFrame(C, index=df.index, columns=[f"C{i+1}" for i in range(C.shape[1])])
t_C.to_csv("Componente.csv")

# 3) corelatii factoriale r = corr(X, C)
corr = np.corrcoef(X_std, C, rowvar=False)
m = X.shape[1]
r_x_c = corr[:m, m:]
t_r = pd.DataFrame(r_x_c, index=variabile, columns=[f"C{i+1}" for i in range(C.shape[1])])
t_r.to_csv("r.csv")
print(t_r)

# 4) cercul corelatiilor (C1 vs C2)
if C.shape[1] >= 2:
    theta = np.linspace(0, 2*np.pi, 200)
    plt.figure(figsize=(7,7))
    plt.plot(np.cos(theta), np.sin(theta))
    plt.axhline(0); plt.axvline(0)
    plt.scatter(t_r["C1"], t_r["C2"])
    for i in range(len(t_r)):
        plt.text(t_r["C1"].iloc[i], t_r["C2"].iloc[i], t_r.index[i])
    plt.gca().set_aspect("equal")
    plt.title("Cercul corelatiilor C1 vs C2")
    plt.show()
