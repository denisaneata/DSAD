#importurile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.decomposition import PCA

# citim datele
df = pd.read_csv("date.csv", index_col=0)        
variabile = df.columns.tolist()                  # etichete variabile
X = df.values                                    # matricea datelor

#standardizare (OBLIGATORIU la PCA)
X_std = (X - X.mean(axis=0)) / X.std(axis=0)

#model PCA
pca = PCA()
pca.fit(X_std)

#valori proprii (varianta componentelor)
alpha = pca.explained_variance_
ponderi = alpha / alpha.sum()
ponderi_cum = np.cumsum(ponderi)

t_var = pd.DataFrame({
    "alpha": alpha,
    "Procent": ponderi,
    "Cumulat": ponderi_cum
}, index=[f"C{i+1}" for i in range(len(alpha))])

print(t_var)
t_var.to_csv("VariantaComponente.csv")

#plot varianta + criteriul Kaiser (alpha > 1)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(alpha)+1), alpha, marker="o")
plt.axhline(1, color="red", linestyle="--")  # Kaiser
plt.title("Scree plot (Kaiser: alpha>1)")
plt.xlabel("Componenta")
plt.ylabel("Valoare proprie (alpha)")
plt.show()

#scoruri / componente (coordonatele instantelor)
C = pca.transform(X_std)
t_C = pd.DataFrame(C, index=df.index, columns=[f"C{i+1}" for i in range(C.shape[1])])
t_C.to_csv("Componente.csv")
print(t_C.head())

#corelatii factoriale (variabile observate - componente)
corr = np.corrcoef(X_std, C, rowvar=False)
m = X.shape[1]
r_x_c = corr[:m, m:]

t_r = pd.DataFrame(r_x_c, index=variabile, columns=[f"C{i+1}" for i in range(C.shape[1])])
t_r.to_csv("r.csv")
print(t_r)

#corelograma corelatii factoriale
plt.figure(figsize=(9, 7))
sb.heatmap(t_r, vmin=-1, vmax=1, cmap="RdYlBu", annot=True)
plt.title("Corelograma corelatii variabile - componente")
plt.show()

#cercul corelatiilor (C1 vs C2) cu protectie
if C.shape[1] >= 2:
    plt.figure(figsize=(8, 8))
    theta = np.linspace(0, 2*np.pi, 200)
    plt.plot(np.cos(theta), np.sin(theta))
    plt.axhline(0); plt.axvline(0)

    plt.scatter(t_r["C1"], t_r["C2"])
    for i in range(len(t_r)):
        plt.text(t_r["C1"].iloc[i], t_r["C2"].iloc[i], t_r.index[i])

    plt.gca().set_aspect("equal")
    plt.title("Cercul corelatiilor (C1 vs C2)")
    plt.xlabel("C1"); plt.ylabel("C2")
    plt.show()
else:
    print("Nu pot desena cercul corelatiilor: <2 componente")

#comunalitati (cumul de r^2 pe componente)
comunalitati = np.cumsum(r_x_c**2, axis=1)
t_com = pd.DataFrame(comunalitati, index=variabile, columns=[f"C{i+1}" for i in range(C.shape[1])])
t_com.to_csv("Comunalitati.csv")

plt.figure(figsize=(9, 7))
sb.heatmap(t_com, vmin=0, vmax=1, cmap="RdYlGn", annot=True)
plt.title("Corelograma comunalitati")
plt.show()

#cosinusuri (pe instante)
c2 = C**2
cosin = c2 / c2.sum(axis=1, keepdims=True)
t_cos = pd.DataFrame(cosin, index=df.index, columns=[f"C{i+1}" for i in range(C.shape[1])])
t_cos.to_csv("Cosinusuri.csv")

#contributii (pe componente)
contrib = c2 / c2.sum(axis=0, keepdims=True)
t_contrib = pd.DataFrame(contrib, index=df.index, columns=[f"C{i+1}" for i in range(C.shape[1])])
t_contrib.to_csv("Contributii.csv")
