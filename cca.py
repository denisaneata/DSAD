# =========================
# ANALIZA CORELATIILOR CANONICE (ACC / CCA)
# =========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

# 1) citire date
df = pd.read_csv("DataSet_34.csv", index_col=0)

# 2) definire seturi X si Y (schimbi coloanele in functie de subiect)
x_cols = df.columns[:4]     # <-- productie
y_cols = df.columns[4:]     # <-- consum

X = df[x_cols].values
Y = df[y_cols].values

# 3) standardizare (OBLIGATORIU)
scx = StandardScaler()
scy = StandardScaler()
X_std = scx.fit_transform(X)
Y_std = scy.fit_transform(Y)

pd.DataFrame(X_std, index=df.index, columns=x_cols).to_csv("Xstd.csv")
pd.DataFrame(Y_std, index=df.index, columns=y_cols).to_csv("Ystd.csv")

# 4) model CCA
m = min(X.shape[1], Y.shape[1])
cca = CCA(n_components=m)
cca.fit(X_std, Y_std)

# 5) scoruri canonice
Z, U = cca.transform(X_std, Y_std)

t_Z = pd.DataFrame(Z, index=df.index, columns=[f"z{i+1}" for i in range(m)])
t_U = pd.DataFrame(U, index=df.index, columns=[f"u{i+1}" for i in range(m)])

t_Z.to_csv("Xscore.csv")
t_U.to_csv("Yscore.csv")

# 6) corelatii canonice (r)
r = np.array([np.corrcoef(Z[:, i], U[:, i])[0, 1] for i in range(m)])
print("Corelatii canonice:", r)

# 7) corelatii variabile observate - variabile canonice
corr_xz = np.corrcoef(X_std, Z, rowvar=False)
Rxz = corr_xz[:X.shape[1], X.shape[1]:]
t_Rxz = pd.DataFrame(Rxz, index=x_cols, columns=[f"z{i+1}" for i in range(m)])
t_Rxz.to_csv("Rxz.csv")

corr_yu = np.corrcoef(Y_std, U, rowvar=False)
Ryu = corr_yu[:Y.shape[1], Y.shape[1]:]
t_Ryu = pd.DataFrame(Ryu, index=y_cols, columns=[f"u{i+1}" for i in range(m)])
t_Ryu.to_csv("Ryu.csv")

# 8) biplot (z1,z2 si u1,u2) cu protectie
if m >= 2:
    plt.figure(figsize=(9, 9))
    plt.axhline(0); plt.axvline(0)

    plt.scatter(t_Z["z1"], t_Z["z2"], label="Z (X)", marker="o")
    plt.scatter(t_U["u1"], t_U["u2"], label="U (Y)", marker="s")

    for i in range(len(df.index)):
        plt.text(t_Z["z1"].iloc[i], t_Z["z2"].iloc[i], df.index[i])

    plt.title("Biplot (z1,z2) si (u1,u2)")
    plt.legend()
    plt.show()
else:
    print("Nu pot face biplot: m < 2")
