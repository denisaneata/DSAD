# ===== CCA CRIZA =====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("DataSet_34.csv", index_col=0)  # <-- schimbi

x_cols = df.columns[:4]  # <-- schimbi
y_cols = df.columns[4:]  # <-- schimbi
X = df[x_cols].values
Y = df[y_cols].values

Xstd = StandardScaler().fit_transform(X)
Ystd = StandardScaler().fit_transform(Y)

pd.DataFrame(Xstd, index=df.index, columns=x_cols).to_csv("Xstd.csv")
pd.DataFrame(Ystd, index=df.index, columns=y_cols).to_csv("Ystd.csv")

m = min(X.shape[1], Y.shape[1])
cca = CCA(n_components=m)
cca.fit(Xstd, Ystd)
Z, U = cca.transform(Xstd, Ystd)

pd.DataFrame(Z, index=df.index, columns=[f"z{i+1}" for i in range(m)]).to_csv("Xscore.csv")
pd.DataFrame(U, index=df.index, columns=[f"u{i+1}" for i in range(m)]).to_csv("Yscore.csv")

Rxz = np.corrcoef(Xstd, Z, rowvar=False)[:X.shape[1], X.shape[1]:]
Ryu = np.corrcoef(Ystd, U, rowvar=False)[:Y.shape[1], Y.shape[1]:]
pd.DataFrame(Rxz, index=x_cols, columns=[f"z{i+1}" for i in range(m)]).to_csv("Rxz.csv")
pd.DataFrame(Ryu, index=y_cols, columns=[f"u{i+1}" for i in range(m)]).to_csv("Ryu.csv")

if m >= 2:
    plt.figure(figsize=(7,7))
    plt.axhline(0); plt.axvline(0)
    plt.scatter(Z[:,0], Z[:,1], label="Z")
    plt.scatter(U[:,0], U[:,1], label="U")
    plt.legend()
    plt.title("Biplot z1-z2 si u1-u2")
    plt.show()
