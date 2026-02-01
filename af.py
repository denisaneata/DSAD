#analiza factoriala

#facem mereu importurile 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from factor_analyzer import FactorAnalyzer

#citim datele

df = pd.read_csv("date.csv", index_col=0)
X = df.values
variabile = df.columns

#factorabilitatea

from factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

chi2, p = calculate_bartlett_sphericity(X)
kmo_all, kmo = calculate_kmo(X)

#numarul de factori(Kaiser)

fa_test = FactorAnalyzer(rotation=None)
fa_test.fit(X)

valori_proprii, _ = fa_test.get_eigenvalues()
n_factori = sum(valori_proprii > 1)

#modelul final(Varimax)
fa = FactorAnalyzer(n_factors=n_factori, rotation="varimax")
fa.fit(X)

#output-uri

#variatia factorilor
var, prop, cum = fa.get_factor_variance()

df_var = pd.DataFrame({
    "Varianta": var,
    "Procent": prop,
    "Cumulat": cum
}, index=[f"F{i+1}" for i in range(n_factori)])

df_var.to_csv("Varianta.csv")

#corelatii factoriale

loadings = fa.loadings_

df_r = pd.DataFrame(
    loadings,
    index=variabile,
    columns=[f"F{i+1}" for i in range(n_factori)]
)

df_r.to_csv("r.csv")

#cercul corelatiilor

plt.figure(figsize=(8,8))

theta = np.linspace(0, 2*np.pi, 200)
plt.plot(np.cos(theta), np.sin(theta))

plt.axhline(0)
plt.axvline(0)

plt.scatter(df_r["F1"], df_r["F2"])

for i in range(len(df_r)):
    plt.text(df_r["F1"].iloc[i], df_r["F2"].iloc[i], df_r.index[i])

plt.xlabel("F1")
plt.ylabel("F2")
plt.gca().set_aspect("equal")
plt.show()


