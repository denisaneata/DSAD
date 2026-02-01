#analiza factoriala-varianta scurta

X = df.values
variabile = df.columns

fa_test = FactorAnalyzer(rotation=None)
fa_test.fit(X)
valori_proprii, _ = fa_test.get_eigenvalues()
n_factori = sum(valori_proprii > 1)

fa = FactorAnalyzer(n_factors=n_factori, rotation="varimax")
fa.fit(X)

var, prop, cum = fa.get_factor_variance()
pd.DataFrame({
    "Varianta": var,
    "Procent": prop,
    "Cumulat": cum
}).to_csv("Varianta.csv")
