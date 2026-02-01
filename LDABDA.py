# ======= SCHEMA GENERALA: LDA + BDA (Naive Bayes) =======
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, cohen_kappa_score

# -------------------------------------------------------
# (0) CITIRE DATE + SETARI (AICI ADAPTEZI)
# -------------------------------------------------------
train = pd.read_csv("train.csv")          # <-- schimbi: fisier cu tinta
apply = pd.read_csv("apply.csv")          # <-- schimbi: fisier fara tinta (sau "test.csv")

tinta = "outcome"                         # <-- schimbi daca tinta are alt nume
id_col = "Id_sim"                         # <-- optional: daca exista ID; altfel pune None

# predictorii:
# Varianta 1 (cea mai sigura): exclud tinta si ID daca exista
predictori = train.columns.drop([tinta] + ([id_col] if id_col in train.columns else []))

# Daca vrei, poti forta interval "de la ... la ..." (cand cerinta zice explicit)
# predictori = train.loc[:, "vconst_corr":"Prandtl"].columns

Xtr = train[predictori].values
ytr = train[tinta].values
Xap = apply[predictori].values

# -------------------------------------------------------
# FUNCTII UTILE (OPTIONAL)
# -------------------------------------------------------
def acc(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc_global = np.round(np.diag(cm).sum() * 100 / cm.sum(), 3)
    acc_grup = np.round(np.diag(cm) * 100 / cm.sum(axis=1), 3)
    acc_mediu = np.round(acc_grup.mean(), 3)
    return cm, acc_global, acc_grup, acc_mediu

# -------------------------------------------------------
# (1) LDA: antrenare + (optional) evaluare
# -------------------------------------------------------
lda = LinearDiscriminantAnalysis()
lda.fit(Xtr, ytr)
clase_lda = lda.classes_

# (optional evaluare pe train - NU e cerut mereu, dar e ok)
y_pred_train_lda = lda.predict(Xtr)
cm_lda, acc_g_lda, acc_gr_lda, acc_m_lda = acc(ytr, y_pred_train_lda, clase_lda)

print("--- LDA (train) ---")
print("Global:", acc_g_lda)
print("Grup:", acc_gr_lda)
print("Mediu:", acc_m_lda)
print("Kappa:", cohen_kappa_score(ytr, y_pred_train_lda))

# salvare optionala
pd.DataFrame(cm_lda, index=clase_lda, columns=clase_lda).to_csv("MatConf_LDA.csv")
pd.DataFrame({tinta: ytr, "Pred_LDA": y_pred_train_lda}).to_csv("TrainPred_LDA.csv", index=False)

# predictie pe apply/test (daca ai nevoie)
pred_lda_ap = lda.predict(Xap)
pd.DataFrame({"Pred_LDA": pred_lda_ap}).to_csv("ApplyPred_LDA.csv", index=False)

# -------------------------------------------------------
# (2) BDA: antrenare + predictii
# -------------------------------------------------------
bda = GaussianNB()
bda.fit(Xtr, ytr)
clase_bda = bda.classes_

y_pred_train_bda = bda.predict(Xtr)
cm_bda, acc_g_bda, acc_gr_bda, acc_m_bda = acc(ytr, y_pred_train_bda, clase_bda)

print("--- BDA (train) ---")
print("Global:", acc_g_bda)
print("Grup:", acc_gr_bda)
print("Mediu:", acc_m_bda)
print("Kappa:", cohen_kappa_score(ytr, y_pred_train_bda))

pred_bda_ap = bda.predict(Xap)
pd.DataFrame({"Pred_BDA": pred_bda_ap}).to_csv("ApplyPred_BDA.csv", index=False)

# -------------------------------------------------------
# (B1) SCORURI DISCRIMINANTE -> z.csv  (DOAR LDA)
# -------------------------------------------------------
Z = lda.transform(Xtr)   # scorurile in axele LD

z = pd.DataFrame(Z, columns=[f"LD{i+1}" for i in range(Z.shape[1])])

# daca vrei sa legi scorurile de observatii (optional, util):
if id_col in train.columns:
    z.insert(0, id_col, train[id_col].values)

z.to_csv("z.csv", index=False)

# -------------------------------------------------------
# (B2) GRAFICE DISTRIBUTIE IN AXELE DISCRIMINANTE
# -------------------------------------------------------
# 2a) Scatter LD1-LD2 daca exista minim 2 axe
if Z.shape[1] >= 2:
    plt.figure(figsize=(9, 9))
    sb.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=ytr)
    plt.axhline(0); plt.axvline(0)
    plt.xlabel("LD1"); plt.ylabel("LD2")
    plt.title("Instante in spatiul axelor discriminante (LD1-LD2)")
    plt.show()
else:
    print("Nu pot desena scatter LD1/LD2: exista < 2 axe discriminante")

# 2b) Distributii pe fiecare axa (KDE sau hist)
for i in range(Z.shape[1]):
    plt.figure(figsize=(9, 6))
    for cls in clase_lda:
        sb.kdeplot(Z[ytr == cls, i], fill=True, label=f"outcome={cls}")
    plt.title(f"Distributie pe LD{i+1}")
    plt.legend()
    plt.show()

# -------------------------------------------------------
# (B3) DIFERENTE LDA vs BDA pe setul de aplicare
# -------------------------------------------------------
mask = pred_lda_ap != pred_bda_ap

cols_out = []
if id_col in apply.columns:
    cols_out.append(id_col)

dif = apply.loc[mask, cols_out].copy() if cols_out else pd.DataFrame(index=apply.index[mask])

dif["Pred_LDA"] = pred_lda_ap[mask]
dif["Pred_BDA"] = pred_bda_ap[mask]
dif.to_csv("diferente.csv", index=False)
