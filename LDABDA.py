

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, cohen_kappa_score

# 1) citim datele
train = pd.read_csv("train.csv")        # <-- schimbi fisierul
test  = pd.read_csv("test.csv")         # <-- schimbi fisierul (daca exista)

tinta = train.columns[-1]               # <-- variabila tinta (ultima coloana)
predictori = train.columns[:-1]         # <-- restul coloanelor

X_train = train[predictori].values
y_train = train[tinta].values
X_test  = test[predictori].values

# functie acuratete (ca in seminar)
def acc(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc_global = np.round(np.diag(cm).sum() * 100 / cm.sum(), 3)
    acc_grup = np.round(np.diag(cm) * 100 / cm.sum(axis=1), 3)
    acc_mediu = np.round(acc_grup.mean(), 3)
    return cm, acc_global, acc_grup, acc_mediu

# 2) BDA (GaussianNB)
bda = GaussianNB()
bda.fit(X_train, y_train)
clase_bda = bda.classes_

y_pred_train_bda = bda.predict(X_train)
y_pred_test_bda  = bda.predict(X_test)

cm_bda, acc_g_bda, acc_gr_bda, acc_m_bda = acc(y_train, y_pred_train_bda, clase_bda)

print("---BDA---")
print("Global:", acc_g_bda)
print("Grup:", acc_gr_bda)
print("Mediu:", acc_m_bda)
print("Kappa:", cohen_kappa_score(y_train, y_pred_train_bda))

pd.DataFrame(cm_bda, index=clase_bda, columns=clase_bda).to_csv("MatConf_BDA.csv")
pd.DataFrame({tinta: y_train, "Pred_BDA": y_pred_train_bda}).to_csv("TrainPred_BDA.csv", index=False)
pd.DataFrame({"Pred_BDA": y_pred_test_bda}).to_csv("TestPred_BDA.csv", index=False)

# 3) LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
clase_lda = lda.classes_

y_pred_train_lda = lda.predict(X_train)
y_pred_test_lda  = lda.predict(X_test)

cm_lda, acc_g_lda, acc_gr_lda, acc_m_lda = acc(y_train, y_pred_train_lda, clase_lda)

print("---LDA---")
print("Global:", acc_g_lda)
print("Grup:", acc_gr_lda)
print("Mediu:", acc_m_lda)
print("Kappa:", cohen_kappa_score(y_train, y_pred_train_lda))

pd.DataFrame(cm_lda, index=clase_lda, columns=clase_lda).to_csv("MatConf_LDA.csv")
pd.DataFrame({tinta: y_train, "Pred_LDA": y_pred_train_lda}).to_csv("TrainPred_LDA.csv", index=False)
pd.DataFrame({"Pred_LDA": y_pred_test_lda}).to_csv("TestPred_LDA.csv", index=False)

# 4) scoruri discriminante + plot (daca ai minim 2 axe)
Z = lda.transform(X_train)
n_axe = min(X_train.shape[1], len(clase_lda) - 1)

if n_axe >= 2:
    plt.figure(figsize=(9, 9))
    sb.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=y_train, hue_order=clase_lda)
    plt.xlabel("LD1"); plt.ylabel("LD2")
    plt.title("Instante in spatiul axelor discriminante")
    plt.show()
else:
    print("Nu pot desena scatter LD1/LD2: <2 axe")

# 5) distributii pe axe (KDE)
for i in range(n_axe):
    plt.figure(figsize=(9, 6))
    for cls in clase_lda:
        sb.kdeplot(Z[y_train == cls, i], fill=True, label=cls)
    plt.title(f"Distributie pe LD{i+1}")
    plt.legend()
    plt.show()
