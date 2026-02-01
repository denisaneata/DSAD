# ===== DISCRIMINANTA CRIZA =====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

train = pd.read_csv("train.csv")  # <-- schimbi
test  = pd.read_csv("test.csv")   # <-- daca exista

tinta = train.columns[-1]
predictori = train.columns[:-1]

Xtr = train[predictori].values
ytr = train[tinta].values
Xte = test[predictori].values

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(Xtr, ytr)
yp = lda.predict(Xtr)
cm = confusion_matrix(ytr, yp, labels=lda.classes_)
pd.DataFrame(cm, index=lda.classes_, columns=lda.classes_).to_csv("MatConf_LDA.csv")
pd.DataFrame({tinta: ytr, "Pred_LDA": yp}).to_csv("TrainPred_LDA.csv", index=False)
pd.DataFrame({"Pred_LDA": lda.predict(Xte)}).to_csv("TestPred_LDA.csv", index=False)

# BDA
bda = GaussianNB()
bda.fit(Xtr, ytr)
yp2 = bda.predict(Xtr)
cm2 = confusion_matrix(ytr, yp2, labels=bda.classes_)
pd.DataFrame(cm2, index=bda.classes_, columns=bda.classes_).to_csv("MatConf_BDA.csv")

# Plot rapid daca ai 2 axe
Z = lda.transform(Xtr)
if Z.shape[1] >= 2:
    plt.figure(figsize=(7,7))
    plt.axhline(0); plt.axvline(0)
    plt.scatter(Z[:,0], Z[:,1])
    plt.title("Instante in LD1-LD2")
    plt.show()
