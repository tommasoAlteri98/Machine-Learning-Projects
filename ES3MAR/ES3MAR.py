import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import randint, uniform

# dataset
percorso_estrazione = r"C:/Users/Utente/Desktop/CORSO PHYTON/ES3MAR"
dati_train = pd.read_csv(os.path.join(percorso_estrazione, "train.csv"))
dati_test = pd.read_csv(os.path.join(percorso_estrazione, "test.csv"))
dati_sottomissione = pd.read_csv(os.path.join(percorso_estrazione, "gender_submission.csv"))

# valori nulli
dati_train.loc[:, "Age"] = dati_train["Age"].fillna(dati_train["Age"].median())
dati_train.loc[:, "Fare"] = dati_train["Fare"].fillna(dati_train["Fare"].median())
dati_train.loc[:, "Embarked"] = dati_train["Embarked"].fillna(dati_train["Embarked"].mode()[0])

dati_test.loc[:, "Age"] = dati_test["Age"].fillna(dati_test["Age"].median())
dati_test.loc[:, "Fare"] = dati_test["Fare"].fillna(dati_test["Fare"].median())
dati_test.loc[:, "Embarked"] = dati_test["Embarked"].fillna(dati_test["Embarked"].mode()[0])

# Conversione in num
dati_train["Sex"] = dati_train["Sex"].map({"male": 0, "female": 1})
dati_train["Embarked"] = dati_train["Embarked"].map({"S": 0, "C": 1, "Q": 2})

dati_test["Sex"] = dati_test["Sex"].map({"male": 0, "female": 1})
dati_test["Embarked"] = dati_test["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# caratteristiche e target
caratteristiche = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
target = "Survived"
X = dati_train[caratteristiche]
y = dati_train[target]

# scaling numeriche
scaler = StandardScaler()
X.loc[:, "Age"] = scaler.fit_transform(X[["Age"]])
X.loc[:, "Fare"] = scaler.fit_transform(X[["Fare"]])

# divisione
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Gradient Boosting
modello_gb = GradientBoostingClassifier(random_state=42)

# griglia di iperparametri per RandomizedSearchCV
distribuzione_parametri = {
    "n_estimators": randint(50, 200),
    "learning_rate": uniform(0.01, 0.2),
    "max_depth": randint(1, 5),
    "min_samples_split": randint(2, 10),
    "min_samples_leaf": randint(1, 10),
}

# ricerca degli iperparametri con StratifiedKFold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ricerca_casuale = RandomizedSearchCV(
    modello_gb, param_distributions=distribuzione_parametri, n_iter=20, cv=kf,
    scoring="accuracy", random_state=42, n_jobs=-1
)

# addestramento del modello
ricerca_casuale.fit(X_train, y_train)

# valutazione
previsioni = ricerca_casuale.best_estimator_.predict(X_test)
accuratezza = accuracy_score(y_test, previsioni)

# Risultati
print("Migliori iperparametri:", ricerca_casuale.best_params_)
print("Accuratezza sul test set:", accuratezza)

# matrice di confusione
cm = confusion_matrix(y_test, previsioni)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non sopravvissuto", "Sopravvissuto"], yticklabels=["Non sopravvissuto", "Sopravvissuto"])
plt.xlabel("Predetto")
plt.ylabel("Reale")
plt.title("Matrice di Confusione")
plt.show()

# grafico distribuzione delle età tra sopravvissuti e non
plt.figure(figsize=(8, 5))
sns.histplot(dati_train, x="Age", hue="Survived", bins=30, kde=True, palette={0: "red", 1: "green"})
plt.xlabel("Età")
plt.ylabel("Conteggio")
plt.title("Distribuzione dell'Età tra Sopravvissuti e Non")
plt.show()