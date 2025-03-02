import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report)

# 1 dataset Wine
dati_vino = load_wine()
X = dati_vino.data
y = dati_vino.target
df_vino = pd.DataFrame(X, columns = dati_vino.feature_names)
df_vino['categoria'] = y

# 2.A Esplorazione dataset
print("Dimensioni del dataset:", df_vino.shape)
print("\nNumero di campioni per ciascuna classe:")
print(df_vino['categoria'].value_counts())
print("\nStatistiche descrittive delle caratteristiche:")
print(df_vino.describe())
# 2.B Visualizzazione delle classi
plt.figure(figsize = (8, 6))
sns.countplot(x = 'categoria', data = df_vino)
plt.title('Distribuzione delle classi nel dataset Vino')
plt.xlabel('Classe di vino')
plt.ylabel('Numero di campioni')
plt.show()

# 3.A Riduzione della dimensionalità con PCA
riduzione_pca = PCA(n_components = 2)
X_ridotto = riduzione_pca.fit_transform(X)
# 3.B Visualizzazione PCA: grafico a dispersione 2D
plt.figure(figsize = (8, 6))
sns.scatterplot(x = X_ridotto[:, 0], y = X_ridotto[:, 1], hue = y, palette = 'viridis', s = 100)
plt.title('PCA: Prime 2 Componenti Principali')
plt.xlabel('Componente Principale 1')
plt.ylabel('Componente Principale 2')
plt.legend(title = 'Classe di vino')
plt.show()

# 4 Divisione dei dati in set di addestramento e test (80% - 20%)
X_addestramento, X_test, y_addestramento, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

# 5 Creazione e addestramento Random Forest
modello_random_forest = RandomForestClassifier(random_state = 42)
modello_random_forest.fit(X_addestramento, y_addestramento)
predizioni = modello_random_forest.predict(X_test)

# 6 Valutazione prestazioni
accuratezza = accuracy_score(y_test, predizioni)
precisione = precision_score(y_test, predizioni, average = 'weighted')
ripresa = recall_score(y_test, predizioni, average = 'weighted')
f1 = f1_score(y_test, predizioni, average='weighted')

print("\nValutazione del modello Random Forest:")
print(f"Accuratezza: {accuratezza:.4f}")
print(f"Precisione: {precisione:.4f}")
print(f"Richiamo: {ripresa:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nRapporto di Classificazione:\n", classification_report(y_test, predizioni))

# 7
importanza_caratteristiche = modello_random_forest.feature_importances_
importanza_ord = pd.Series(importanza_caratteristiche, index = dati_vino.feature_names).sort_values(ascending = False)

plt.figure(figsize = (10, 6))
sns.barplot(x = importanza_ord, y=importanza_ord.index)
plt.title("Importanza delle caratteristiche secondo Random Forest")
plt.xlabel("Importanza")
plt.ylabel("Caratteristica")
plt.show()

# 8 confusion matrix
matrice_confusione = confusion_matrix(y_test, predizioni)
plt.figure(figsize = (8, 6))
sns.heatmap(matrice_confusione, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = dati_vino.target_names, yticklabels = dati_vino.target_names)
plt.title("Matrice di Confusione")
plt.xlabel("Classe Predetta")
plt.ylabel("Classe Reale")
plt.show()

# 9 Ottimizzazione del modello
parametri_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state = 42), parametri_grid, cv = 5, scoring = 'accuracy')
grid_search.fit(X_addestramento, y_addestramento)

print("Migliori parametri trovati:", grid_search.best_params_)
print(f"Miglior accuratezza ottenuta (CV): {grid_search.best_score_:.4f}")