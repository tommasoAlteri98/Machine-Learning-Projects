import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, adjusted_rand_score, homogeneity_score

dati_iris = datasets.load_iris()
dati = dati_iris.data
etichette_reali = dati_iris.target

num_gruppi = 3
modello = KMeans(n_clusters = num_gruppi, random_state = 42, n_init = 10)
modello.fit(dati)

etichette_gruppi = modello.labels_

matrice_confusione = confusion_matrix(etichette_reali, etichette_gruppi)
print("Matrice di Confusione:")
print(matrice_confusione)

indice_rand = adjusted_rand_score(etichette_reali, etichette_gruppi)
omegeneita = homogeneity_score(etichette_reali, etichette_gruppi)
print(f"Indice di Rand Adjusted: {indice_rand:.4f}")
print(f"Indice di Omogeneità: {omegeneita:.4f}")

riduzione_pca = PCA(n_components = 2)
dati_ridotti = riduzione_pca.fit_transform(dati)

plt.figure(figsize = (8, 6))
plt.scatter(dati_ridotti[:, 0], dati_ridotti[:, 1], c = etichette_gruppi, cmap = 'viridis', edgecolors = 'k')
plt.title("Raggruppamento con K-Means")
plt.xlabel("Componente Principale 1")
plt.ylabel("Componente Principale 2")
plt.show()

plt.figure(figsize = (6, 5))
sns.heatmap(matrice_confusione, annot = True, fmt = "d", cmap = "Blues", xticklabels = ["Gruppo 0", "Gruppo 1", "Gruppo 2"], yticklabels = ["Setosa", "Versicolor", "Virginica"])
plt.xlabel("Gruppi Predetti")
plt.ylabel("Specie Reali")
plt.title("Matrice di Confusione del Raggruppamento")
plt.show()
