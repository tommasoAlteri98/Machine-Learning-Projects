import numpy as np

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


# 1. Creiamo un dataset di esempio

# Supponiamo di avere 200 campioni e 5 feature

np.random.seed(42)

X = np.random.rand(200, 5)


# 2. Normalizzazione delle feature (opzionale ma spesso consigliata)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# 3. Istanzia PCA

# n_components = 2 significa che vogliamo ridurre i dati a 2 componenti principali

pca = PCA(n_components=2)


# 4. Eseguiamo PCA

X_pca = pca.fit_transform(X_scaled)


# 5. Possiamo stampare la varianza spiegata da ogni componente

print("Varianza spiegata da ciascuna componente:", pca.explained_variance_ratio_)


# 6. Visualizzazione dei dati su 2 componenti principali

plt.figure(figsize=(8,6))

plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', edgecolor='k', alpha=0.7)

plt.xlabel("Prima componente principale")

plt.ylabel("Seconda componente principale")

plt.title("Visualizzazione dati dopo PCA (2D)")

plt.show()