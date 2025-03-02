import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, adjusted_rand_score, homogeneity_score, classification_report

class KMeansIrisApp:
    def __init__(self):
        self.dati, self.etichette_reali = self.carica_dati()
        self.modello = None
        self.etichette_predette = None

    def carica_dati(self):
        dati_iris = datasets.load_iris()
        return dati_iris.data, dati_iris.target

    def genera_modello(self, num_gruppi = 3):
        self.modello = KMeans(n_clusters = num_gruppi, random_state = 42, n_init = 10)
        self.modello.fit(self.dati)
        self.etichette_predette = self.modello.labels_
        print("Modello generato con successo!")

    def analisi_descrittiva(self):
        print("Statistiche Descrittive dei Dati:")
        print(f"Media: {np.mean(self.dati, axis = 0)}")
        print(f"Deviazione Standard: {np.std(self.dati, axis = 0)}")

    def visualizza_grafico(self):
        if self.etichette_predette is not None:
            riduzione_pca = PCA(n_components = 2)
            dati_ridotti = riduzione_pca.fit_transform(self.dati)
            plt.figure(figsize = (8, 6))
            plt.scatter(dati_ridotti[:, 0], dati_ridotti[:, 1], c = self.etichette_predette, cmap = 'viridis', edgecolors = 'k')
            plt.title("Raggruppamento con K-Means")
            plt.xlabel("Componente Principale 1")
            plt.ylabel("Componente Principale 2")
            plt.show()
        else:
            print("Genera prima il modello!")

    def risultati_classification_report(self):
        if self.etichette_predette is not None:
            print("Report di Classificazione:")
            print(classification_report(self.etichette_reali, self.etichette_predette))
        else:
            print("Genera prima il modello!")