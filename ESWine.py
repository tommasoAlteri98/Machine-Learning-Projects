import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import numpy as np

vino_dataset = datasets.load_wine()
tabella_vini = pd.DataFrame(vino_dataset.data, columns=vino_dataset.feature_names)
tabella_vini['categoria'] = vino_dataset.target  

scalatore = StandardScaler()
matrice_normalizzata = scalatore.fit_transform(tabella_vini.drop(columns=['categoria']))
etichette = tabella_vini['categoria']

dati_addestramento, dati_verifica, etichette_addestramento, etichette_verifica = train_test_split(
    matrice_normalizzata, etichette, test_size=0.2, random_state=42, stratify=etichette
)

riduttore_dimensionale = PCA(n_components=2)
componenti_addestramento = riduttore_dimensionale.fit_transform(dati_addestramento)
componenti_verifica = riduttore_dimensionale.transform(dati_verifica)

tabella_componenti = pd.DataFrame(componenti_addestramento, columns=['Asse1', 'Asse2'])
tabella_componenti['categoria'] = etichette_addestramento.values

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=tabella_componenti['Asse1'], 
    y=tabella_componenti['Asse2'], 
    hue=tabella_componenti['categoria'], 
    palette=['red', 'green', 'blue'], 
    style=tabella_componenti['categoria'], 
    s=100
)

plt.xlabel("Prima Componente Principale")
plt.ylabel("Seconda Componente Principale")
plt.title("PCA - Rappresentazione Bidimensionale del Set di Addestramento")
plt.legend(title="Tipo di Vino", labels=vino_dataset.target_names)
plt.grid(True)
plt.show()

print(f"Dimensioni Set di Addestramento: {dati_addestramento.shape}")
print(f"Dimensioni Set di Verifica: {dati_verifica.shape}")

modello = RandomForestClassifier(n_estimators=100, random_state=42)
modello.fit(dati_addestramento, etichette_addestramento)

predizioni = modello.predict(dati_verifica)

accuratezza = accuracy_score(etichette_verifica, predizioni)
precisione = precision_score(etichette_verifica, predizioni, average='weighted')
richiamo = recall_score(etichette_verifica, predizioni, average='weighted')
f1 = f1_score(etichette_verifica, predizioni, average='weighted')

report_classificazione = classification_report(etichette_verifica, predizioni, target_names=vino_dataset.target_names)

print(f"Accuratezza del modello: {accuratezza:.2f}")
print(f"Precisione: {precisione:.2f}")
print(f"Recall: {richiamo:.2f}")
print(f"F1-score: {f1:.2f}")
print("\nReport di classificazione:\n", report_classificazione)

importanza_caratteristiche = modello.feature_importances_
nomi_caratteristiche = vino_dataset.feature_names

indice_ordinato = np.argsort(importanza_caratteristiche)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(
    x=np.array(nomi_caratteristiche)[indice_ordinato], 
    y=importanza_caratteristiche[indice_ordinato], 
    palette="viridis"
)

plt.xticks(rotation=90)
plt.xlabel("Caratteristiche")
plt.ylabel("Importanza")
plt.title("Importanza delle Caratteristiche nel Modello Random Forest")
plt.show()
