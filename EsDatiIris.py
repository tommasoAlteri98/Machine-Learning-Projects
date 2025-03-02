import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# dataset Iris
dati_iris = load_iris()
df_iris = pd.DataFrame(dati_iris.data, columns = dati_iris.feature_names)
df_iris['specie'] = pd.Categorical.from_codes(dati_iris.target, dati_iris.target_names)

# info base
print("Informazioni sul dataset:")
print(df_iris.info())
print("\nValori nulli presenti:")
print(df_iris.isnull().sum())

# Statistica descrittiva
statistiche_descrittive = df_iris.describe()
print("\nStatistiche descrittive:")
print(statistiche_descrittive)

# deviazione standard per specie
deviazione_standard = df_iris.groupby('specie').std()
print("\nDeviazione standard per specie:")
print(deviazione_standard)

# media per specie
media_specie = df_iris.groupby('specie').mean()
print("\nMedia per specie:")
print(media_specie)

# valore minimo per specie
minimo_specie = df_iris.groupby('specie').min()
print("\nValore minimo per specie:")
print(minimo_specie)

# valore massimo per specie
massimo_specie = df_iris.groupby('specie').max()
print("\nValore massimo per specie:")
print(massimo_specie)

# Conteggio delle specie
conteggio_specie = df_iris['specie'].value_counts()
print("\nConteggio delle specie:")
print(conteggio_specie)

# conteggio specie
sns.countplot(x = 'specie', data = df_iris)
plt.title("Distribuzione delle specie nel dataset Iris")
plt.xlabel("Specie")
plt.ylabel("Conteggio")
plt.show()

# correlazioni
matrice_correlazione = df_iris.iloc[:, :-1].corr()
print("\nMatrice di correlazione:")
print(matrice_correlazione)

sns.heatmap(matrice_correlazione, annot = True, cmap = 'coolwarm')
plt.title("Matrice di correlazione delle caratteristiche")
plt.show()

# Raggruppamenti per specie
raggruppamento_specie = df_iris.groupby('specie').agg(['mean', 'max'])
print("\nStatistiche per specie (media e massimo):")
print(raggruppamento_specie)

# distribuzioni per specie
sns.pairplot(df_iris, hue = 'specie', diag_kind = 'kde')
plt.show()

print("Il grafico seguente mostra la disrtibuzione delle caratteristiche del dataset per ogni specie. Aiuta a individuare la variabilità, la presenza di outlier e le differenze tra le specie.")
plt.figure(figsize = (12, 6))
sns.boxplot(x = 'specie', y = 'sepal length (cm)', data = df_iris)
plt.title("Boxplot della lunghezza del sepalo per specie")
plt.xlabel("Specie")
plt.ylabel("Lunghezza del sepalo (cm)")
plt.show()


print("Il grafico a torta seguente mostra la distribuzione delle specie nel dataset, fornendo una rappresentazione visiva della loro proporzione relativa.")
plt.figure(figsize = (8, 8))
plt.pie(conteggio_specie, labels = conteggio_specie.index, autopct = '%1.1f%%', colors = sns.color_palette('pastel'))
plt.title("Distribuzione percentuale delle specie nel dataset Iris")
plt.show()