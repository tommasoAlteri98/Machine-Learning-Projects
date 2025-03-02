from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Caricare dataset Iris
irisDataset = load_iris()
X, y = irisDataset.data, irisDataset.target

# Suddivide dati in training (80%) e test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applica K-Nearest Neighbors con n_neighbors=5
kneigh = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='minkowski', p=2)
kneigh.fit(X_train, y_train)

# Fare previsioni
y_pred = kneigh.predict(X_test)

# Valutare l'accuratezza
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuratezza del modello: {accuracy:.2f}')
