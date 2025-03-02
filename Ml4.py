from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

diabete = load_diabetes()

X = diabete.data  
y = diabete.target  

df = pd.DataFrame(X, columns=diabete.feature_names)
df['Target'] = y

print(f"Numero di campioni: {X.shape[0]}")
print(f"Numero di caratteristiche: {X.shape[1]}")
print(f"Nomi delle caratteristiche: {diabete.feature_names}")
print("\nStatistiche descrittive:")
print(df.describe())

plt.hist(y, bins=30, edgecolor='blue')
plt.xlabel('Progressione della malattia')
plt.ylabel('Frequenza')
plt.title('Frequenza Progressione del Diabete')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modello = LinearRegression()
modello.fit(X_train, y_train)

predizioni = modello.predict(X_test)
mse = mean_squared_error(y_test, predizioni)
r2 = r2_score(y_test, predizioni)

print(f"Errore Quadratico Medio (MSE): {mse}")
print(f"Coefficiente di Determinazione (R²): {r2}")

plt.scatter(y_test, predizioni, alpha=0.5, color='blue')
plt.xlabel("Valori Reali")
plt.ylabel("Predizioni")
plt.title("Predizioni vs Valori Reali")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.show()