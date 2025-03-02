from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carica dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Standardizza caratteristiche
stand = StandardScaler()
X = stand.fit_transform(X)

# Suddivide dati training(70%) e test(30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Applica DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

#previsioni
prevY = clf.predict(X_test)

# Valuta performance del modello
report = classification_report(y_test, prevY, target_names=iris.target_names)
matriceConf = confusion_matrix(y_test, prevY)
matriceConf = confusion_matrix(y_test, prevY)

plt.figure(figsize=(6,4))
sns.heatmap(matriceConf, annot=True, fmt='d', cmap='Blues',
        xticklabels=iris.target_names,
        yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(matriceConf)
