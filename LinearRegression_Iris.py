# LinearRegression_Iris.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Cargar dataset Iris
iris = load_iris(as_frame=True)
data_iris = iris.frame

# Selección de características independientes y variable dependiente
caracteristicas_iris = ['sepal length (cm)', 'sepal width (cm)', 'petal width (cm)']
X_iris = data_iris[caracteristicas_iris]
y_iris = data_iris['petal length (cm)']

def evaluar_modelo(X, y, dataset):
    # División en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    model = LinearRegression()

    # Entrenamiento
    model.fit(X_train, y_train)

    # Evaluación
    score = model.score(X_test, y_test)
    print(f"Parámetro R^2 del dataset {dataset}: {score:.4f}")

    # Predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Gráfico real vs predicho
    plt.scatter(y_test, y_pred, color="blue", alpha=0.6, edgecolor="k")
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             "r--", lw=2)
    plt.xlabel("Valores reales")
    plt.ylabel("Valores predichos")
    plt.title(f"Regresión Lineal en {dataset}\nReal vs Predicho")
    plt.grid(True)
    plt.show()

    return model

def generar_flor_aleatoria():
    return {
        'sepal length (cm)': np.random.uniform(4.0, 8.0),
        'sepal width (cm)': np.random.uniform(2.0, 4.5),
        'petal width (cm)': np.random.uniform(0.1, 2.5)
    }

# Entrenar modelo con iris
modelo_iris = evaluar_modelo(X_iris, y_iris, "Iris")

# Generar flor aleatoria y predecir largo del pétalo
nueva_flor = generar_flor_aleatoria()
nueva_flor_df = pd.DataFrame([nueva_flor])[caracteristicas_iris]

largo_petalo = modelo_iris.predict(nueva_flor_df)

print("Nueva flor generada:", nueva_flor)
print(f"El largo estimado del pétalo es: {largo_petalo[0]:.2f} cm")
