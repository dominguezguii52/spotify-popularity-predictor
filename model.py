import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Cargar datos
df = pd.read_csv('data/raw/spotify-tracks-dataset.csv')

# 2. Seleccionar las columnas que vamos a usar para predecir (Features) y lo que queremos adivinar (Target)
features = ['danceability', 'energy', 'valence', 'tempo', 'acousticness', 'instrumentalness']
target = 'popularity'

# Limpiar posibles valores nulos por las dudas
df_model = df[features + [target]].dropna()

X = df_model[features]
y = df_model[target]

# 3. Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Entrenar un modelo de Regresión Lineal simple
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 5. Probar el modelo con los datos de prueba
predicciones = modelo.predict(X_test)

# 6. Ver cómo rindió el modelo
print("--- RESULTADOS DEL MODELO ---")
print(f"R² Score (cuánto explica el modelo): {r2_score(y_test, predicciones):.4f}")
print(f"Error cuadrático medio (MSE): {mean_squared_error(y_test, predicciones):.4f}")