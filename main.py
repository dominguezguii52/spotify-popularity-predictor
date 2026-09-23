import pandas as pd

# Cargar el archivo CSV desde la carpeta data/raw
df = pd.read_csv('data/raw/spotify-tracks-dataset.csv')

# Mostrar las primeras 3 filas para ver que funcione
print("¡El archivo se leyó perfecto!")
print(df.head(3))
# 1. Ver los 5 artistas con más canciones en el dataset
print("\n--- TOP 5 ARTISTAS ---")
print(df['artists'].value_counts().head(5))

# 2. Ver un resumen estadístico de la popularidad de las canciones
print("\n--- ESTADÍSTICAS DE POPULARIDAD ---")
print(df['popularity'].describe())
# 3. Ver los 5 géneros con más canciones en el dataset
print("\n--- TOP 5 GÉNEROS ---")
print(df['track_genre'].value_counts().head(5))
# 4. Correlación entre características de audio y popularidad
print("\n--- CORRELACIÓN CON LA POPULARIDAD ---")
columnas_interes = ['popularity', 'danceability', 'energy', 'valence', 'tempo']
print(df[columnas_interes].corr()['popularity'])
import pandas as pd

# Cargar el dataset
df = pd.read_csv('data/raw/spotify-tracks-dataset.csv')

# 1. ¿Qué géneros tienen el promedio de popularidad más alto?
print("\n--- TOP GÉNEROS MÁS POPULARES ---")
top_genres = df.groupby('track_genre')['popularity'].mean().sort_values(ascending=False).head(5)
print(top_genres)

# 2. ¿Qué artistas tienen el promedio de popularidad más alto?
print("\n--- TOP ARTISTAS MÁS POPULARES ---")
top_artists_pop = df.groupby('artists')['popularity'].mean().sort_values(ascending=False).head(5)
print(top_artists_pop)