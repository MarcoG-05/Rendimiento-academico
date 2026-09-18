import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Asegurar que existan las carpetas de salida
os.makedirs("outputs/resultados", exist_ok=True)

print("--- 1. CARGA DEL DATASET ---")
# 1. Cargar el dataset desde la carpeta data/
df = pd.read_csv("data/dataset.csv")
print(f"Dataset cargado exitosamente con {df.shape[0]} filas y {df.shape[1]} columnas.\n")

print("--- 2. EXPLORACIÓN INICIAL ---")
print("Primeras filas del dataset:")
print(df.head(), "\n")
print("Tipos de datos y valores nulos:")
print(df.info(), "\n")
print("Valores faltantes por columna:")
print(df.isnull().sum(), "\n")
print("Estadísticas descriptivas:")
print(df.describe(), "\n")

# 3. Limpieza y preprocesamiento (eliminación de duplicados por seguridad)
df = df.drop_duplicates()

print("--- 3. CREACIÓN DE NUEVAS VARIABLES ---")
# 4. Crear la nueva variable 'average_score' (promedio de las 3 áreas)
df['average_score'] = (df['math score'] + df['reading score'] + df['writing score']) / 3

# 5. Crear clasificación del rendimiento académico
def clasificar_rendimiento(score):
    if score < 60:
        return 'Bajo'
    elif score < 80:
        return 'Medio'
    else:
        return 'Alto'

df['performance_category'] = df['average_score'].apply(clasificar_rendimiento)
print("Variable 'average_score' y categoría de rendimiento creadas exitosamente.\n")

print("--- 4. ANÁLISIS REQUERIDOS ---")
# Análisis 1: Promedio por área académica
promedios_areas = df[['math score', 'reading score', 'writing score']].mean()
print("1. Promedio general por área:\n", promedios_areas, "\n")

# Análisis 2: Impacto del curso de preparación en el promedio
prep_analisis = df.groupby('test preparation course')['average_score'].mean()
print("2. Promedio según curso de preparación:\n", prep_analisis, "\n")

# Análisis 3: Rendimiento según el nivel educativo de los padres
padres_analisis = df.groupby('parental level of education')['average_score'].mean().sort_values(ascending=False)
print("3. Promedio según nivel educativo de los padres:\n", padres_analisis, "\n")

# Análisis 4: Distribución porcentual de las categorías de rendimiento
porcentaje_categorias = df['performance_category'].value_counts(normalize=True) * 100
print("4. Porcentaje de estudiantes por categoría de rendimiento:\n", porcentaje_categorias, "\n")

print("--- 5. GENERACIÓN DE VISUALIZACIONES ---")
sns.set_theme(style="whitegrid")

# Visualización 1: Distribución de los promedios generales
plt.figure(figsize=(8, 5))
sns.histplot(df['average_score'], bins=20, kde=True, color='skyblue')
plt.title('Distribución del Promedio de Calificaciones')
plt.xlabel('Promedio')
plt.ylabel('Frecuencia')
plt.savefig('outputs/resultados/distribucion_promedios.png')
plt.close()

# Visualización 2: Promedio según el curso de preparación
plt.figure(figsize=(6, 4))
sns.barplot(x=prep_analisis.index, y=prep_analisis.values, palette='Set2', hue=prep_analisis.index, legend=False)
plt.title('Impacto del Curso de Preparación en el Promedio')
plt.xlabel('Curso de Preparación')
plt.ylabel('Promedio General')
plt.savefig('outputs/resultados/impacto_curso_preparacion.png')
plt.close()

# Visualización 3: Rendimiento según nivel educativo de los padres
plt.figure(figsize=(10, 5))
sns.barplot(x=padres_analisis.values, y=padres_analisis.index, palette='viridis', hue=padres_analisis.index, legend=False)
plt.title('Rendimiento Académico vs Nivel Educativo de los Padres')
plt.xlabel('Promedio General')
plt.ylabel('Nivel Educativo')
plt.tight_layout()
plt.savefig('outputs/resultados/rendimiento_padres.png')
plt.close()

print("¡Gráficas guardadas con éxito en la carpeta 'outputs/resultados/'!")