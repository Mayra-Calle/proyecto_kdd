import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report)

st.set_page_config(page_title="Proyecto KDD - Estudiantes",layout="wide")
st.title("Proyecto KDD con Datos de Estudiantes")


archivo1 = st.file_uploader("Subir archivo Excel 1",type=["xls", "xlsx"])

archivo2 = st.file_uploader("Subir archivo Excel 2",type=["xls", "xlsx"])

if archivo1 and archivo2:

    st.markdown("## 1. Seleccion de Datos")
st.write("""En esta fase se seleccionan los datos relevantes para el análisis.""")
df1 = pd.read_excel(archivo1)
df2 = pd.read_excel(archivo2)
df = pd.concat([df1, df2], ignore_index=True)

st.success("Archivos cargados correctamente")

st.subheader("Vista previa de los datos")
st.dataframe(df.head())

st.subheader("Dimensiones del dataset")
st.write(f"Filas: {df.shape[0]}")
st.write(f"Columnas: {df.shape[1]}")
st.markdown("## 2. Preprocesamiento de Datos")

st.write("""En esta etapa se limpian los datos: - Valores nulos  - Tipos de datos - Duplicados""")

st.subheader("Tipos de datos")
st.write(df.dtypes)

st.subheader("Valores nulos")
st.write(df.isnull().sum())


duplicados = df.duplicated().sum()

st.write(f"Registros duplicados encontrados: {duplicados}")

df = df.drop_duplicates()
df = df.fillna("Desconocido")
st.success("Preprocesamiento completado")
st.markdown("## 3️. Transformación")

st.write("""En esta fase los datos categóricos se convierten en valores numéricos para ser utilizados por el modelo de Machine Learning.""")

datos = df.copy()

columnas_categoricas = [
    "sexo",
    "tipo_colegio",
    "etnia",
    "provincia_residencia",
    "tipo_estudiante"
]

le = LabelEncoder()

for col in columnas_categoricas:

    datos[col] = datos[col].astype(str)

    datos[col] = le.fit_transform(datos[col])


st.subheader("Datos transformados")
st.dataframe(datos.head())

X = datos[
    [
        "sexo",
        "tipo_colegio",
        "etnia",
        "provincia_residencia"
    ]
]
y = datos["tipo_estudiante"]

st.write("Variables predictoras:")
st.write(X.columns.tolist())

st.markdown("## 4. Minería de Datos")

st.write("""Se aplica un algoritmo de Machine Learning para encontrar patrones y realizar predicciones.""")

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.3,
random_state=42)

modelo = RandomForestClassifier(n_estimators=100,random_state=42)
modelo.fit(X_train, y_train)
predicciones = modelo.predict(X_test)
st.success("Modelo entrenado correctamente")

st.markdown("## 5. Evaluación")

st.write("""Se evalúa el rendimiento del modelo mediante métricas de precisión.""")

accuracy = accuracy_score(y_test, predicciones)

st.metric(label="Accuracy del Modelo",value=round(accuracy, 2))
st.subheader("Matriz de Confusión")
matriz = confusion_matrix(y_test, predicciones)
st.write(matriz)
st.subheader("Reporte de Clasificación")
reporte = classification_report(y_test, predicciones)
st.text(reporte)

st.markdown("## 6. Presentación del Conocimiento")

st.write("""Finalmente se presentan los resultados obtenidos mediante gráficos y conclusiones.""")
col1, col2 = st.columns(2)


st.subheader("Distribución por Sexo")
fig1, ax1 = plt.subplots()
df["sexo"].value_counts().plot(kind="bar", ax=ax1)
ax1.set_xlabel("Sexo")
ax1.set_ylabel("Cantidad")  
st.pyplot(fig1)

st.subheader("Tipo de Colegio")
fig2, ax2 = plt.subplots()
df["tipo_colegio"].value_counts().plot(kind="bar",ax=ax2)
ax2.set_xlabel("Tipo Colegio")
ax2.set_ylabel("Cantidad")
st.pyplot(fig2)

st.subheader("Top 10 Carreras")
fig3, ax3 = plt.subplots(figsize=(10, 5))
df["nombre_carrera"].value_counts().head(10).plot(kind="bar",ax=ax3)
ax3.set_xlabel("Carrera")
ax3.set_ylabel("Cantidad")
st.pyplot(fig3)

st.success(f"""El proceso KDD se ejecutó correctamente. Se logró entrenar un modelo de Random Forest con una precisión del {round(accuracy, 2)}. Se identificaron patrones en la distribución de sexo, tipo de colegio y carreras más populares entre los estudiantes.""")