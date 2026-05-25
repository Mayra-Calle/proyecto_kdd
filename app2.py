
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Proyecto CRISP-DM", layout="wide")

st.title("Proyecto CRISP-DM con Datos de Estudiantes")

st.markdown("## 1. Comprensión del Negocio")
st.write("""
Objetivo: Analizar características de los estudiantes y predecir el tipo de estudiante.
""")

archivo1 = st.file_uploader("Subir archivo Excel 1", type=["xls","xlsx"])
archivo2 = st.file_uploader("Subir archivo Excel 2", type=["xls","xlsx"])

if archivo1 and archivo2:

    df1 = pd.read_excel(archivo1)
    df2 = pd.read_excel(archivo2)

    df = pd.concat([df1, df2], ignore_index=True)

    st.markdown("## 2. Comprensión de los Datos")

    st.subheader("Vista previa")
    st.dataframe(df.head())

    st.subheader("Información general")
    st.write(df.shape)
    st.write(df.dtypes)

    st.subheader("Valores nulos")
    st.write(df.isnull().sum())

    st.markdown("## 3. Visualización")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribución por sexo")
        fig, ax = plt.subplots()
        df["sexo"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Tipo de colegio")
        fig2, ax2 = plt.subplots()
        df["tipo_colegio"].value_counts().plot(kind="bar", ax=ax2)
        st.pyplot(fig2)

    st.subheader("Carreras con más estudiantes")
    fig3, ax3 = plt.subplots(figsize=(10,5))
    df["nombre_carrera"].value_counts().head(10).plot(kind="bar", ax=ax3)
    st.pyplot(fig3)

    st.markdown("## 4. Preparación de Datos")

    datos = df.copy()

    columnas = ["sexo","tipo_colegio","etnia","provincia_residencia","tipo_estudiante"]

    le = LabelEncoder()

    for col in columnas:
        datos[col] = datos[col].astype(str)
        datos[col] = le.fit_transform(datos[col])

    datos = datos.fillna(0)

    X = datos[["sexo","tipo_colegio","etnia","provincia_residencia"]]
    y = datos["tipo_estudiante"]

    st.write("Variables predictoras:", X.columns.tolist())

    st.markdown("## 5. Modelado")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    modelo = RandomForestClassifier()
    modelo.fit(X_train, y_train)

    pred = modelo.predict(X_test)

    st.markdown("## 6. Evaluación")

    acc = accuracy_score(y_test, pred)

    st.metric("Accuracy", round(acc,2))

    st.subheader("Matriz de Confusión")
    st.write(confusion_matrix(y_test, pred))

    st.success("Proyecto CRISP-DM ejecutado correctamente.")
