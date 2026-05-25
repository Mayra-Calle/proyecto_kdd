import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(
    page_title="Proyecto SEMMA",
    layout="wide"
)

st.title("Proyecto SEMMA con Datos de Estudiantes")

st.write("""
Metodología SEMMA:
- S = Sample (Muestreo)
- E = Explore (Exploración)
- M = Modify (Modificación)
- M = Model (Modelado)
- A = Assess (Evaluación)
""")

st.header("FASE 1: SAMPLE (MUESTREO)")

st.write("""
En esta fase se recopilan y seleccionan los datos
que serán utilizados en el análisis.
""")

archivo1 = st.file_uploader(
    "Subir archivo Excel 1",
    type=["xls", "xlsx"]
)

archivo2 = st.file_uploader(
    "Subir archivo Excel 2",
    type=["xls", "xlsx"]
)

if archivo1 and archivo2:

    # Leer archivos
    df1 = pd.read_excel(archivo1)
    df2 = pd.read_excel(archivo2)

    # Unir datasets
    df = pd.concat([df1, df2], ignore_index=True)

    st.success("Archivos cargados correctamente")

    st.subheader("Cantidad de registros y columnas")
    st.write(df.shape)

    st.subheader("Muestra de datos")
    st.dataframe(df.head())


    st.header("FASE 2: EXPLORE (EXPLORACIÓN)")

    st.write("""
    En esta fase se analizan los datos para identificar:
    - patrones
    - valores nulos
    - distribuciones
    - relaciones entre variables
    """)

    st.subheader("Tipos de datos")
    st.write(df.dtypes)

    st.subheader("Valores nulos")
    st.write(df.isnull().sum())

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Distribución por sexo")

        fig, ax = plt.subplots()

        df["sexo"].value_counts().plot(
            kind="bar",
            ax=ax
        )

        st.pyplot(fig)

    with col2:

        st.subheader("Tipo de colegio")

        fig2, ax2 = plt.subplots()

        df["tipo_colegio"].value_counts().plot(
            kind="bar",
            ax=ax2
        )

        st.pyplot(fig2)

    st.subheader("Carreras con más estudiantes")

    fig3, ax3 = plt.subplots(figsize=(10, 5))

    df["nombre_carrera"].value_counts().head(10).plot(
        kind="bar",
        ax=ax3
    )

    st.pyplot(fig3)
    st.header("FASE 3: MODIFY (MODIFICACIÓN)")

    st.write("""
    En esta fase se preparan los datos para el modelo:
    - limpieza
    - codificación
    - transformación
    """)

    datos = df.copy()

    columnas = [
        "sexo",
        "tipo_colegio",
        "etnia",
        "provincia_residencia",
        "tipo_estudiante"
    ]

    le = LabelEncoder()

    for col in columnas:

        datos[col] = datos[col].astype(str)

        datos[col] = le.fit_transform(datos[col])

    datos = datos.fillna(0)

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

    st.subheader("Variables predictoras")
    st.write(X.columns.tolist())

    st.header("FASE 4: MODEL (MODELADO)")

    st.write("""
    En esta fase se construye el modelo predictivo
    utilizando Machine Learning.
    """)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    modelo = RandomForestClassifier(
        random_state=42
    )

    modelo.fit(X_train, y_train)

    pred = modelo.predict(X_test)

    st.success("Modelo entrenado correctamente")

    st.header("FASE 5: ASSESS (EVALUACIÓN)")

    st.write("""
    En esta fase se evalúa el rendimiento del modelo.
    """)

    acc = accuracy_score(y_test, pred)

    st.metric(
        "Accuracy del modelo",
        round(acc, 2)
    )

    st.subheader("Matriz de Confusión")

    matriz = confusion_matrix(y_test, pred)

    st.write(matriz)

    st.success("Proyecto SEMMA ejecutado correctamente")