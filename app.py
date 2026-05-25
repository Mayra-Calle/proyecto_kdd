import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
st.title("Metodologias para el proceso de mineria de datos")
st.write("PRACTIA 1")
#Dataframe de productos
data = {
    'Producto': ['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Smartwatch'],
    'Precio': [1000, 500, 300, 150, 200],
    'Stock': [10, 20, 15, 5, 10]
}
df=pd.DataFrame(data)
st.subheader("Dataframe de productos")
st.dataframe(df)
st.write("Graficas")
st.header("Grafica de precio de los productos")
fig, ax = plt.subplots()
ax.bar(df['Producto'], df['Precio'], color='blue')
ax.set_xlabel('Producto')
ax.set_ylabel('Precio')
ax.set_title('Precio de los productos')
st.pyplot(fig)
st.header("Grafica de cantidad de los productos")
fig, ax = plt.subplots()
ax.bar(df['Producto'], df['Stock'], color='orange')
ax.set_xlabel('Producto')   
ax.set_ylabel('Stock')
ax.set_title('Cantidad de los productos en stock')
st.pyplot(fig)

