from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
#import yfinance as yf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
from PIL import Image as im
#Importar otras páginas

import Inicio

#Crear entorno virtual :
#   conda create -n streamlit -y

#Inicializar el entorno virtual:
#   conda activate streamlit

#Desactivar el entorno virtual
#conda deactivate

#Ejecutamos con:
#   streamlit run mi_programa.py [-- otros posibles argumentos]


# Datos de entrenamiento

#--------------------------------------------------------------------------

refreshed = st.session_state.get('refreshed', False)

st.session_state.refreshed = False

st.set_page_config(
    layout = "wide",
    page_icon = '🧠'
)
st.set_option('deprecation.showPyplotGlobalUse', False)

opiniones = np.load('array_opinions.npy')

etiquetas = np.load('array_labels.npy')
col1, col2, col3 = st.columns(3)
with col2:
    st.image('Imagenes\PortadaEditada.jpg', width = 500)
col1, col2, col3 = st.columns([1,3,1])
with col2:
    st.write("### **Jesús Bryan parada Pérez**")
    st.write("###         Luis Fernando Meza Argüello")
    st.write("###         Maria Fernanda Maya Ortega")

    opinion_nueva = st.text_input("Ingrese su opinion",
                                label_visibility="visible",
                                disabled=False, 
                                placeholder="Escribe aqui",
                            )

    # if opinion_nueva :
    #     st.session_state.refreshed = False
    
    if st.button('Opiniones', type="primary"):
        for x in range(0,len(opiniones)):
            st.write("\n\n-----------------------------------\n\n" + opiniones[x] + "\n\n" + etiquetas[x] + "\n\n-----------------------------------")



        
# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(opiniones, etiquetas, test_size=0.2, random_state=42)

# Crear un modelo de clasificación usando MLP (red neuronal)
model = make_pipeline(TfidfVectorizer(), MLPClassifier(hidden_layer_sizes=(100,), max_iter=300))

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Mostrar métricas de rendimiento
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nInforme de clasificación:")
print(classification_report(y_test, y_pred, zero_division=1))  # Agrega zero_division=1 para evitar las advertencias

# Predecir el sentimiento de nuevos textos
# --- aqui va el input ----
nuevos_textos = [opinion_nueva]

nueva_prediccion = model.predict(nuevos_textos)

print("\nNuevas predicciones:")
opinion_text = ""
prediccion_text = ""
for texto, prediccion in zip(nuevos_textos, nueva_prediccion):
    print(f"Texto: {texto}, Sentimiento: {prediccion}")
    opinion_text = texto
    prediccion_text = prediccion

labels = ["Positivo", "Negativo", "Neutral"]



with col2:

    if opinion_nueva and not refreshed:
        
        st.title("Texto: " + opinion_text + "\nSentimiento: " + prediccion_text)
        st.write("La predicción fue correcta?")

        option = st.radio('', ['Si', 'No'], index = None)
            
        if option == 'Si':
            
            if st.button("¿Añadir a la lista de entrenamiento?"):
                st.session_state.refreshed = True
                numpy_array_opinion = np.array(nuevos_textos)
                opiniones = np.append(opiniones,numpy_array_opinion)
                np.save("array_opinions.npy",opiniones)
                etiquetas = np.append(etiquetas, prediccion_text)
                np.save("array_labels.npy",etiquetas)
                st.rerun()

        elif option == 'No':
            label = st.radio('Ingrese la predicción correcta', labels, index=None)
            
            if label == "Positivo":
                st.write("Positivo será")
                st.session_state.refreshed = True
                numpy_array_opinion = np.array(nuevos_textos)
                opiniones = np.append(opiniones,numpy_array_opinion)
                np.save("array_opinions.npy",opiniones)
                etiquetas = np.append(etiquetas, "positivo")
                np.save("array_labels.npy",etiquetas)
                st.rerun()
            elif label == "Neutral":
                st.write("Neutral será")
                st.session_state.refreshed = True
                numpy_array_opinion = np.array(nuevos_textos)
                opiniones = np.append(opiniones,numpy_array_opinion)
                np.save("array_opinions.npy",opiniones)
                etiquetas = np.append(etiquetas, "neutral")
                np.save("array_labels.npy",etiquetas)
                st.rerun()
            if label == "Negativo":
                st.write("Negativo será")
                st.session_state.refreshed = True
                numpy_array_opinion = np.array(nuevos_textos)
                opiniones = np.append(opiniones,numpy_array_opinion)
                np.save("array_opinions.npy",opiniones)
                etiquetas = np.append(etiquetas, "negativo")
                np.save("array_labels.npy",etiquetas)
                st.rerun()
