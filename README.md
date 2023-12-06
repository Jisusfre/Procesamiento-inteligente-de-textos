# Procesamiento-inteligente-de-textos
Repositorio del proyecto final de la materia Procesamiento inteligente de textos.

_____________________Pasos para ejecutar el programa_____________________

1. Descargar anaconda del siguiente link: https://www.anaconda.com/download
2. Descargar la herramienta streamlit con el comando:
     pip install streamlit
3. Descargar la herramienta scikit con el comando:
     pip install scikit-learn
4. Posicionarse a la carpeta donde esté el main del programa y ejecutar en terminal
   los siguientes comandos:
     conda create -n streamlit -y
     conda activate streamlit
     streamlit run main.py
5. Ingresar una cadena de texto enfocada hacia un comentario de algún restaurante o negocio de comida, ya sea positivo, negativo o neutral y dar Enter.
6. El algoritmo procesará la cadena ingresada, después se mostrará por pantalla si el comentario fue positivo, negativo o neutral.
7. Podrá elegir si el algoritmo acertó en la predicción y poder subir el comentario para que el algoritmo “aprenda” de cada cadena ingresada, o bien, puede usted elegir que la predicción no fue acertada e ingresar el tipo de sentimiento correcto para que el algoritmo siga aprendiendo y añadiendo la cadena a su lista de entrenamiento.
