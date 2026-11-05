import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

st.title("🐾 Clasificador de Perros y Gatos")
st.write("Proyecto parcial de Programación - ENES Unidad León")
st.write("Sube una imagen de un perro o un gato para que la red neuronal profunda prediga el resultado.")

@st.cache_resource
def load_my_model():
    model_path = 'modelo_perros_gatos.keras'
    
    if not os.path.exists(model_path):
        file_id = '1QuKK4j7nHIBGhX7m5yG9S5s3Uq0zx0Rs' 
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)
        
    model = tf.keras.models.load_model(model_path)
    return model

with st.spinner('Cargando el modelo de inteligencia artificial (esto puede tomar unos segundos)...'):
    model = load_my_model()

uploaded_file = st.file_uploader("Elige una imagen (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen cargada por el usuario')
    
    st.write("Analizando la imagen con la red neuronal...")
    
    image = image.resize((150, 150))
    img_array = np.array(image) / 255.0  
    img_array = np.expand_dims(img_array, axis=0) 
    
    # Realizar la predicción
    prediction = model.predict(img_array)[0][0]
    
    if prediction > 0.5:
        st.success(f"🐶 ¡Es un **Perro**! (Confianza: {prediction * 100:.2f}%)")
    else:
        st.error(f"🐱 ¡Es un **Gato**! (Confianza: {(1 - prediction) * 100:.2f}%)")