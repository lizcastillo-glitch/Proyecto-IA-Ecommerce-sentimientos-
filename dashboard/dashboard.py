import streamlit as st
import sys
import os

# Configuración de página
st.set_page_config(page_title="IA Ecommerce Amazon Review Dashboard", page_icon="🛒", layout="centered")

# Agregamos la carpeta raíz al path para importar src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference import EcoSentModel

# Cargar Modelo (con caché para no recargar en cada click)
@st.cache_resource
def get_model():
    model_path = os.path.join(os.path.dirname(__file__), "../modelos/sentimiento_xlmroberta_v1")
    return EcoSentModel(model_path)

model = get_model()

# Interfaz
st.title("🛒IA Ecommerce Amazon Review Dashboard")
st.markdown("---")
st.markdown("**Monitor de Sentimientos en Tiempo Real (Prototipo)**")

texto = st.text_area("Ingresa la reseña del cliente (Español):", height=100, placeholder="Ej: El envío fue rápido pero el producto llegó golpeado.")

if st.button("Analizar"):
    if texto:
        with st.spinner("Traduciendo y analizando..."):
            resultado = model.predict(texto)
        
        if "error" in resultado:
            st.error(resultado["error"])
        else:
            sentimiento = resultado["sentimiento"]
            color = "green" if sentimiento == "Positivo" else "red" if sentimiento == "Negativo" else "orange"
            
            st.markdown(f"### Sentimiento Detectado: :{color}[{sentimiento}]")
            st.progress(resultado["score_confianza"])
            
            with st.expander("🔍 Detalles Técnicos (Backend)"):
                st.json(resultado)
    else:
        st.warning("Escribe algo para analizar.")
