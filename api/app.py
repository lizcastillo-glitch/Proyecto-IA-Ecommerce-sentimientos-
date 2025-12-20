import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from deep_translator import GoogleTranslator
import torch.nn.functional as F

# Configuración de la página
st.set_page_config(page_title="IA Ecommerce Amazon Review Dashboard", page_icon="🛒")

st.title("🛒 IA Ecommerce Amazon Review Dashboard : Análisis de Sentimientos")
st.markdown("Prototipo de clasificación híbrida (Español -> Inglés -> IA).")

# --- 1. CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    # Asumimos que el modelo está en la carpeta 'modelos' dentro del contenedor
    model_path = "./modelos/sentimiento_xlmroberta_v1"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return tokenizer, model
    except OSError:
        st.error("⚠️ No se encontró el modelo. Asegúrate de haber descargado la carpeta 'modelos' de tu Drive y ponerla aquí.")
        return None, None

tokenizer, model = load_model()

# --- 2. LÓGICA DE INFERENCIA ---
def predecir(texto):
    if not tokenizer or not model:
        return None
    
    # A. Traducción (Capa Híbrida)
    try:
        traductor = GoogleTranslator(source='auto', target='en')
        texto_en = traductor.translate(texto)
    except Exception as e:
        st.error(f"Error de traducción: {e}")
        return None

    # B. Inferencia
    inputs = tokenizer(texto_en, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    
    probs = F.softmax(logits, dim=-1).squeeze().numpy()
    pred_idx = probs.argmax()
    
    # C. Regla de Umbral para Neutros
    top1 = probs[pred_idx]
    top2 = sorted(probs, reverse=True)[1]
    
    # Si la diferencia es menor a 0.22, es Neutro (Lógica del Notebook 3)
    if (top1 - top2) < 0.22:
        pred_idx = 1 # Forzar Neutro
        
    return pred_idx, probs, texto_en

# --- 3. INTERFAZ DE USUARIO ---
texto_input = st.text_area("Escribe una reseña del producto:", height=100)

if st.button("Analizar Sentimiento"):
    if texto_input:
        with st.spinner('Procesando...'):
            idx, probs, traducido = predecir(texto_input)
            
            if idx is not None:
                labels = {0: "Negativo 😡", 1: "Neutro 😐", 2: "Positivo 😃"}
                colors = {0: "red", 1: "orange", 2: "green"}
                
                st.subheader(f"Resultado: :{colors[idx]}[{labels[idx]}]")
                
                # Mostrar detalles técnicos (Caja expansible)
                with st.expander("Ver detalles técnicos (Traducción y Probabilidades)"):
                    st.write(f"**Traducción interna:** `{traducido}`")
                    st.write(f"**Prob. Negativo:** {probs[0]:.4f}")
                    st.write(f"**Prob. Neutro:** {probs[1]:.4f}")
                    st.write(f"**Prob. Positivo:** {probs[2]:.4f}")
                    
                st.progress(float(probs[idx]))
    else:
        st.warning("Por favor escribe algo.")
