# Análisis de Sentimientos para E-Commerce 🌎🛒

Esta es una plataforma de análisis de sentimientos diseñada para resolver el problema de la escasez de datos etiquetados en español para el comercio electrónico. Utiliza una Arquitectura Híbrida que aprovecha modelos Transformers entrenados con datos globales (Inglés) para realizar inferencias precisas en el mercado local (Español).

## 🚀 Características Principales

Modelo SOTA: Utiliza xlm-roberta-base (multilingüe) ajustado con Transfer Learning.

Estrategia Cross-Lingual: Entrenado con +20,000 reseñas de Amazon en Inglés, pero capaz de procesar español mediante una capa de traducción en tiempo real.

Alta Precisión: Accuracy del 89.08% validado en el conjunto de prueba.

Detección de Matices: Clasificación en 3 clases (Positivo, Neutro, Negativo) con lógica de umbral para manejar la incertidumbre.

🛠️ Arquitectura del Sistema

El proyecto sigue un flujo de datos híbrido para maximizar la calidad del análisis sin requerir un dataset masivo en español:

graph LR
    A[Usuario (Español)] -->|Texto: 'Llegó roto'| B(Traductor ES->EN)
    B -->|Texto: 'Arrived broken'| C{Modelo XLM-RoBERTa}
    C -->|Logits| D[Clasificación]
    D -->|Resultado| E[Negativo 😡]


Ingesta: Entrada de texto en Español.

Adaptación: Traducción automática al inglés usando deep-translator.

Inferencia: El modelo Transformer (fine-tuned) procesa el texto en inglés.

Salida: Etiqueta de sentimiento final.

📂 Estructura del Proyecto

El repositorio está organizado en 3 Notebooks principales que cubren el ciclo de vida del ML:

01_EDA_Limpieza.ipynb:

Ingesta del dataset Amazon Product Reviews (Inglés).

Limpieza de texto (Regex) y normalización.

División estratificada (80/20) para manejar el desbalance de clases.

02_Entrenamiento.ipynb:

Tokenización con AutoTokenizer (XLM-R).

Fine-tuning usando la API Trainer de Hugging Face.

Persistencia del modelo y tokenizador.

03_Evaluacion_Inferencia.ipynb:

Cálculo de métricas (Matriz de Confusión, F1-Score).

Implementación de la función predecir_sentimiento() con traducción integrada.

💻 Instalación y Requisitos

Este proyecto fue desarrollado en Google Colab. Para ejecutarlo localmente, necesitas las siguientes dependencias:

pip install torch transformers accelerate datasets scikit-learn pandas deep-translator emoji


🤖 Ejemplo de Uso (Inferencia)

Una vez cargado el modelo entrenado, puedes realizar predicciones en español así:

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from deep_translator import GoogleTranslator
import torch

# 1. Cargar Modelo
MODEL_PATH = "./modelos/sentimiento_xlmroberta_v1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# 2. Función de Predicción Híbrida
def predecir(texto_espanol):
    # Traducir
    traductor = GoogleTranslator(source='es', target='en')
    texto_en = traductor.translate(texto_espanol)
    
    # Tokenizar e Inferir
    inputs = tokenizer(texto_en, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Post-procesamiento
    pred_idx = logits.argmax(-1).item()
    etiquetas = {0: "Negativo 🔴", 1: "Neutro 🟡", 2: "Positivo 🟢"}
    
    return etiquetas[pred_idx]

# 3. Prueba
print(predecir("El producto es excelente, llegó muy rápido."))
# Salida: Positivo 🟢


📊 Resultados Obtenidos

Métrica

Valor

Descripción

Accuracy

89.08%

Exactitud global del modelo.

F1-Score

0.8456

Promedio ponderado (Weighted).

Loss

< 0.40

Convergencia estable en 1 época.

Nota: Se observó un desafío en la detección de la clase "Neutra" debido al desbalance del dataset original (<2% de muestras neutras). Se recomienda usar un umbral de confianza para mejorar esto en producción.

👥 Autores (Grupo 3)

Proyecto desarrollado para la Maestría en Inteligencia de Negocios y Ciencia de Datos - UEES.

Liz Eliana Castillo Zamora

Pablo Mauricio Castro Hinostroza

Erick Sebastián Rivas

Ángel Israel Romero Medina

Made with ❤️  by Group 3.
