# Análisis de Sentimientos Híbrido para E-Commerce 🌎🛒

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Hugging Face](https://img.shields.io/badge/🤗%20Transformers-XLM--RoBERTa-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![License](https://img.shields.io/badge/License-MIT-green)

es una plataforma de análisis de sentimientos desarrollada como prototipo funcional para la Maestría en Inteligencia de Negocios y Ciencia de Datos (UEES).

El proyecto implementa una **Arquitectura Híbrida (Cross-Lingual)** innovadora: entrena un modelo Transformer de última generación (`xlm-roberta-base`) utilizando datasets masivos en **Inglés** (Amazon Reviews), pero permite realizar inferencias y clasificaciones en **Español** mediante una capa de traducción en tiempo real.

---

## 🚀 Características del Proyecto

* **Modelo SOTA:** Implementación de `XLM-RoBERTa`, un modelo optimizado para tareas multilingües.
* **Entrenamiento Robusto:** Fine-tuning realizado con +20,000 reseñas reales de productos.
* **Inferencia Híbrida:** Capacidad de recibir texto en español, traducirlo internamente y clasificarlo con el motor analítico entrenado en inglés.
* **Alta Precisión:** **Accuracy del 89.08%** validado en el conjunto de prueba.
* **Manejo de Incertidumbre:** Lógica de umbral para detectar reseñas "Neutras" en un entorno de datos polarizados.

---

## 🛠️ Arquitectura Técnica

El flujo de datos diseñado para este prototipo maximiza el uso de recursos open-source disponibles:

```mermaid
graph LR
    A[Usuario (Input Español)] -->|'El envío demoró mucho'| B(Capa de Traducción)
    B -->|'Shipping took too long'| C{Modelo XLM-RoBERTa}
    C -->|Análisis de Atención| D[Clasificación Softmax]
    D -->|Resultado Final| E[Negativo 😡]
Ingesta: El usuario ingresa una reseña en español.

Pre-procesamiento: Normalización y traducción automática (ES -> EN) usando deep-translator.

Inferencia: El modelo predice la polaridad (Positivo, Neutro, Negativo).

Post-procesamiento: Aplicación de reglas de negocio para refinar la clase neutra.

📂 Contenido del Repositorio
Este repositorio contiene los 3 Notebooks que componen el pipeline completo de ML:

📘 01_EDA_Limpieza.ipynb:

Ingesta del dataset Amazon Product Reviews.

Limpieza de texto con Expresiones Regulares (Regex).

Estratificación de datos (Train/Test Split).

📙 02_Entrenamiento.ipynb:

Configuración del Tokenizador AutoTokenizer.

Entrenamiento con la API Trainer de Hugging Face (GPU T4).

Persistencia del modelo entrenado.

📗 03_Evaluacion_Inferencia.ipynb:

Evaluación de métricas (Matriz de Confusión, F1-Score).

Función de predicción final para consumo del modelo con traducción integrada.

💻 Instalación y Uso
Este proyecto está diseñado para ejecutarse en Google Colab. Si deseas correrlo localmente:

Clonar el repositorio:

Bash

git clone [https://github.com/tu-usuario/EcoSent-IA.git](https://github.com/tu-usuario/EcoSent-IA.git)
cd EcoSent-IA
Instalar dependencias:

Bash

pip install torch transformers accelerate datasets scikit-learn pandas deep-translator emoji
Ejecutar inferencia (Ejemplo en Python):

Python

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from deep_translator import GoogleTranslator
import torch

# Cargar modelo (asegúrate de tener la carpeta del modelo descargada)
modelo_path = "./modelos/sentimiento_xlmroberta_v1"
tokenizer = AutoTokenizer.from_pretrained(modelo_path)
model = AutoModelForSequenceClassification.from_pretrained(modelo_path)

def analizar_sentimiento(texto):
    # Capa de traducción Híbrida
    traductor = GoogleTranslator(source='es', target='en')
    texto_en = traductor.translate(texto)

    # Inferencia
    inputs = tokenizer(texto_en, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits.argmax(-1).item()

print(analizar_sentimiento("¡Me encantó el producto, llegó rapidísimo!")) 
# Resultado esperado: 2 (Positivo)
