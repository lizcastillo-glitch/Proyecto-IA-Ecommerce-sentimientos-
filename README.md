# Plataforma de Análisis de Sentimientos en tiempo real para E-Commerce basada en técnicas de procesamiento del lenguaje natural y modelos Transformers. 🌎🛒

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Hugging Face](https://img.shields.io/badge/🤗%20Transformers-XLM--RoBERTa-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![License](https://img.shields.io/badge/License-MIT-green)

La presente plataforma de análisis de sentimientos desarrollada como prototipo funcional para la Maestría en Inteligencia de Negocios y Ciencia de Datos de la Universidad Espíritu Santo.

El proyecto implementa una **Arquitectura Híbrida (Cross-Lingual)** innovadora: entrena un modelo Transformer de última generación (`xlm-roberta-base`) utilizando datasets masivos en **Inglés** (Amazon Reviews), pero permite realizar inferencias y clasificaciones en **Español** mediante una capa de traducción en tiempo real.

---

## 🚀 Características del Proyecto

* **Modelo SOTA:** Implementación de `XLM-RoBERTa`, un modelo optimizado para tareas multilingües.
* **Entrenamiento Robusto:** Fine-tuning realizado con +20,000 reseñas reales de productos.
* **Inferencia Híbrida:** Capacidad de recibir texto en español, traducirlo internamente y clasificarlo con el motor analítico entrenado en inglés.
* **Alta Precisión:** **Accuracy del 89.08%** validado en el conjunto de prueba.
* **Manejo de Incertidumbre:** Lógica de umbral para detectar reseñas "Neutras" en un entorno de datos polarizados.

---

## 📂 Contenido del Repositorio
El laboratorio consta de 3 notebooks principales ubicados en la carpeta `Notebooks`:

📘 01_EDA_Limpieza.ipynb:[Eda_y_limpieza](Notebooks/01_Eda_y_limpieza.ipynb)

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

