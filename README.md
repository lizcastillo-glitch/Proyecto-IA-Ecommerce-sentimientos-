# 🛒 Análisis de Sentimientos en Reseñas de Amazon con XLM-Roberta

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Transformers-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Status](https://img.shields.io/badge/Status-Prototipo-yellow)

Este proyecto implementa un modelo de **Procesamiento de Lenguaje Natural (NLP)** capaz de clasificar reseñas de productos de Amazon en tres categorías: **Positivo, Neutro y Negativo**. 

Utiliza el modelo pre-entrenado **XLM-Roberta** y una estrategia de traducción para permitir inferencia multilingüe (Español/Inglés).

## 🗂️ Estructura del Proyecto

El flujo de trabajo se divide en 4 etapas principales (Notebooks):

**Lenguaje y Entorno:**
* Python 3.10+: Lenguaje base para todo el procesamiento.
* Google Colab: Entorno de ejecución en la nube con aceleración por hardware (GPU T4) para el entrenamiento del Transformer.

El laboratorio consta de 3 notebooks principales ubicados en la carpeta `Notebooks`:

## 📘 [01_Eda_y_limpieza](Notebooks/01_Eda_y_limpieza.ipynb)

* **Ingesta del dataset Amazon Product Reviews.**

* **Limpieza de texto con Expresiones Regulares (Regex).**

* **Estratificación de datos (Train/Test Split).**

## 📙 [02 Entranamiento](Notebooks/02_Entrenamiento_modelo.ipynb)

* **Configuración del Tokenizador AutoTokenizer.**

* **Entrenamiento con la API Trainer de Hugging Face (GPU T4).**

* **Persistencia del modelo entrenado.**

## 📗 [03 Evaluación](03_Evaluacion_comparacion.ipynb)

* **Evaluación de métricas (Matriz de Confusión, F1-Score).**

* **Función de predicción final para consumo del modelo con traducción integrada.**

## 📊 Dataset

Se utilizó el conjunto de datos **Amazon Product Reviews** disponible en Kaggle.
* **Total de muestras:** ~21,000 reseñas.
* **Clases:** * `Positive` (2): ~18,800
    * `Negative` (0): ~2,100
    * `Neutral` (1): ~300

> ⚠️ **Nota:** El dataset presenta un fuerte desbalance de clases, predominando masivamente las reseñas positivas.

## 🛠️ Tecnologías Utilizadas

* **Python** (Entorno Google Colab)
* **Transformers (Hugging Face):** Para el modelo XLM-Roberta y Tokenizer.
* **PyTorch:** Backend de Deep Learning.
* **Scikit-Learn:** Para métricas y división de datos.
* **Deep-Translator:** Para pipeline de traducción en inferencia (ES -> EN).
* **Pandas & Matplotlib:** Manipulación y visualización de datos.

## 🚀 Instalación y Uso

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/tu-usuario/amazon-sentiment-analysis.git](https://github.com/tu-usuario/amazon-sentiment-analysis.git)
   cd amazon-sentiment-analysis

## 👥 Autores - Grupo 3

* Liz Eliana Castillo Zamora

* Pablo Mauricio Castro Hinostroza

* Erick Sebastián Rivas

* Ángel Israel Romero Medina

**Proyecto académico para la asignatura de Inteligencia Artificial - UEES.**
