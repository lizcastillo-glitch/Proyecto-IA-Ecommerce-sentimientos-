# Comparación de Modelos – Análisis de Sentimientos

## 1. Objetivo del Análisis Comparativo
El objetivo de este análisis comparativo es evaluar el desempeño de un
modelo baseline clásico frente a un modelo basado en Transformers para
la tarea de análisis de sentimientos en reseñas de productos.

La comparación permite cuantificar las mejoras obtenidas al utilizar
arquitecturas de aprendizaje profundo y justificar técnicamente la
elección del modelo final que se expone mediante una API de inferencia.

Este análisis se basa directamente en los resultados obtenidos en el
Notebook 3: *Evaluación y comparación de resultados*.

---

## 2. Modelo Baseline

### 2.1 Descripción del Modelo Baseline
El modelo baseline corresponde a un enfoque clásico de clasificación de
texto, compuesto por una etapa de vectorización TF-IDF y un clasificador
de Regresión Logística.

Este tipo de modelo se utiliza como referencia inicial debido a su
simplicidad, bajo costo computacional y facilidad de interpretación,
siendo ampliamente empleado en tareas tradicionales de análisis de
sentimientos.

### 2.2 Componentes del Baseline
- Limpieza y preprocesamiento básico del texto.
- Vectorización mediante TF-IDF.
- Clasificador lineal: Regresión Logística.
- Entrenamiento supervisado con etiquetas de sentimiento.

El modelo baseline fue implementado y evaluado como punto de comparación
en el proceso de análisis.

---

## 3. Modelo Basado en Transformers

### 3.1 Descripción del Modelo Transformer
El modelo avanzado corresponde a un Transformer preentrenado
`xlm-roberta-base`, ajustado mediante fine-tuning para la tarea de
clasificación de sentimientos.

Este modelo permite capturar relaciones semánticas complejas y contexto
de largo alcance, superando las limitaciones de los enfoques lineales
tradicionales.

### 3.2 Características Clave
- Arquitectura Transformer multilingüe.
- Fine-tuning supervisado sobre reseñas etiquetadas.
- Capacidad de manejar ambigüedad semántica.
- Integración de traducción automática (ES → EN) antes de la inferencia
  para mejorar la coherencia de las predicciones en español.
- Preparado para ser consumido mediante una API de inferencia.

---

## 4. Métricas de Evaluación
La comparación entre modelos se realizó utilizando métricas estándar de
clasificación multiclase, calculadas sobre un conjunto de prueba no
visto:

- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)

Las métricas fueron calculadas utilizando la librería
`sklearn.metrics`, tal como se implementa en el Notebook 3.

---

## 5. Resultados del Modelo Transformer
A continuación, se presentan los resultados obtenidos por el modelo
basado en Transformers sobre el conjunto de prueba:

- **Accuracy:** 0.8857
- **Precision (weighted):** 0.8347
- **Recall (weighted):** 0.8857
- **F1-score (weighted):** 0.8333

La matriz de confusión muestra que la mayoría de las predicciones se
concentran correctamente en las clases positiva y negativa, mientras
que la clase neutra presenta una mayor dificultad de identificación,
lo cual es consistente con la naturaleza ambigua de este tipo de
sentimientos.

---

## 6. Resultados Comparativos

| Modelo                          | Accuracy | Precision (w) | Recall (w) | F1-score (w) |
|---------------------------------|----------|----------------|------------|---------------|
| Baseline (TF-IDF + Reg. Log.)   | Menor    | Menor          | Menor      | Menor         |
| Transformer (XLM-RoBERTa)       | 0.8857   | 0.8347         | 0.8857     | 0.8333        |

> Los resultados evidencian que el modelo Transformer supera de forma
> consistente al modelo baseline en todas las métricas evaluadas, tal
> como se observa en el Notebook 3.

---

## 7. Análisis Comparativo de Resultados
El modelo baseline presenta un desempeño aceptable en textos simples,
pero muestra limitaciones al enfrentar reseñas con ambigüedad semántica
o dependencias de contexto más complejas.

En contraste, el modelo basado en Transformers logra una representación
más rica del lenguaje, permitiendo una clasificación más precisa y
robusta. El análisis de errores evidencia que muchas reseñas
“intermedias” o sin una emoción fuerte tienden a confundirse con clases
positivas o negativas, un comportamiento esperado dada la dificultad
intrínseca de la clase neutra.

La incorporación de una etapa de traducción automática previa a la
inferencia contribuyó a mejorar la coherencia de las predicciones en
textos originalmente escritos en español.

---

## 8. Conclusiones del Análisis Comparativo
El análisis comparativo valida que los modelos basados en Transformers
superan de manera consistente a los enfoques clásicos de aprendizaje
automático en tareas de análisis de sentimientos.

Estos resultados justifican técnicamente la selección del modelo
`xlm-roberta-base` como solución final del proyecto y respaldan su
integración en una arquitectura basada en API para inferencia en tiempo
real.
