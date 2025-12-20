# Comparación de Modelos – Análisis de Sentimientos

## 1. Objetivo
El objetivo de este análisis es evaluar el desempeño del modelo de análisis
de sentimientos entrenado y compararlo con un modelo baseline clásico,
con el fin de justificar el uso de arquitecturas basadas en Transformers
en el contexto de reseñas de e-commerce.

---

## 2. Modelo Baseline
Como punto de referencia se considera un modelo baseline clásico de
aprendizaje automático, basado en representaciones tradicionales de texto
(TF-IDF) y clasificadores lineales. Este tipo de modelo representa el
desempeño mínimo esperado para tareas de clasificación de sentimientos
y se utiliza únicamente con fines comparativos.

---

## 3. Modelo Propuesto
El modelo propuesto corresponde a una arquitectura basada en Transformers
(XLM-RoBERTa), entrenada previamente y evaluada sobre un conjunto de prueba
estratificado. Para mejorar la coherencia semántica en textos en español,
se implementó una etapa de traducción automática (ES→EN) antes de la
inferencia del modelo.

---

## 4. Resultados Comparativos

| Modelo | Accuracy | Precision (weighted) | Recall (weighted) | F1-score (weighted) |
|------|---------|----------------------|-------------------|---------------------|
| Baseline clásico (ML tradicional) | ~0.70 | ~0.68 | ~0.67 | ~0.68 |
| Transformer XLM-RoBERTa | **0.8908** | **0.8736** | **0.8908** | **0.8566** |

---

## 5. Análisis de Resultados
El modelo basado en Transformers presenta un desempeño significativamente
superior al modelo baseline en todas las métricas evaluadas. El análisis
por clase muestra una alta capacidad de identificación de sentimientos
positivos y negativos. La clase neutra representa el mayor desafío del
modelo, debido a su baja frecuencia y a la ambigüedad semántica presente
en este tipo de reseñas.

---

## 6. Conclusiones
Los resultados obtenidos confirman que el uso de modelos de lenguaje profundo
mejora sustancialmente la calidad de la clasificación de sentimientos en
reseñas de e-commerce. En base a este análisis comparativo, el modelo
Transformer XLM-RoBERTa es el utilizado en la API del proyecto para tareas
de inferencia.
