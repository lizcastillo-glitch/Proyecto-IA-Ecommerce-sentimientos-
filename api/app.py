# api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Traducción opcional (capa híbrida)
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except Exception:
    TRANSLATOR_AVAILABLE = False
# -------------------------------------------------
# Inicialización de la API
# -------------------------------------------------
app = FastAPI(
    title="API de Análisis de Sentimientos",
    description="Clasificación de sentimientos en reseñas de E-commerce usando Transformer",
    version="1.0.0"
)
# -------------------------------------------------
# Configuración del modelo
# -------------------------------------------------
MODEL_PATH = "./models/sentimiento_xlmroberta_v1"  # carpeta del modelo
LABELS = {0: "Negativo", 1: "Neutro", 2: "Positivo"}
NEUTRAL_THRESHOLD = 0.22
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = None
model = None
# -------------------------------------------------
# Esquemas de entrada y salida
# -------------------------------------------------
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Texto a analizar")
    translate: bool = Field(
        default=True,
        description="Traduce el texto a inglés antes de la inferencia"
    )


class SentimentResponse(BaseModel):
    label: str
    label_id: int
    confidence: float
    probabilities: Dict[str, float]
    translated_text: Optional[str]
    device: str
# -------------------------------------------------
# Carga del modelo al iniciar la API
# -------------------------------------------------
@app.on_event("startup")
def load_model():
    global tokenizer, model
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
    except Exception as e:
        print("Error cargando el modelo:", e)
        tokenizer, model = None, None
# -------------------------------------------------
# Funciones auxiliares
# -------------------------------------------------
def translate_text(text: str) -> str:
    if not TRANSLATOR_AVAILABLE:
        raise RuntimeError("deep-translator no está disponible")
    translator = GoogleTranslator(source="auto", target="en")
    return translator.translate(text)
def predict_sentiment(text: str, translate: bool = True):
    if tokenizer is None or model is None:
        raise RuntimeError("Modelo no cargado correctamente")

    translated_text = None
    model_input = text

    # Traducción (capa híbrida)
    if translate:
        translated_text = translate_text(text)
        model_input = translated_text

    inputs = tokenizer(
        model_input,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
    pred_idx = int(probs.argmax())

    # Regla de neutralidad
    sorted_probs = sorted(probs, reverse=True)
    if (sorted_probs[0] - sorted_probs[1]) < NEUTRAL_THRESHOLD:
        pred_idx = 1  # Neutro
    return pred_idx, probs, translated_text
# -------------------------------------------------
# Endpoints
# -------------------------------------------------
@app.get("/health")
def health_check():
    return {
        "status": "ok" if model is not None else "error",
        "model_loaded": model is not None,
        "device": device
    }


@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest):
    try:
        pred_idx, probs, translated = predict_sentiment(
            request.text, request.translate
        )

        return {
            "label": LABELS[pred_idx],
            "label_id": pred_idx,
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                "Negativo": float(probs[0]),
                "Neutro": float(probs[1]),
                "Positivo": float(probs[2])
            },
            "translated_text": translated,
            "device": device
        }

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inesperado: {e}")
