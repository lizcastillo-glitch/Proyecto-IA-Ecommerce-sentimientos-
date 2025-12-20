import os
from fastapi import FastAPI, HTTPException
from schemas import PredictRequest, PredictResponse
from inference import SentimentModel

MODEL_DIR = os.getenv(
    "MODEL_DIR",
    "/content/drive/MyDrive/proyecto_ia/modelos/sentimiento_xlmroberta_v1"
)

app = FastAPI(title="Sentiment API - Proyecto IA")

model_service = None

@app.on_event("startup")
def load_model():
    global model_service
    if not os.path.exists(MODEL_DIR):
        raise RuntimeError(f"No se encontró el modelo en: {MODEL_DIR}")
    model_service = SentimentModel(MODEL_DIR)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model_service is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    return model_service.predict(req.text)
