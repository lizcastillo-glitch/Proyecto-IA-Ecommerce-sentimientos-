import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from deep_translator import GoogleTranslator
import torch.nn.functional as F
import os

class EcoSentModel:
    def __init__(self, model_path):
        """
        Carga el modelo y el tokenizador al iniciar la clase.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Cargando modelo desde {model_path} en {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval() # Modo evaluación
            print("✅ Modelo cargado exitosamente.")
        except Exception as e:
            print(f"❌ Error cargando el modelo: {e}")
            self.tokenizer = None
            self.model = None

    def predict(self, text_es):
        """
        Recibe texto en español, traduce e infiere.
        """
        if not self.model or not self.tokenizer:
            return {"error": "Modelo no cargado"}

        # 1. Traducción (Híbrida)
        try:
            translator = GoogleTranslator(source='auto', target='en')
            text_en = translator.translate(text_es)
        except Exception as e:
            return {"error": f"Fallo en traducción: {str(e)}"}

        # 2. Preprocesamiento
        inputs = self.tokenizer(
            text_en, 
            return_tensors="pt", 
            truncation=True, 
            max_length=128
        ).to(self.device)

        # 3. Inferencia
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
        pred_idx = probs.argmax()

        # 4. Regla de Neutros (Umbral 0.22)
        top1 = probs[pred_idx]
        top2 = sorted(probs, reverse=True)[1]
        
        if (top1 - top2) < 0.22:
            pred_idx = 1 # Forzar Neutro

        labels = {0: "Negativo", 1: "Neutro", 2: "Positivo"}
        
        return {
            "sentimiento": labels[pred_idx],
            "score_confianza": float(probs[pred_idx]),
            "traduccion_usada": text_en,
            "probabilidades": {
                "negativo": float(probs[0]),
                "neutro": float(probs[1]),
                "positivo": float(probs[2])
            }
        }
