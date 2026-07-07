from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

CLASSES = ["cat", "dog", "fish", "bird", "rabbit", "bear", "lion", "tiger", "elephant", "horse"]

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "saved_model" / "doodlenet.keras"

model = tf.keras.models.load_model(MODEL_PATH)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DrawingRequest(BaseModel):
    pixels: list[float]

@app.get("/")
def home():
    return {
        "message": "DoodleNet API is running",
        "classes": CLASSES,
        "model_loaded": True,
    }

@app.post("/predict")
def predict(request: DrawingRequest):
    pixels = np.array(request.pixels, dtype=np.float32)

    if pixels.size != 784:
        return {"error": f"Expected 784 pixels, received {pixels.size}"}

    image = pixels.reshape(1, 28, 28, 1)

    logits = model.predict(image, verbose=0)[0]
    probabilities = tf.nn.softmax(logits).numpy()

    predicted_id = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_id])

    return {
        "prediction": CLASSES[predicted_id],
        "confidence": confidence,
        "probabilities": {
            CLASSES[i]: float(probabilities[i])
            for i in range(len(CLASSES))
        },
    }