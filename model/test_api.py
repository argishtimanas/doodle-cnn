from pathlib import Path
import numpy as np
import requests

DATA_DIR = Path("../data/processed")

X_test = np.load(DATA_DIR / "X_test.npy")

image = X_test[0]
pixels = image.reshape(-1).tolist()

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"pixels": pixels},
)

print(response.json())