from pathlib import Path
from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

CLASSES = ["cat", "dog", "fish", "bird", "rabbit", "bear", "lion", "tiger", "elephant", "horse"]

DATA_DIR = Path("../data/processed")
MODEL_PATH = Path("../saved_model/doodlenet.keras")

model = tf.keras.models.load_model(MODEL_PATH)

X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

logits = model.predict(X_test, batch_size=128)

y_pred = np.argmax(logits, axis=1)

cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASSES,
)

display.plot(
    xticks_rotation=45,
    values_format="d",
)

plt.title("DoodleNet Confusion Matrix")
plt.tight_layout()
plt.show()