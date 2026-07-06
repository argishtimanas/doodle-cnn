from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

CLASSES = ["cat", "dog", "fish", "bird", "rabbit", "bear", "lion", "tiger", "elephant", "horse"]

DATA_DIR = Path("../data/processed")
MODEL_PATH = Path("../saved_model/doodlenet.keras")

model = tf.keras.models.load_model(MODEL_PATH)

X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

fig, axes = plt.subplots(3, 3, figsize=(9, 9))
axes = axes.flatten()

for index in range(9):
    image = X_test[index]
    true_label = y_test[index]

    batch = np.expand_dims(image, axis=0)

    logits = model.predict(batch, verbose=0)

    single_logits = logits[0]
    probabilities = tf.nn.softmax(single_logits).numpy()

    predicted_id = np.argmax(probabilities)
    predicted_animal = CLASSES[predicted_id]
    confidence = probabilities[predicted_id]

    print(
        index,
        "True:", CLASSES[true_label],
        "| Predicted:", predicted_animal,
        "| Confidence:", f"{confidence * 100:.2f}%"
    )

    axes[index].imshow(image.squeeze(), cmap="gray")

    axes[index].set_title(
        f"True: {CLASSES[true_label]}\n"
        f"Pred: {predicted_animal} ({confidence * 100:.1f}%)"
    )

    axes[index].axis("off")


plt.tight_layout()
plt.show()