from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path("../data/raw")

cat_path = DATA_DIR / "cat.npy"

cat_drawings = np.load(cat_path)

first_drawing = cat_drawings[0]

image = first_drawing.reshape(28, 28)
normalized_image = image / 255.0

print(normalized_image.min())
print(normalized_image.max())
print(normalized_image.dtype)