from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path("../data/raw")

cat_path = DATA_DIR / "cat.npy"

cat_drawings = np.load(cat_path)

first_drawing = cat_drawings[0]

image = first_drawing.reshape(28, 28)

print(image.shape)

plt.imshow(image, cmap="gray")
plt.title("First cat drawing")
plt.axis("off")
plt.show()

print(image.min())
print(image.max())
print(np.unique(image)[:20])