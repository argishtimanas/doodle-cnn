from pathlib import Path
import numpy as np

SAMPLES_PER_CLASS = 10000

CLASSES = ["cat","dog","fish","bird","rabbit","bear","lion","tiger","elephant","horse",]

DATA_DIR = Path("../data/raw")

for class_id, class_name in enumerate(CLASSES):
    file_path = DATA_DIR / f"{class_name}.npy"
    drawings = np.load(file_path)
    selected_drawings = drawings[:SAMPLES_PER_CLASS]
    class_labels = np.full(SAMPLES_PER_CLASS, class_id)
    print(class_name, selected_drawings.shape, class_labels.shape, class_labels[:5])

