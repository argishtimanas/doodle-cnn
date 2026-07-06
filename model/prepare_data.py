from pathlib import Path
import numpy as np

SAMPLES_PER_CLASS = 10000

CLASSES = ["cat","dog","fish","bird","rabbit","bear","lion","tiger","elephant","horse",]

DATA_DIR = Path("../data/raw")
PROCESSED_DIR = Path("../data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

all_drawings = []
all_labels = []

for class_id, class_name in enumerate(CLASSES):
    file_path = DATA_DIR / f"{class_name}.npy"
    drawings = np.load(file_path)
    selected_drawings = drawings[:SAMPLES_PER_CLASS]
    class_labels = np.full(SAMPLES_PER_CLASS, class_id)
    all_drawings.append(selected_drawings)
    all_labels.append(class_labels)

X = np.concatenate(all_drawings, axis=0)
y = np.concatenate(all_labels, axis=0)

rng = np.random.default_rng(42)

indices = rng.permutation(len(X))

X = X[indices]
y = y[indices]

X_train = X[:80000] 
y_train = y[:80000]

X_val = X[80000:90000] 
y_val = y[80000:90000] 

X_test = X[90000:] 
y_test = y[90000:]

X_train = X_train.reshape(-1, 28, 28, 1)
X_val = X_val.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

X_train = X_train.astype("float32") / 255.0
X_val = X_val.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

np.save(PROCESSED_DIR / "X_train.npy", X_train)
np.save(PROCESSED_DIR / "y_train.npy", y_train)

np.save(PROCESSED_DIR / "X_val.npy", X_val)
np.save(PROCESSED_DIR / "y_val.npy", y_val)

np.save(PROCESSED_DIR / "X_test.npy", X_test)
np.save(PROCESSED_DIR / "y_test.npy", y_test)

print("Saved processed dataset")
