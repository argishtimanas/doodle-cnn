from pathlib import Path
import requests

CLASSES = ["cat","dog","fish","bird","rabbit","bear","lion","tiger","elephant","horse",]

BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"

DATA_DIR = Path("../data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

for class_name in CLASSES:
    url = f"{BASE_URL}/{class_name}.npy"
    print(f"Downloading {class_name}...")
    response = requests.get(url)
    response.raise_for_status()
    output_path = DATA_DIR / f"{class_name}.npy"
    output_path.write_bytes(response.content)
    print(f"Saved {class_name}")
