from pathlib import Path
import requests

CLASSES = ["cat","dog","fish","bird","rabbit","bear","lion","tiger","elephant","horse",]

BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"

for class_name in CLASSES:
    url = f"{BASE_URL}/{class_name}.npy"
    print(url)