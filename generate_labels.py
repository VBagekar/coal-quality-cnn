import cv2
import os
import numpy as np
import pandas as pd

folder = "Data/coal_Images"
rows = []

for filename in os.listdir(folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(folder, filename)
        img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img_gray is None:
            print(f" Could not read {filename}, skipping.")
            continue

        # Parameters
        base_ash = 20
        base_moisture = 3
        max_ash = 45
        max_moisture = 8.1

        brightness = np.mean(img_gray)

        ash = np.clip(base_ash + (255 - brightness) * 0.10, base_ash, max_ash)
        moisture = np.clip(base_moisture + brightness * 0.02, base_moisture, max_moisture)

        rows.append([filename, round(moisture, 2), round(ash, 2)])

df = pd.DataFrame(rows, columns=["filename", "moisture", "ash"])
df.to_csv("labels.csv", index=False)

print("labels.csv created using grayscale image statistics.")
