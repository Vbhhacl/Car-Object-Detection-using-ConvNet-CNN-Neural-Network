import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from src.model import build_model

IMG_SIZE = 128

# -------- LOAD DATA --------
def load_data(image_folder, csv_path):
    df = pd.read_csv(csv_path)

    images = []
    boxes = []

    for _, row in df.iterrows():
        img_path = os.path.join(image_folder, row["image"])

        img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img = img_to_array(img) / 255.0

        images.append(img)

        # bbox normalized (IMPORTANT)
        x1 = row["xmin"] / IMG_SIZE
        y1 = row["ymin"] / IMG_SIZE
        x2 = row["xmax"] / IMG_SIZE
        y2 = row["ymax"] / IMG_SIZE

        boxes.append([x1, y1, x2, y2])

    return np.array(images), np.array(boxes)


# -------- PATHS --------
image_path = "data/training_images"
csv_path = "data/labels.csv"

X, y_bbox = load_data(image_path, csv_path)

# 👉 ALL images contain cars in this dataset
y_class = np.ones((len(X), 1))

print("Images:", X.shape)
print("Boxes:", y_bbox.shape)

# -------- SPLIT --------
X_train, X_val, yb_train, yb_val, yc_train, yc_val = train_test_split(
    X, y_bbox, y_class, test_size=0.2, random_state=42
)

# -------- MODEL --------
model = build_model()

model.compile(
    optimizer="adam",
    loss={
        "class": "binary_crossentropy",
        "bbox": "mae"
    },
    metrics={
        "class": "accuracy"
    }
)

# -------- TRAIN --------
history = model.fit(
    X_train,
    {"class": yc_train, "bbox": yb_train},
    validation_data=(X_val, {"class": yc_val, "bbox": yb_val}),
    epochs=5,
    batch_size=8
)

# -------- SAVE MODEL --------
model.save("results/car_detector.h5")

print("Model saved in /results")