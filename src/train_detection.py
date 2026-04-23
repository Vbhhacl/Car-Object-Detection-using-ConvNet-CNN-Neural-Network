import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from src.model import build_model

IMG_SIZE = 128

def load_data(image_folder, csv_path):
    df = pd.read_csv(csv_path)
    images = []
    boxes = []

    for _, row in df.iterrows():
        img_path = os.path.join(image_folder, row["image"])
        if not os.path.exists(img_path): continue

        # Get original dimensions
        temp_img = cv2.imread(img_path)
        orig_h, orig_w = temp_img.shape[:2]

        # Load and resize image
        img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img = img_to_array(img) / 255.0
        images.append(img)

        # Normalize coordinates [ymin, xmin, ymax, xmax] 
        # This order matches MobileNet's natural prediction pattern
        ymin = row["ymin"] / orig_h
        xmin = row["xmin"] / orig_w
        ymax = row["ymax"] / orig_h
        xmax = row["xmax"] / orig_w

        boxes.append([ymin, xmin, ymax, xmax])

    return np.array(images), np.array(boxes)

# -------- PREP --------
X, y_bbox = load_data("data/training_images", "data/labels.csv")
y_class = np.ones((len(X), 1))

X_train, X_val, yb_train, yb_val, yc_train, yc_val = train_test_split(
    X, y_bbox, y_class, test_size=0.2, random_state=42
)

# -------- MODEL --------
model = build_model()

# PHASE 1: Warm up the heads
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={"class": "binary_crossentropy", "bbox": "mse"},
    metrics={"class": "accuracy"}
)

print("Starting Phase 1: Training Heads...")
model.fit(X_train, {"class": yc_train, "bbox": yb_train}, epochs=20, batch_size=16)

# PHASE 2: Fine-Tuning (Unfreeze the base model)
print("Starting Phase 2: Fine-Tuning base model...")
model.trainable = True # Unfreeze everything
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), # Very slow
    loss={"class": "binary_crossentropy", "bbox": "mse"},
    metrics={"class": "accuracy"}
)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(
    X_train, {"class": yc_train, "bbox": yb_train},
    validation_data=(X_val, {"class": yc_val, "bbox": yb_val}),
    epochs=80, 
    batch_size=16,
    callbacks=[early_stop]
)

# -------- SAVE --------
if not os.path.exists("results"): os.makedirs("results")
model.save("results/car_detector.keras")
print("Model saved as results/car_detector.keras")