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
    classes = []

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
        classes.append(1)

    # Also include some negative images (no car annotations) if present
    all_files = sorted(os.listdir(image_folder))
    labeled_files = set(df["image"].unique())
    negative_files = [f for f in all_files if f not in labeled_files]

    # sample negatives up to number of positives (1:1) to balance
    max_neg = len(images)
    for nf in negative_files[:max_neg]:
        img_path = os.path.join(image_folder, nf)
        try:
            img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            img = img_to_array(img) / 255.0
        except Exception:
            continue
        images.append(img)
        # bbox zeros for negatives
        boxes.append([0.0, 0.0, 0.0, 0.0])
        classes.append(0)

    return np.array(images), np.array(boxes), np.array(classes)

# -------- PREP --------
X, y_bbox, y_class = load_data("data/training_images", "data/labels.csv")

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
# For bbox loss, only apply it to positive samples: use sample weights
sb_train = yc_train.reshape(-1)
sb_val = yc_val.reshape(-1)

model.fit(
    X_train,
    {"class": yc_train, "bbox": yb_train},
    sample_weight=[None, sb_train],
    validation_data=(X_val, {"class": yc_val, "bbox": yb_val}, [None, sb_val]),
    epochs=20,
    batch_size=16
)

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
    sample_weight=[None, sb_train],
    validation_data=(X_val, {"class": yc_val, "bbox": yb_val}, [None, sb_val]),
    epochs=80, 
    batch_size=16,
    callbacks=[early_stop]
)

# -------- SAVE --------
if not os.path.exists("results"): os.makedirs("results")
model.save("results/car_detector.keras")
print("Model saved as results/car_detector.keras")