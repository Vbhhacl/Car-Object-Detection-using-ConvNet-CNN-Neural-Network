import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.model_selection import train_test_split

IMG_SIZE = 128


# -------- LOAD DATA --------
def load_data(image_folder, csv_path):
    df = pd.read_csv(csv_path)

    images = []
    boxes = []

    for i, row in df.iterrows():
        img_path = os.path.join(image_folder, row['image'])

        if not os.path.exists(img_path):
            continue

        img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img = img_to_array(img) / 255.0

        # Normalize bounding box
        x1 = row['xmin'] / IMG_SIZE
        y1 = row['ymin'] / IMG_SIZE
        x2 = row['xmax'] / IMG_SIZE
        y2 = row['ymax'] / IMG_SIZE

        images.append(img)
        boxes.append([x1, y1, x2, y2])

        if i > 500:   # limit for memory
            break

    return np.array(images), np.array(boxes)


# -------- MODEL --------
def build_model():
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(128, activation='relu')(x)
    output = Dense(4, activation='sigmoid')(x)  # 4 values (bbox)

    model = Model(inputs=base_model.input, outputs=output)

    for layer in base_model.layers:
        layer.trainable = False

    return model


# -------- PATHS --------
image_path = "data/training_images"
csv_path = "data/labels.csv"

X, y = load_data(image_path, csv_path)

print("Images:", X.shape)
print("Boxes:", y.shape)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- TRAIN --------
model = build_model()

model.compile(
    optimizer='adam',
    loss='mse',   # regression
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=8
)
