import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.model_selection import train_test_split
from src.model import build_model
from src.utils import plot_results
from tensorflow.keras.utils import load_img, img_to_array


# -------- LOAD DATA --------
def load_images(folder):
    images = []
    labels = []

    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return np.array([]), np.array([])

    files = os.listdir(folder)
    np.random.shuffle(files)

    for i, file in enumerate(files):

        # limit dataset (avoid memory crash)
        if i > 500:
            break

        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(folder, file)

        try:
            # ✅ reduced size (memory fix)
            img = load_img(img_path, target_size=(128, 128))
            img = img_to_array(img) / 255.0

            images.append(img)
            labels.append(1)  # all cars

        except Exception as e:
            print("Error loading:", img_path)

    print("Total images loaded:", len(images))
    return np.array(images), np.array(labels)


# -------- PATH --------
data_path = "data/training_images"

X, y = load_images(data_path)

if len(X) == 0:
    raise ValueError("No images loaded. Check dataset path.")


# -------- SPLIT --------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -------- TRAINING --------
optimizers = {
    "Adam": Adam(learning_rate=0.0001),
    "SGD": SGD(learning_rate=0.0001),
    "RMSprop": RMSprop(learning_rate=0.0001),
}

results = {}

for name, opt in optimizers.items():
    print(f"\nTraining with {name}...\n")

    model = build_model()

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",   # ✅ FIXED
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=8   # ✅ reduced memory
    )

    results[name] = history


# -------- PLOT --------
plot_results(results)