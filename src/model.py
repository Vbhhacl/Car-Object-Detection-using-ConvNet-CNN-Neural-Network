import tensorflow as tf

def build_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights="imagenet"
    )

    # Start with frozen base to train the new heads
    base_model.trainable = False

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x) # Added dropout to prevent overfitting

    # Classification head
    class_output = tf.keras.layers.Dense(1, activation="sigmoid", name="class")(x)

    # Bounding box head (Sigmoid is perfect since we normalized 0-1)
    box_output = tf.keras.layers.Dense(4, activation="sigmoid", name="bbox")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=[class_output, box_output])
    return model