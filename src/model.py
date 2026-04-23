import tensorflow as tf

def build_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Shared features
    x = tf.keras.layers.Dense(128, activation="relu")(x)

    # 🟢 Classification head (car / no car)
    class_output = tf.keras.layers.Dense(
        1, activation="sigmoid", name="class"
    )(x)

    # 🟢 Bounding box head
    box_output = tf.keras.layers.Dense(
        4, activation="sigmoid", name="bbox"
    )(x)

    model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[class_output, box_output]
    )

    return model