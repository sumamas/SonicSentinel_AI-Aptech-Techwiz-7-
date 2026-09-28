from __future__ import annotations

from src.ml.training_config import N_MELS, TARGET_MEL_FRAMES, CLASSES, SEED


def build_custom_cnn():
    """Build SonicSentinel's custom CNN from random initialization.

    This function intentionally does not load a pretrained network, transfer
    learning backbone, or external weights.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required. Install it with: "
            "python -m pip install -r requirements-ml.txt"
        ) from exc

    tf.keras.utils.set_random_seed(SEED)

    inputs = tf.keras.Input(shape=(N_MELS, TARGET_MEL_FRAMES, 1), name="log_mel")
    x = inputs

    for filters, dropout_rate in ((32, 0.10), (64, 0.15), (128, 0.20), (256, 0.25)):
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer="he_normal",
            use_bias=False,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Dropout(0.40)(x)
    outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax", name="class_probabilities")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="sonicsentinel_custom_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
