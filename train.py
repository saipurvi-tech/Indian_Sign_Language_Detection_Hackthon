import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

DATA_DIR, IMG, BATCH, EPOCHS = "data", 64, 32, 15

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=42,
    image_size=(IMG, IMG), batch_size=BATCH, color_mode="grayscale")
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=42,
    image_size=(IMG, IMG), batch_size=BATCH, color_mode="grayscale")

class_names = train_ds.class_names
print("Classes:", class_names)

model = models.Sequential([
    layers.Input((IMG, IMG, 1)),
    layers.Rescaling(1 / 255),
    layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation="softmax"),
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

model.save("model.keras")
with open("labels.json", "w") as f:
    json.dump(class_names, f)