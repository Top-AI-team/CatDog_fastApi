import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from PIL import Image
import io


model = Sequential([
  layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(150, 150, 3)),
  layers.BatchNormalization(),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  tf.keras.layers.Dropout(0.5),
  layers.Flatten(),
  layers.Dense(256, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

gpus = tf.config.experimental.list_logical_devices()
print(gpus)

model.load_weights('Weights_folder/Weights').expect_partial()


def predict_class(img_bytes):
    global model
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert('RGB')
    img = img.resize((150, 150), Image.NEAREST)

    x = keras.preprocessing.image.img_to_array(img) / 255.
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    result = model.predict(images, batch_size=10, verbose="silent")

    if result[0] > 0.5:
        return "Dog"
    else:
        return "Cat"
