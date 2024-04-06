import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np

tf.random.set_seed(42)
np.random.seed(42)

# CNN model construction
model = models.Sequential()
model.add(layers.Conv2D(32, kernel_size=3, padding="same", activation="relu", input_shape=(28, 28, 1)))
model.add(layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"))
model.add(layers.MaxPool2D())

# add classifier
model.add(layers.Flatten())
model.add(layers.Dropout(0.25))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation="softmax"))

# check constructed model
model.summary()

# data preparation
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train_full = X_train_full / 255.
X_test = X_test / 255.
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

X_train = X_train[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# training phase
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
                 metrics=["accuracy"])

model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

# testing phase
test_loss, test_acc = model.evaluate(X_test, y_test)
print('test accuracy ', test_acc)

