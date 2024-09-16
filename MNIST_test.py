import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D

# Create Model
tf.random.set_seed(42)
inputs = tf.keras.Input(shape=(28, 28, 1))
x = Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
x = MaxPooling2D(pool_size=2)(x)
x = Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = MaxPooling2D(pool_size=2)(x)
x = Flatten()(x)
x = Dense(10, activation="relu")(x)
outputs = Dense(10, activation="softmax")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()

# Load and preprocess MNIST data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[..., np.newaxis] / 255.0
x_test = x_test[..., np.newaxis] / 255.0

# Compile and train the model
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
his = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# Plot training history
pd.DataFrame(his.history).plot()
plt.show()  # Ensure the plot is displayed

def testmnist(pic):
    pred = model.predict(np.reshape(x_test[pic], (1, 28, 28, 1)))
    plt.imshow(x_test[pic].squeeze(), cmap='gray')  # Squeeze to remove single-dimensional entries
    plt.show()
    print(np.reshape(pred, (10, 1)))
    maxpos = np.argmax(pred)
    name = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    print("Prediction = %s" % name[maxpos])

# Test with a specific image index
testmnist(6)
