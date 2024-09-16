import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras
import pandas as pd
import sklearn
from keras.layers import Flatten,Dense
from keras import Sequential,layers
from sklearn.metrics import confusion_matrix

#Create Model
tf.random.set_seed(42)
inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Flatten()(x)
x = layers.Dense(10, activation="relu")(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

#Plot on pandas
mnist = keras.datasets.mnist
(x_train,y_train),(x_test,y_test)= mnist.load_data()
x_train=x_train/255.0
x_test=x_test/255.0
model.compile(optimizer="rmsprop",loss="sparse_categorical_crossentropy",\
metrics=["accuracy"])
his=model.fit(x_train, y_train, epochs=5, \
batch_size=64,validation_data=(x_test, y_test))
pd.DataFrame(his.history).plot()

def testmnist(pic):
    pred=model.predict(np.reshape(x_test[pic],(1,28,28)))
    plt.imshow(x_test[pic],cmap='gray')
    print(np.reshape(pred,(10,1)))
    maxpos=np.argmax(pred)
    name=['zero','one','two','three','four',
    'five','six','seven','eight','nine']
    print("Prediction = %s"%name[maxpos])

testmnist(6)