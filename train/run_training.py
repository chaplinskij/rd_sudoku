import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras import Model
from time import time

from train.services import *

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = Loader.load_digit_dataset(data_dir='out')
    num_classes = 10
    size = x_train.shape[1]

    # Normalization
    x_train = x_train / 255
    x_test = x_test / 255

    # One-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    inputs = Input(shape=(size, size, 1))

    net = Conv2D(16, kernel_size=(3, 3), activation="relu")(inputs)
    net = BatchNormalization()(net)
    net = MaxPooling2D(pool_size=(2, 2))(net)
    net = Conv2D(32, kernel_size=(3, 3), activation="relu")(net)
    net = MaxPooling2D(pool_size=(2, 2))(net)
    net = Flatten()(net)
    net = Dropout(0.3)(net)
    outputs = Dense(num_classes, activation="softmax")(net)

    model = Model(inputs, outputs)

    epochs = 25
    batch_size = 128
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    start = time()
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    print('Elapsed time', time() - start)
    model.save('model.h5')
