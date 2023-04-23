import pickle
import time

import numpy as np
import tensorflow as tf
from keras.layers import (ELU, Activation, BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, LeakyReLU, MaxPooling2D, Reshape)
from keras.optimizers import SGD, Nadam, RMSprop
from keras.regularizers import l2
from keras.utils.generic_utils import get_custom_objects
from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout,
                                     LeakyReLU)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2



def createModel(size):
    input_shape = (4, size, size)
    output_shape = size * size

    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), padding='same', input_shape=input_shape, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(output_shape, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    return model


def loadModel(path, compile=True):
    return tf.keras.models.load_model(path, compile=compile)