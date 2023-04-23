import tensorflow as tf
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten)
from keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
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