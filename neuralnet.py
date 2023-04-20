import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.layers import Dropout, BatchNormalization
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from keras.layers import Conv2D, Flatten, Reshape, MaxPooling2D

def createModel(size):
    input_shape = (size, size, 4)  # Assumes square images with 4 channels
    model = Sequential()
    model.add(Reshape(input_shape, input_shape=(size*size*4,)))  # Reshape the flattened input data
    model.add(Conv2D(size, kernel_size=(3, 3), activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(size*size, activation='sigmoid'))
    model.add(Dense(size*size, activation='softmax', kernel_initializer='he_uniform'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.005), metrics=['accuracy'])
    return model

def loadModel(path):
    return tf.keras.models.load_model(path)