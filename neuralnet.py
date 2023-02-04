from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

size=4
inputSize = size*size
outputSize = size*size

# regression model
def getModel():
    model = Sequential()
    model.add(Dense(16, input_dim=inputSize, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(outputSize, activation='softmax'))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
    return model