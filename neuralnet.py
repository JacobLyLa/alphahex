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
    model.add(Dense(size*size, activation='softmax', kernel_initializer='he_uniform'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])
    return model

def loadModel(path):
    return tf.keras.models.load_model(path)

if __name__ == '__main__':
    from hex import HexGame
    from player import NeuralNetPlayer, RandomPlayer, MCTSPlayer, NeuralMCTSPlayer
    from tournament import Tournament
    rounds = 1
    boardSize = 7
    model = loadModel('bestmodel.7')

    nnMctsPlayer = NeuralMCTSPlayer(model=model, maxIters=15, maxTime=5)
    mctsPlayer = MCTSPlayer(100, maxTime=5)
    tournament = Tournament(HexGame, [nnMctsPlayer, mctsPlayer], boardSize=boardSize, plot=True)
    tournament.run(rounds)
    tournament.printResults()

    replay = nnMctsPlayer.mcts.replayBuffer
    print(f'Length of replay buffer: {len(replay)}')

    # train model
    X = np.array([x[0] for x in replay]).reshape(len(replay), -1)
    y = np.array([x[1] for x in replay]).reshape(len(replay), -1)
    print(X)
    print(y)
    model.fit(X, y, epochs=2, verbose=1)