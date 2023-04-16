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

# ANET
def createModel(size):
    model = Sequential()
    model.add(Dense(size*size, input_dim=size*size+size, activation='relu'))
    model.add(BatchNormalization())
    # dropout layer
    model.add(Dropout(0.3))
    model.add(Dense(size, activation='relu'))
    model.add(BatchNormalization())
    # Add another Dense layer
    model.add(Dense(size, activation='relu'))
    model.add(BatchNormalization())
    # Add another dropout layer
    model.add(Dropout(0.3))
    # Final layer is a softmax layer for all possible moves
    model.add(Dense(size*size, activation='softmax', kernel_initializer='he_uniform'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

# CRITIC
def createCriticModel(size):
    model = Sequential()
    model.add(Dense(size*size, input_dim=size*size+size, activation='relu'))
    model.add(BatchNormalization())
    # dropout layer
    model.add(Dropout(0.3))
    model.add(Dense(size, activation='relu'))
    model.add(BatchNormalization())
    # Add another Dense layer
    model.add(Dense(size, activation='relu'))
    model.add(BatchNormalization())
    # Add another dropout layer
    model.add(Dropout(0.3))
    # final layer is a singular value from 0 to 1
    model.add(Dense(1, activation='sigmoid', kernel_initializer='he_uniform'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
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