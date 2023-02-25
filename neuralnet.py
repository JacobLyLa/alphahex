import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# regression model
def createModel(size):
    model = Sequential()
    model.add(Dense(size*size, input_dim=size*size, activation='relu'))
    model.add(Dense(size*size, activation='softmax'))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01))
    return model

def loadModel(path):
    return tf.keras.models.load_model(path)

if __name__ == '__main__':
    from hex import HexGame
    from mcts import MCTSPlayer, NeuralMCTSPlayer
    from player import NeuralNetPlayer, RandomPlayer
    from tournament import Tournament
    rounds = 1
    model = createModel()
    nnMctsPlayer = NeuralMCTSPlayer(model=model, maxIters=30, maxTime=5)
    randomPlayer = RandomPlayer()
    tournament = Tournament(HexGame, nnMctsPlayer, randomPlayer)
    tournament.run(rounds)
    wins, losses, draws = tournament.getResults()
    print(f"NN MCTS Player: {wins} wins, {losses} losses, {draws} draws")

    replay = nnMctsPlayer.mcts.replayBuffer
    print(replay)
    print(f'Length of replay buffer: {len(replay)}')

    # train model
    X = np.array([x[0] for x in replay])
    y = np.array([x[1] for x in replay])
    model.fit(X, y, epochs=10, verbose=1)