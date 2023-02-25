import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

size=4
inputSize = size*size
outputSize = size*size

# regression model
def getModel():
    model = Sequential()
    model.add(Dense(16, input_dim=inputSize, activation='relu'))
    model.add(Dense(outputSize, activation='softmax'))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01))
    return model

if __name__ == '__main__':
    from hex import HexGame
    from mcts import MCTSPlayer, NeuralMCTSPlayer
    from player import NeuralNetPlayer, RandomPlayer
    from tournament import Tournament
    rounds = 1
    model = getModel()
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