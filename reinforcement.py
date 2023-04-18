import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import random

from tournament import Tournament
from hex import HexGame
from mcts import Mcts
from neuralnet import createModel, loadModel
from player import MCTSPlayer, NeuralMCTSPlayer, NeuralNetPlayer, RandomPlayer


class ReinforcementLearner:
    def __init__(self, epsilonMultiplier, avgGameTime, saveInterval, miniBatchSize, boardSize, model, replayBufferSize):
        self.saveInterval = saveInterval
        self.miniBatchSize = miniBatchSize
        self.boardSize = boardSize
        self.model = model
        self.replayBufferSize = replayBufferSize

        self.episodesDone = 0
        self.randomWinrate = []
        self.mctsWinrate = []
        self.bestModelWinrate = []

        # calculate time per move
        maxMoves = boardSize * boardSize
        self.timePerMove = avgGameTime / maxMoves
        print("Time per move:", self.timePerMove)

        # clear replay buffer
        dataName = f'replayBuffer{boardSize}.pickle'
        with open(dataName, "wb") as f:
            pickle.dump([], f)

        # save this model as the best model
        self.saveModel(model, f'bestmodel.{self.boardSize}')
        self.neuralPlayer = NeuralNetPlayer(model=self.model, epsilonMultiplier=epsilonMultiplier, argmax=True)
        self.neuralMctsPlayer = NeuralMCTSPlayer(model=self.model, maxIters=99999, maxTime=self.timePerMove, argmax=False)

    def oneIteration(self):
        start = time.time()
        print("Starting episode", self.episodesDone)
        print("----------------------------------------")
        nnMctsPlayer = self.neuralMctsPlayer

        # create game
        game = HexGame(nnMctsPlayer, nnMctsPlayer, size=self.boardSize)
        game.playGame()
        end = time.time()
        print("Episode", self.episodesDone, "took", end - start, "seconds")

        self.saveReplayBuffer(nnMctsPlayer.mcts.replayBuffer)
        self.trainMiniBatch()
        if self.episodesDone % self.saveInterval == 0:
            self.testModel()
            self.saveModel(self.model, f'model.{self.boardSize}')
        # self.analyze()
        self.neuralPlayer.updateEpsilon()
        self.episodesDone += 1

    def saveReplayBuffer(self, episodeBuffer):
        # if file exists, load and append
        dataName = f'replayBuffer{self.boardSize}.pickle'
        try:
            with open(dataName, "rb") as f:
                replayBuffer = pickle.load(f) + episodeBuffer 
        except:
            print("No replay buffer found!")

        # keep only the last replayBufferSize games if full
        if len(replayBuffer) > self.replayBufferSize:
            replayBuffer = replayBuffer[-self.replayBufferSize:]

        with open(dataName, "wb") as f:
            pickle.dump(replayBuffer, f)

        print("Saved", len(replayBuffer), "data points to", dataName)

    def trainMiniBatch(self):
        dataName = f'replayBuffer{self.boardSize}.pickle'
        with open(dataName, 'rb') as f:
            replay = pickle.load(f)

        X = np.array([x[0] for x in replay]).reshape(len(replay), -1)
        y = np.array([x[1] for x in replay]).reshape(len(replay), -1)

        # choose mini batch
        if len(X) < self.miniBatchSize:
            idx = np.random.choice(len(X), size=len(X), replace=False)
        else:
            idx = np.random.choice(len(X), size=self.miniBatchSize, replace=False)
        X_mini = X[idx]
        y_mini = y[idx]

        self.model.fit(X_mini, y_mini, epochs=1, verbose=0)
        # test accuracy on full replay buffer

        loss, acc = self.model.evaluate(X, y, verbose=0)
        print("Accuracy on full replay buffer:", acc)
        print("Loss on full replay buffer:", loss)


    def saveModel(self, model, modelName):
        model.save(modelName)
        print("Saved model to", modelName)

    def testModel(self):
        numTournamentRounds = 50

        # test vs random
        randomPlayer = RandomPlayer()
        tournament = Tournament(HexGame, [self.neuralPlayer, randomPlayer], boardSize=self.boardSize, plot=False)
        tournament.run(numTournamentRounds)
        wins, losses, draws = tournament.getPlayerResults(self.neuralPlayer)
        winrate = wins / (wins + losses + draws)
        self.randomWinrate.append(winrate)
        print(f"NeuralNet@{self.episodesDone} vs Random: {wins} wins, {losses} losses, {draws} draws")

        '''
        # test vs mcts
        mctsPlayer = MCTSPlayer(maxIters=50, maxTime=10, argmax=True)
        tournament = Tournament(HexGame, [self.neuralPlayer, mctsPlayer], boardSize=self.boardSize, plot=False)
        tournament.run(numTournamentRounds)
        wins, losses, draws = tournament.getPlayerResults(self.neuralPlayer)
        winrate = wins / (wins + losses + draws)
        self.mctsWinrate.append(winrate)
        print(f"NeuralNet@{self.episodesDone} vs MCTS: {wins} wins, {losses} losses, {draws} draws")
        '''
        

    def analyze(self):
        game = HexGame(None, None, size=self.boardSize)
        prediction = self.model(game.getNNState())

        # plot distribution of actions predictions of empty board
        plt.scatter(range(len(prediction[0])), prediction[0], label='prediction')

        # plot distribution actions of empty board with mcts
        mc = Mcts(maxIters=5000, maxTime=10)
        mc.search(game)
        dist = mc.replayBuffer
        plt.scatter(range(len(dist[0][1])), dist[0][1], label='mcts convergence')

        # go through replay buffer and find average for each point when x is all zeros
        dataName = f'replayBuffer{self.boardSize}.pickle'
        with open(dataName, 'rb') as f:
            replay = pickle.load(f)

        X = np.array([x[0] for x in replay]).reshape(len(replay), -1)
        y = np.array([x[1] for x in replay]).reshape(len(replay), -1)

        avg_y = np.zeros(len(y[0]))
        samples = 0
        for i in range(len(X)):
            # if it is the same gamestate
            if (X[i] == game.getNNState().numpy()).all():
                samples += 1
                avg_y += y[i]
        avg_y /= samples

        plt.scatter(range(len(avg_y)), avg_y, label='avg y')
        plt.legend()
        plt.show()

        # do a random move and do the same plot again
        game.playAction(random.choice(game.getActions()))
        prediction = self.model(game.getNNState())
        plt.scatter(range(len(prediction[0])), prediction[0], label='prediction')

        mc = Mcts(maxIters=5000, maxTime=10)
        mc.search(game)
        dist = mc.replayBuffer
        plt.scatter(range(len(dist[0][1])), dist[0][1], label='mcts convergence')

        avg_y = np.zeros(len(y[0]))
        samples = 0
        for i in range(len(X)):
            # if it is the same gamestate
            if (X[i] == game.getNNState().numpy()).all():
                samples += 1
                avg_y += y[i]
        avg_y /= samples

        plt.scatter(range(len(avg_y)), avg_y, label='avg y')
        plt.legend()
        plt.show()


        # plot winrates
        plt.title(f'Winrate vs. batches played')
        plt.plot(self.randomWinrate, label='random')
        plt.plot(self.mctsWinrate, label='mcts')
        plt.plot(self.bestModelWinrate, label='best model')
        plt.legend()
        plt.show()


def main():
    epsilonMultiplier = 0.995
    avgGameTime = 10
    boardSize = 4
    saveInterval = 1
    miniBatchSize = 32
    replayBufferSize = boardSize*boardSize*20

    modelName = f'model.{boardSize}'
    initialModel = createModel(size=boardSize)

    RL = ReinforcementLearner(epsilonMultiplier, avgGameTime, saveInterval, miniBatchSize, boardSize, initialModel, replayBufferSize)
    RL.testModel()
    for i in range(100):
        RL.oneIteration()

if __name__ == "__main__":
    main()