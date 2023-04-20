import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from hex import HexGame
from mcts import Mcts
from neuralnet import createModel, loadModel
from player import MCTSPlayer, NeuralMCTSPlayer, NeuralNetPlayer, RandomPlayer
from tournament import Tournament


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

        # save this model as the best model
        self.saveModel(model, f'bestmodel.{self.boardSize}')
        # this is for testing
        self.neuralPlayer = NeuralNetPlayer(model=self.model, argmax=True) 
        # this is for training
        self.neuralMctsPlayer = NeuralMCTSPlayer(model=self.model, epsilonMultiplier=epsilonMultiplier, maxIters=99999, maxTime=self.timePerMove, argmax=False)
        self.testModel()

    def oneIteration(self):
        start = time.time()
        print("----------------------------------------")
        print ("Episode", self.episodesDone)
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
        self.neuralMctsPlayer.updateEpsilon()
        self.episodesDone += 1

    def saveReplayBuffer(self, replayBuffer):
        dataName = f'replayBuffer{self.boardSize}.pickle'

        # keep only the last replayBufferSize games if full
        if len(replayBuffer) > self.replayBufferSize:
            replayBuffer = replayBuffer[-self.replayBufferSize:]

        with open(dataName, "wb") as f:
            pickle.dump(replayBuffer, f)

        print("Saved replay buffer with size:", len(replayBuffer))

    def trainMiniBatch(self):
        replayBuffer = self.neuralMctsPlayer.mcts.replayBuffer
        miniSize = self.miniBatchSize

        if len(replayBuffer) < miniSize:
            miniSize = len(replayBuffer)

        # Randomly select miniSize number of pairs from the replay buffer
        miniBatch = random.sample(replayBuffer, miniSize)

        # Separate the selected pairs into x and y arrays
        x_data, y_data = [], []
        for x, y in miniBatch:
            x_data.append(tf.squeeze(x, axis=0))  # Remove the extra dimension
            y_data.append(y)

       # Convert x and y arrays to tensors
        x_data = tf.stack(x_data, axis=0)  # Stack tensors along the batch axis
        y_data = tf.convert_to_tensor(y_data, dtype=tf.float32)

        # Train the model on one epoch using the mini-batch
        self.model.train_on_batch(x_data, y_data)

        # Evaluate the accuracy and loss for the entire replay buffer
        x_buffer, y_buffer = [], []
        for x, y in replayBuffer:
            x_buffer.append(tf.squeeze(x, axis=0))
            y_buffer.append(y)

        x_buffer = tf.stack(x_buffer, axis=0)
        y_buffer = tf.convert_to_tensor(y_buffer, dtype=tf.float32)

        loss, accuracy = self.model.evaluate(x_buffer, y_buffer, verbose=0)
        print(f"Replay buffer evaluation: Loss = {loss}, Accuracy = {accuracy}")


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
    epsilonMultiplier = 0.997
    avgGameTime = 10
    boardSize = 6
    saveInterval = 1
    miniBatchSize = 8
    replayBufferSize = boardSize*boardSize*50

    modelName = f'model.{boardSize}'
    initialModel = createModel(size=boardSize)

    RL = ReinforcementLearner(epsilonMultiplier, avgGameTime, saveInterval, miniBatchSize, boardSize, initialModel, replayBufferSize)
    for i in range(100):
        RL.oneIteration()

if __name__ == "__main__":
    main()