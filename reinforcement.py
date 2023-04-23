import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

from hex import HexGame
from mcts import Mcts
from neuralnet import createModel, loadModel
from player import MCTSPlayer, NeuralMCTSPlayer, NeuralNetPlayer, RandomPlayer
from tournament import Tournament


class ReinforcementLearner:
    def __init__(self, epochs, epsilonMultiplier, avgGameTime, saveInterval, miniBatchSize, boardSize, model, replayBufferSize):
        self.epochs = epochs
        self.saveInterval = saveInterval
        self.miniBatchSize = miniBatchSize
        self.boardSize = boardSize
        self.model = model
        self.replayBufferSize = replayBufferSize

        self.episodesDone = 0
        self.randomWinrate = []
        self.trainLoss = []
        self.trainAccuracy = []
        self.testLoss = []
        self.testAccuracy = []

        self.folderName = f'{boardSize}b {avgGameTime}s {epsilonMultiplier}e {miniBatchSize}m {epochs}t'
        if not os.path.exists(self.folderName):
            os.makedirs(self.folderName)
        self.folderName += '/'

        # load x_test and y_test (Created in notebook)
        with open(f'testSet{boardSize}.pickle', 'rb') as f:
            self.x_test, self.y_test = pickle.load(f)

        # calculate time per move
        maxMoves = boardSize * boardSize
        self.timePerMove = avgGameTime / maxMoves
        print("Time per move:", self.timePerMove)

        # this is for testing
        self.neuralPlayer = NeuralNetPlayer(model=self.model, argmax=True) 
        # this is for training
        self.neuralMctsPlayer = NeuralMCTSPlayer(model=self.model, epsilonMultiplier=epsilonMultiplier, maxIters=99999, maxTime=self.timePerMove, argmax=False)

    def oneIteration(self):
        print("----------------------------------------")
        print ("Episode", self.episodesDone)
        nnMctsPlayer = self.neuralMctsPlayer

        # create game
        game = HexGame(nnMctsPlayer, nnMctsPlayer, size=self.boardSize)
        game.playGame()
        print()
        self.saveReplayBuffer(nnMctsPlayer.mcts.replayBuffer)
        self.trainMiniBatch()
        if self.episodesDone % self.saveInterval == 0:
            self.testModel()
            self.saveModel(self.model, f'model.{self.episodesDone}e')
        self.neuralMctsPlayer.updateEpsilon()
        self.episodesDone += 1
        

    def saveReplayBuffer(self, replayBuffer):
        dataName = self.folderName + 'replayBuffer.pickle'

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

        for _ in range(self.epochs):
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
        print(f"Train: Loss = {loss:.5f}, Accuracy = {accuracy:.5f}")

        self.trainLoss.append(loss)
        self.trainAccuracy.append(accuracy)

    def saveModel(self, model, modelName):
        model.save(self.folderName + modelName)
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

        # test on test set
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"Test: Loss = {loss:.5f}, Accuracy = {accuracy:.5f}")

        self.testLoss.append(loss)
        self.testAccuracy.append(accuracy)

        # save all metrics
        with open(self.folderName + 'metrics.pickle', "wb") as f:
            pickle.dump([self.trainLoss, self.trainAccuracy, self.testLoss, self.testAccuracy, self.randomWinrate], f)

def main():
    epsilonMultiplier = 0.9999
    avgGameTime = 240
    epochs = 5
    miniBatchSize = 128
    
    saveInterval = 10
    boardSize = 7
    replayBufferSize = boardSize*boardSize*100
    # do random hyperparamenters
    '''
    epsilonMultiplier = round(random.uniform(0.9, 0.999), 4)
    avgGameTime = random.randint(5, 300)
    epochs = random.randint(1, 10)
    miniBatchSize = 2**random.randint(2, 8)
    '''

    modelName = f'model.{boardSize}'
    initialModel = createModel(size=boardSize)

    RL = ReinforcementLearner(epochs, epsilonMultiplier, avgGameTime, saveInterval, miniBatchSize, boardSize, initialModel, replayBufferSize)
    for i in range(2000):
        RL.oneIteration()

if __name__ == "__main__":
    main()