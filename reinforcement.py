import pickle
import threading
import numpy as np
import matplotlib.pyplot as plt

from tournament import Tournament
from hex import HexGame
from mcts import ThreadManager, batchRolloutPolicy, Mcts
from neuralnet import createModel, loadModel
from player import MCTSPlayer, NeuralMCTSPlayer, NeuralNetPlayer, RandomPlayer

class ReinforcementLearner:
    def __init__(self, saveInterval, miniBatchSize, parallelGames, boardSize, model, replayBufferSize):
        self.saveInterval = saveInterval
        self.miniBatchSize = miniBatchSize
        self.parallelGames = parallelGames
        self.boardSize = boardSize
        self.model = model
        self.replayBufferSize = replayBufferSize
        
        self.batchesDone = 0
        self.randomWinrate = []
        self.mctsWinrate = []
        self.bestModelWinrate = []

        # clear replay buffer
        dataName = f'replayBuffer{boardSize}.pickle'
        with open(dataName, "wb") as f:
            pickle.dump([], f)

        # save this model as the best model
        self.saveModel(model, f'bestmodel.{self.boardSize}')

        # always use batch rollouts when learning
        self.TM = ThreadManager(batchSize=parallelGames, boardSize=boardSize, model=model)

    def playBatch(self):
        print("Starting batch", self.batchesDone)
        TM = ThreadManager(self.parallelGames, self.boardSize, self.model)
        for i in range(TM.batchSize):
            # create player1
            nnMctsPlayer = NeuralMCTSPlayer(model=TM.model, maxIters=50, maxTime=10, argmax=False, TM=TM)
            nnMctsPlayer.mcts.rolloutPolicy = batchRolloutPolicy

            # create player2
            nnMctsPlayer2 = NeuralMCTSPlayer(model=TM.model, maxIters=50, maxTime=10, argmax=False, TM=TM)
            nnMctsPlayer2.mcts.rolloutPolicy = batchRolloutPolicy

            # create game and start thread
            game = HexGame(nnMctsPlayer, nnMctsPlayer2, size=TM.boardSize)
            t = threading.Thread(target=TM.threadJob, args=(game,))
            TM.threads.append(t)

        for t in TM.threads:
            t.start()

        for t in TM.threads:
            t.join()

        self.saveReplayBuffer(TM.replayBufferList)
        # normaly 1 minibatch per episode, but difficult when using batches
        for i in range(max(self.parallelGames//2, 1)): 
            self.trainMiniBatch()
        if self.batchesDone % self.saveInterval == 0:
            self.testModel()
            self.saveModel(self.model, f'model.{self.boardSize}')
        # self.analyze()
        self.batchesDone += 1

    def saveReplayBuffer(self, replayBufferList):
        replayBufferList = [item for sublist in replayBufferList for item in sublist]

        # if file exists, load and append
        dataName = f'replayBuffer{self.boardSize}.pickle'
        try:
            with open(dataName, "rb") as f:
                replayBufferList = pickle.load(f) + replayBufferList
        except:
            pass

        # keep only the last replayBufferSize games if full
        if len(replayBufferList) > self.replayBufferSize:
            replayBufferList = replayBufferList[-self.replayBufferSize:]

        with open(dataName, "wb") as f:
            pickle.dump(replayBufferList, f)

        print("Saved", len(replayBufferList), "data points to", dataName)

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

        self.model.fit(X_mini, y_mini, epochs=1, verbose=1)

    def saveModel(self, model, modelName):
        model.save(modelName)
        print("Saved model to", modelName)

    def testModel(self):
        numGames = 11

        newModelPlayer = NeuralNetPlayer(model=self.model, argmax=True)
        bestModel = loadModel(f'bestmodel.{self.boardSize}')
        oldModelPlayer = NeuralNetPlayer(model=bestModel, argmax=True)
        randomPlayer = RandomPlayer()
        mctsPlayer = MCTSPlayer(maxIters=50, maxTime=30, argmax=True)
        nnMctsPlayer = NeuralMCTSPlayer(model=self.model, maxIters=50, maxTime=30, argmax=True)

        # test vs random
        tournament = Tournament(HexGame, newModelPlayer, randomPlayer, boardSize=self.boardSize, plot=False)
        tournament.run(numGames)
        wins, losses, draws = tournament.getResults()
        self.randomWinrate.append(wins/numGames)
        print(f"NeuralNet@{self.batchesDone} vs Random: {wins} wins, {losses} losses, {draws} draws")
        
        # test vs mcts
        tournament = Tournament(HexGame, newModelPlayer, mctsPlayer, boardSize=self.boardSize, plot=False)
        tournament.run(numGames)
        wins, losses, draws = tournament.getResults()
        self.mctsWinrate.append(wins/numGames)
        print(f"NeuralNet@{self.batchesDone} vs MCTS: {wins} wins, {losses} losses, {draws} draws")

        '''
        takes ungodly long
        # nnMcts vs mcts (nnMcts should be better than plain mcts)
        tournament = Tournament(HexGame, nnMctsPlayer, mctsPlayer, boardSize=self.boardSize, plot=False)
        tournament.run(numGames)
        wins, losses, draws = tournament.getResults()
        ## self.nnMctsWinrate.append(wins/numGames)
        print(f"nnMcts@{self.batchesDone} vs MCTS: {wins} wins, {losses} losses, {draws} draws")
        '''

        # test vs previous best model
        # cant use argmax here because it will be the same games
        newModelPlayer.argmax = False
        oldModelPlayer.argmax = False
        tournament = Tournament(HexGame, newModelPlayer, oldModelPlayer, boardSize=self.boardSize)
        tournament.run(numGames)
        wins, losses, draws = tournament.getResults()
        self.bestModelWinrate.append(wins/numGames)
        print(f"NeuralNet@{self.batchesDone} vs BestModel: {wins} wins, {losses} losses, {draws} draws")

        # this is simple version of TOPP
        # if we won more than 55% of the time, save as best model
        if wins >= numGames * 0.5:
            bestModelName = f'bestmodel.{self.boardSize}'
            self.saveModel(self.model, bestModelName)
        else:
            print("No improvement, keeping previous best model")
            self.model = bestModel

    def analyze(self):
        game = HexGame(None, None, size=self.boardSize)
        board = game.getNNState()
        prediction = self.model.predict(board)

        # plot distribution of actions predictions of empty board
        plt.scatter(range(len(prediction[0])), prediction[0], label='prediction')

        
        # plot distribution actions of empty board with mcts
        mc = Mcts(maxIters=2000, maxTime=10)
        mc.search(game)
        dist = mc.replayBuffer
        plt.scatter(range(len(dist[0][-1])), dist[0][-1], label='mcts convergence')     

        # go through replay buffer and find average for each point when x is all zeros
        dataName = f'replayBuffer{self.boardSize}.pickle'
        with open(dataName, 'rb') as f:
            replay = pickle.load(f)
        
        X = np.array([x[0] for x in replay]).reshape(len(replay), -1)
        y = np.array([x[1] for x in replay]).reshape(len(replay), -1)

        avg_y = np.zeros(len(y[0]))
        samples = 0
        for i in range(len(X)):
            # if all the first size*size values are 0, then it's an empty board
            if np.sum(X[i][:self.boardSize*self.boardSize]) == 0:
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
    parallelGames = 64
    boardSize = 3
    saveInterval = 1
    miniBatchSize = 32
    replayBufferSize = boardSize*boardSize*parallelGames*4

    modelName = f'model.{boardSize}'
    # initialModel = loadModel(modelName)
    initialModel = createModel(size=boardSize)

    RL = ReinforcementLearner(saveInterval, miniBatchSize, parallelGames, boardSize, initialModel, replayBufferSize)
    for i in range(100):
        RL.playBatch()

if __name__ == "__main__":
    main()