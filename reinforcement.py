import pickle
import threading
import numpy as np
import matplotlib.pyplot as plt
import time

from tournament import Tournament
from hex import HexGame
from mcts import ThreadManager, batchRolloutPolicy, Mcts
from neuralnet import createModel, loadModel, createCriticModel
from player import MCTSPlayer, NeuralMCTSPlayer, NeuralNetPlayer, RandomPlayer

# given a game calculate number of rollouts
def getIters(game):
    ratio = game.gameLength / (game.size * game.size)
    if ratio < 0.1:
        return 5*4
    elif ratio < 0.2:
        return 10*4
    elif ratio < 0.3:
        return 15*4
    elif ratio < 0.4:
        return 20*4
    else:
        return 40*4


class ReinforcementLearner:
    def __init__(self, saveInterval, miniBatchSize, parallelGames, boardSize, model, replayBufferSize):
        self.saveInterval = saveInterval
        self.miniBatchSize = miniBatchSize
        self.parallelGames = parallelGames
        self.boardSize = boardSize
        self.model = model
        # self.criticModel = createCriticModel(boardSize)
        self.criticModel = None
        self.replayBufferSize = replayBufferSize

        self.batchesDone = 0
        self.randomWinrate = []
        self.mctsWinrate = []
        self.bestModelWinrate = []

        # calculate time per move
        avgGameTime = 600
        maxMoves = boardSize * boardSize
        self.timePerMove = avgGameTime / maxMoves
        print("Time per move:", self.timePerMove)

        # clear replay buffer
        dataName = f'replayBuffer{boardSize}.pickle'
        with open(dataName, "wb") as f:
            pickle.dump([], f)

        # save this model as the best model
        self.saveModel(model, f'bestmodel.{self.boardSize}')

        # always use batch rollouts when learning
        self.TM = ThreadManager(batchSize=parallelGames, boardSize=boardSize, model=model)

    def playBatch(self):
        start = time.time()
        print("Starting batch", self.batchesDone)
        print("----------------------------------------")
        TM = ThreadManager(self.parallelGames, self.boardSize, self.model)
        for i in range(TM.batchSize):
            # TODO: IF WE USE THE SAME PLAYER IN THE GAME, WE CAN USE THE SAME MCTS TREE. UPDATE MCTS TO ONLY LOOK FOR FIRST CHILD, NOT GRANDCHILD

            # create player1
            nnMctsPlayer = NeuralMCTSPlayer(model=TM.model, maxIters=99999, maxTime=self.timePerMove, argmax=False, TM=TM)
            nnMctsPlayer.mcts.rolloutPolicy = batchRolloutPolicy

            # create game and start thread
            game = HexGame(nnMctsPlayer, nnMctsPlayer, size=TM.boardSize)
            t = threading.Thread(target=TM.threadJob, args=(game,))
            TM.threads.append(t)

        for t in TM.threads:
            t.start()

        for t in TM.threads:
            t.join()

        end = time.time()
        print("Batch", self.batchesDone, "took", end - start, "seconds")

        self.saveReplayBuffer(TM.replayBufferList)
        # normaly 1 minibatch per episode, but probably more when using batches
        for i in range(1):
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

        self.model.fit(X_mini, y_mini, epochs=3, verbose=0)
        # test accuracy on full replay buffer
        loss, acc = self.model.evaluate(X, y, verbose=0)
        print("Accuracy on full replay buffer:", acc)

        if not self.criticModel:
            return

        # also train critic
        X = [r[0] for r in replay]
        y = [r[-1] for r in replay]
        X = np.array(X).reshape(len(X), -1)
        y = np.array(y).reshape(-1, 1)
        y[y == -1] = 0 # for all -1's in y set it to 0
        # choose mini batch
        if len(X) < self.miniBatchSize:
            idx = np.random.choice(len(X), size=len(X), replace=False)
        else:
            idx = np.random.choice(len(X), size=self.miniBatchSize, replace=False)
        X_mini = X[idx]
        y_mini = y[idx]

        self.criticModel.fit(X_mini, y_mini, epochs=1, verbose=0)
        # print accuracy
        loss, acc = self.criticModel.evaluate(X, y, verbose=0)
        print("Critic accuracy on full replay buffer:", acc)


    def saveModel(self, model, modelName):
        model.save(modelName)
        print("Saved model to", modelName)

    def testModel(self):
        numTournamentRounds = 10
        newModelPlayer = NeuralNetPlayer(model=self.model, argmax=True)

        # test vs random
        randomPlayer = RandomPlayer()
        tournament = Tournament(HexGame, [newModelPlayer, randomPlayer], boardSize=self.boardSize, plot=False)
        tournament.run(numTournamentRounds)
        wins, losses, draws = tournament.getPlayerResults(newModelPlayer)
        winrate = wins / (wins + losses + draws)
        self.randomWinrate.append(winrate)
        print(f"NeuralNet@{self.batchesDone} vs Random: {wins} wins, {losses} losses, {draws} draws")

        # test vs mcts
        mctsPlayer = MCTSPlayer(maxIters=50, maxTime=30, argmax=True)
        tournament = Tournament(HexGame, [newModelPlayer, mctsPlayer], boardSize=self.boardSize, plot=False)
        tournament.run(numTournamentRounds)
        wins, losses, draws = tournament.getPlayerResults(newModelPlayer)
        winrate = wins / (wins + losses + draws)
        self.mctsWinrate.append(winrate)
        print(f"NeuralNet@{self.batchesDone} vs MCTS: {wins} wins, {losses} losses, {draws} draws")

    def analyze(self):
        game = HexGame(None, None, size=self.boardSize)
        board = game.getNNState()
        prediction = self.model.predict(board)

        # plot distribution of actions predictions of empty board
        plt.scatter(range(len(prediction[0])), prediction[0], label='prediction')

        # plot distribution actions of empty board with mcts
        mc = Mcts(maxIters=1000, maxTime=10)
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
    parallelGames = 1 # compare iterations to 1
    boardSize = 5
    saveInterval = 1
    miniBatchSize = 64
    replayBufferSize = boardSize*boardSize*parallelGames*10

    modelName = f'model.{boardSize}'
    # initialModel = loadModel(modelName)
    initialModel = createModel(size=boardSize)

    RL = ReinforcementLearner(saveInterval, miniBatchSize, parallelGames, boardSize, initialModel, replayBufferSize)
    # RL.testModel()
    for i in range(100):
        RL.playBatch()

if __name__ == "__main__":
    main()