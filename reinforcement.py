import pickle
import threading
import numpy as np

from tournament import Tournament
from hex import HexGame
from mcts import ThreadManager, batchRolloutPolicy
from neuralnet import createModel, loadModel
from player import MCTSPlayer, NeuralMCTSPlayer, NeuralNetPlayer, RandomPlayer

class ReinforcementLearner:
    def __init__(self, saveInterval, miniBatchSize, batchSize, boardSize, model):
        self.saveInterval = saveInterval
        self.miniBatchSize = miniBatchSize
        self.batchSize = batchSize
        self.boardSize = boardSize
        self.model = model
        
        self.batchesDone = 0

        # clear replay buffer
        dataName = f'replayBuffer{boardSize}.pickle'
        with open(dataName, "wb") as f:
            pickle.dump([], f)

        # save this model as the best model
        self.saveModel(model, f'bestmodel.{self.boardSize}')

        # always use batch rollouts when learning
        self.TM = ThreadManager(batchSize=batchSize, boardSize=boardSize, model=model)

    def playBatch(self):
        TM = ThreadManager(self.batchSize, self.boardSize, self.model)
        for i in range(TM.batchSize):
            # create player1
            nnMctsPlayer = NeuralMCTSPlayer(model=TM.model, maxIters=400, maxTime=20, TM=TM)
            nnMctsPlayer.mcts.rolloutPolicy = batchRolloutPolicy

            # create player2
            nnMctsPlayer2 = NeuralMCTSPlayer(model=TM.model, maxIters=400, maxTime=20, TM=TM)
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
        for i in range(10): # normally 1 minibatch per episode, but difficult when using batches
            self.trainMiniBatch()
        self.testModel()
        if self.batchesDone % self.saveInterval == 0:
            self.saveModel(self.model, f'model.{self.boardSize}')
        self.batchesDone += 1

    def saveReplayBuffer(self, replayBufferList):
        # flatten replaybuffer and save
        replayBufferList = [item for sublist in replayBufferList for item in sublist]

        # if file exists, load and append
        dataName = f'replayBuffer{self.boardSize}.pickle'
        try:
            with open(dataName, "rb") as f:
                replayBufferList = pickle.load(f) + replayBufferList
        except:
            pass

        # slice if over 10k. TODO: make this a parameter
        if len(replayBufferList) > 10000:
            replayBufferList = replayBufferList[-10000:]

        with open(dataName, "wb") as f:
            pickle.dump(replayBufferList, f)

        print("Saved", len(replayBufferList), "data points to", dataName)

    def trainMiniBatch(self):
        dataName = f'replayBuffer{self.boardSize}.pickle'
        with open(dataName, 'rb') as f:
            replay = pickle.load(f)
        
        X = np.array([x[0] for x in replay]).reshape(len(replay), self.boardSize*self.boardSize)
        y = np.array([x[1] for x in replay]).reshape(len(replay), self.boardSize*self.boardSize)

        # pick random mini batch of size miniBatchSize
        idx = np.random.choice(len(X), size=self.miniBatchSize, replace=False)
        X_mini = X[idx]
        y_mini = y[idx]

        # train model
        self.model.fit(X_mini, y_mini, epochs=1, verbose=1)

    def saveModel(self, model, modelName):
        model.save(modelName)
        print("Saved model to", modelName)

    def testModel(self):
        wins = 0
        tournament = Tournament(HexGame, NeuralNetPlayer(model=self.model), RandomPlayer(), boardSize=self.boardSize)
        tournament.run(31)
        wins, losses, draws = tournament.getResults()
        print(f"NeuralNet@{self.batchesDone} vs Random: {wins} wins, {losses} losses, {draws} draws")
        
        # same but vs mcts instead of random
        wins = 0
        tournament = Tournament(HexGame, NeuralNetPlayer(model=self.model), MCTSPlayer(maxIters=100, maxTime=5), boardSize=self.boardSize)
        tournament.run(31)
        wins, losses, draws = tournament.getResults()
        print(f"NeuralNet@{self.batchesDone} vs MCTS: {wins} wins, {losses} losses, {draws} draws")

        # same but vs best model
        bestModel = loadModel(f'bestmodel.{self.boardSize}')
        wins = 0
        tournament = Tournament(HexGame, NeuralNetPlayer(model=self.model), NeuralNetPlayer(model=bestModel), boardSize=self.boardSize)
        tournament.run(31)
        wins, losses, draws = tournament.getResults()
        print(f"NeuralNet@{self.batchesDone} vs BestModel: {wins} wins, {losses} losses, {draws} draws")

        # if we won more than 50% of the time, save as best model
        if wins > 15:
            bestModelName = f'bestmodel.{self.boardSize}'
            self.saveModel(self.model, bestModelName)

def main():
    batchSize = 64
    boardSize = 4

    modelName = f'model.{boardSize}'
    # initialModel = loadModel(modelName)
    initialModel = createModel(size=boardSize)

    RL = ReinforcementLearner(1, 32, batchSize, boardSize, initialModel)
    for i in range(10):
        RL.playBatch()

if __name__ == "__main__":
    main()