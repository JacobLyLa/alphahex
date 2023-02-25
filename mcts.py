import logging
import random
import threading
import time
from math import log, sqrt

import numpy as np

logging.basicConfig()
# logging.getLogger().setLevel(logging.DEBUG)

from player import NeuralNetPlayer, Player


class MCTSPlayer(Player):
    def __init__(self, maxIters=100, maxTime=1, name="MCTS"):
        super().__init__(name)
        self.mcts = Mcts(maxIters, maxTime)

    def playAction(self, game):
        action = self.mcts.search(game)
        game.playAction(action)

class NeuralMCTSPlayer(Player):
    def __init__(self, model=None, maxIters=100, maxTime=1, name="NeuralMCTS"):
        super().__init__(name)
        self.model = model
        self.rolloutPolicy = lambda game: NeuralNetPlayer(model=model, argmax=False, name="RolloutPolicy").getAction(game)
        self.mcts = Mcts(maxIters, maxTime, self.rolloutPolicy)

    def playAction(self, game):
        action = self.mcts.search(game)
        game.playAction(action)

def batchRolloutPolicy(game):
    gameSize = game.size * game.size
    global games
    global calc_done
    global results
    # check how many threads that are started or initiated
    with lock:
        # find index of this thread ignoring stopped threads
        thread_id = threads.index(threading.current_thread())
        game = game.getNNState()
        games.append(game)

        if len(games) >= len(threads):
            if not calc_done:
                results = model.predict(np.array(games).reshape(len(threads), gameSize), verbose=0)
                condition.notify_all()
                calc_done = True
        else:
            while not calc_done:
                condition.wait()
                
        thread_action = results[thread_id]
        results[thread_id] = thread_action

    # if all threads have come to this point, reset the batch
    barrier.wait()
    games = []
    calc_done = False

# TODO: this is just copied from neuralnet player
def getAction(game, probs):
    gameSize = game.size * game.size
    actionProbs = probs.reshape((1, gameSize))
    legalActions = game.getActionsMask()
    for i in range(len(actionProbs[0])):
        if legalActions[i] == 0:
            actionProbs[0][i] = 0

    # normalize the action probabilities
    actionProbs = actionProbs / np.sum(actionProbs)
    action = np.random.choice(len(actionProbs[0]), p=actionProbs[0])

    if game.turn == -1:
        action = game.flipAction(action)

    return action
       
def randomRolloutPolicy(game):
    return random.choice(game.getActions())

class Node:
    def __init__(self, game, action=None, parent=None):
        self.action = action
        self.parentNode = parent
        self.childNodes = []
        self.untriedActions = game.getActions()
        self.turn = game.getTurn()
        self.visits = 0
        self.reward = 0

    def selectChild(self): # UCT
        if self.turn == 1:
            s = max(self.childNodes, key=lambda c: c.reward/c.visits + sqrt(2*log(self.visits)/c.visits)) # can add temperature here
        else:
            s = min(self.childNodes, key=lambda c: c.reward/c.visits - sqrt(2*log(self.visits)/c.visits))
        return s

    def addChild(self, game, action):
        node = Node(game, action, self)
        self.untriedActions.remove(action)
        self.childNodes.append(node)
        return node

    def update(self, reward):
        self.visits += 1
        self.reward += reward

    def __repr__(self):
        return f"[MCTS Node] Action: {self.action} Visits: {self.visits} Expected Reward: {self.reward/self.visits :.3f}"

    def __str__(self):
        return self.__repr__()


class Mcts:
    def __init__(self, maxIters, maxTime, rolloutPolicy=randomRolloutPolicy):
        self.maxIters = maxIters
        self.maxTime = maxTime
        self.rolloutPolicy = rolloutPolicy
        self.log = logging.getLogger("MCTS")
        self.log.setLevel(logging.INFO)
        self.replayBuffer = []

    def search(self, game):
        root = Node(game)
        start = time.time()
        iters = 0
        while (time.time() - start < self.maxTime) and iters < self.maxIters:
            iters += 1
            node = root
            gameCopy = game.copy()

            node = select(node, gameCopy)
            node = expand(node, gameCopy)

            if self.rolloutPolicy.__name__ == "batchRolloutPolicy":
                reward = self.batchRollout(gameCopy)
            else:
                reward = self.rollout(gameCopy)
            backpropagate(node, reward)

        if iters != self.maxIters:
            self.log.info(f"Search terminated after {iters} maxIters")

        actionNodes = sorted(root.childNodes, key=lambda c: c.visits)
        bestAction = actionNodes[-1].action

        #  Create action distribution probabilities
        totalVisits = sum(node.visits for node in actionNodes)
        actionDist = {node.action: node.visits/totalVisits for node in actionNodes}
        actionDistNumpy = np.zeros((game.size, game.size))
        for action, prob in actionDist.items():
            actionDistNumpy[action] = prob
        if game.getTurn() == -1: # TODO: refactor to make mcts agnostic to turn
            actionDistNumpy = actionDistNumpy.T
        actionDistNumpy = actionDistNumpy.reshape((1, game.size*game.size))
        self.replayBuffer.append((game.getNNState(), actionDistNumpy))

        return bestAction

    def rollout(self, gameCopy):
        while not gameCopy.isTerminal():
            gameCopy.playAction(self.rolloutPolicy(gameCopy))
        return gameCopy.getResult()

    def batchRollout(self, gameCopy):
        activeThreads = [t for t in threads if not t._is_stopped]
        threadid = activeThreads.index(threading.current_thread())
        while not gameCopy.isTerminal():
            self.rolloutPolicy(gameCopy)
            probs = results[threadid]
            action = getAction(gameCopy, probs)
            barrier.wait()
            gameCopy.playAction(action)
        return gameCopy.getResult()

def select(node, gameCopy):
    while node.untriedActions == [] and node.childNodes != []:
        node = node.selectChild()
        gameCopy.playAction(node.action)
    return node

def expand(node, gameCopy):
    if node.untriedActions != []:
        action = random.choice(node.untriedActions)
        gameCopy.playAction(action)
        node = node.addChild(gameCopy, action)
    return node

def backpropagate(node, reward):
    while node != None:
        node.update(reward)
        node = node.parentNode

def threadJob(game, replayBufferList):
    gameCopy = game.copy()
    game.playGame()
    result = []
    p1 = game.player1
    if isinstance(p1, NeuralMCTSPlayer):
        result += p1.mcts.replayBuffer
    
    p2 = game.player2
    if isinstance(p2, NeuralMCTSPlayer):
        result += p2.mcts.replayBuffer

    replayBufferList.append(result)
    print(f"{threading.current_thread().name} done | {int(time.time() - start)}s | {len(replayBufferList)}/{batchSize} | winner={game.getResult()} | data points={len(result)}")

    # when done just continue to call batchRolloutPolicy
    while True: # yes super waste of resources, have to wait for the last game to finish
        batchRolloutPolicy(gameCopy)
        activeThreads = [t for t in threads if not t._is_stopped]
        if len(replayBufferList) > batchSize -1:
            break
        barrier.wait()

if __name__ == "__main__":
    import pickle

    from hex import HexGame
    from neuralnet import createModel, loadModel
    from player import RandomPlayer
    batchSize = 128
    barrier = threading.Barrier(batchSize)
    games = []
    lock = threading.Lock()
    calc_done = False
    condition = threading.Condition(lock)
    results =  [None] * batchSize
    threads = []

    boardSize = 4
    modelName = f'model.{boardSize}'
    model = createModel(size=boardSize)
    # model = loadModel(modelName)
    replayBufferList = []
    start = time.time()
    for i in range(batchSize):
        # create unique players such that replay buffer is unique
        # alternatively use set to remove duplicates
        nnMctsPlayer = NeuralMCTSPlayer(model=model, maxIters=100, maxTime=400)
        nnMctsPlayer.mcts.rolloutPolicy = batchRolloutPolicy
        nnMctsPlayer2 = NeuralMCTSPlayer(model=model, maxIters=100, maxTime=400)
        nnMctsPlayer2.mcts.rolloutPolicy = batchRolloutPolicy
        game = HexGame(nnMctsPlayer, nnMctsPlayer2, size=boardSize)
        t = threading.Thread(target=threadJob, args=(game, replayBufferList))
        threads.append(t)
        
    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # flatten results and save
    replayBufferList = [item for sublist in replayBufferList for item in sublist]
    print("Replaybuffer has", len(replayBufferList), "data points")

    # if file exists, load and append
    dataName = f'replayBuffer{boardSize}.pickle'
    try:
        with open(dataName, "rb") as f:
            replayBufferList = pickle.load(f) + replayBufferList
    except:
        pass

    with open(dataName, "wb") as f:
        pickle.dump(replayBufferList, f)

    print("Saved", len(replayBufferList), "data points to", dataName)