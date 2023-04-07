import logging
import random
import threading
import time
from math import log, sqrt

import numpy as np

logging.basicConfig()
# logging.getLogger().setLevel(logging.DEBUG)

def randomRolloutPolicy(game):
    return random.choice(game.getActions())  

# a part of batch rolloutpolicy, very similar to getAction in NeuralNetPlayer
def selectAction(game, actionProbs):
    legalActionsMask = np.zeros(len(actionProbs))
    for action in game.getActions():
        legalActionsMask[action] = 1
    actionProbs = actionProbs * legalActionsMask

    # normalize the action probabilities
    actionProbs = actionProbs / np.sum(actionProbs)
    action = np.random.choice(len(actionProbs), p=actionProbs)

    return action

# currently results is probabilities, but should be actions preferably
def batchRolloutPolicy(game, TM):
    gameSize = game.size * game.size
    with TM.lock:
        # find index of this thread ignoring stopped threads
        thread_id = TM.threads.index(threading.current_thread())
        TM.games.append(game.getNNState()[0])

        if len(TM.games) >= len(TM.threads):
            # predict entire batch
            if not TM.calcDone:
                TM.results = TM.model.predict(np.array(TM.games), verbose=0)
                TM.condition.notify_all()
                TM.calcDone = True
        else:
            while not TM.calcDone:
                TM.condition.wait()

        # turn action distribution into action
        TM.results[thread_id] = selectAction(game, TM.results[thread_id])

    # if all threads have come to this point, reset the batch
    TM.barrier.wait()
    TM.games = []
    TM.calcDone = False

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
        # TODO: add temperature here
        if self.turn == 1:
            s = max(self.childNodes, key=lambda c: c.reward/c.visits + sqrt(2*log(self.visits)/c.visits))
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
    def __init__(self, maxIters, maxTime, rolloutPolicy=randomRolloutPolicy, TM=None):
        self.maxIters = maxIters
        self.maxTime = maxTime
        self.rolloutPolicy = rolloutPolicy
        self.TM = TM
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

            if self.TM != None:
                reward = self.batchRollout(gameCopy)
            else:
                reward = self.rollout(gameCopy)

            backpropagate(node, reward)

        actionNodes = sorted(root.childNodes, key=lambda c: c.visits)

        # Create action distribution probabilities
        totalVisits = sum(node.visits for node in actionNodes)
        actionDist = {node.action: node.visits/totalVisits for node in actionNodes}
        actionDistNumpy = np.zeros(game.size * game.size)
        for action, prob in actionDist.items():
            actionDistNumpy[action] = prob
        self.replayBuffer.append((game.getNNState(), actionDistNumpy))

        return actionNodes

    def rollout(self, gameCopy):
        while not gameCopy.isTerminal():
            gameCopy.playAction(self.rolloutPolicy(gameCopy))
        return gameCopy.getResult()

    def batchRollout(self, gameCopy):
        activeThreads = [t for t in self.TM.threads if not t._is_stopped]
        threadid = activeThreads.index(threading.current_thread())
        while not gameCopy.isTerminal():
            self.rolloutPolicy(gameCopy, self.TM)
            # all elements in results[threadid] are the same, so just take the first
            action = int(self.TM.results[threadid][0])
            self.TM.barrier.wait()
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

class ThreadManager:
    def __init__(self, batchSize, boardSize, model):
        self.batchSize = batchSize
        self.boardSize = boardSize
        self.model = model

        self.replayBufferList = []
        self.games = []
        self.barrier = threading.Barrier(batchSize)
        self.lock = threading.Lock()
        self.calcDone = False 
        self.condition = threading.Condition(self.lock)
        self.results =  []
        self.threads = []
        self.startTime = time.time()


    def threadJob(self, game):
        gameCopy = game.copy()
        game.playGame()
        result = []
        p1, p2 = game.player1, game.player2

        # only add to replay buffer if player has mcts
        if hasattr(p1, "mcts"):
            result += p1.mcts.replayBuffer

        if hasattr(p2, "mcts"):
            result += p2.mcts.replayBuffer

        self.replayBufferList.append(result)
        logging.getLogger('THREADS').info(f"{threading.current_thread().name} done | {int(time.time() - self.startTime)}s | {len(self.replayBufferList)}/{self.batchSize} | winner={game.getResult()} | data points={len(result)}")

        # when done continue to call batchRolloutPolicy
        while True: # waste of resources. waits for the last game to finish
            batchRolloutPolicy(gameCopy, self)
            activeThreads = [t for t in self.threads if not t._is_stopped]
            if len(self.replayBufferList) > self.batchSize -1:
                break
            self.barrier.wait()

if __name__ == "__main__":
    from player import RandomPlayer, MCTSPlayer
    from hex import HexGame
    from tournament import Tournament

    # fight mcts vs random player
    boardSize = 4
    numGames = 10
    randomPlayer = RandomPlayer()
    mctsPlayer = MCTSPlayer(maxIters=100, maxTime=20)
    tournament = Tournament(HexGame, mctsPlayer, randomPlayer, boardSize=boardSize, plot=False)
    tournament.run(numGames)
    wins, losses, draws = tournament.getResults()
    print(f"mcts won {wins} times, lost {losses} times and drew {draws} times")