import logging
import random
import time
from math import log, sqrt

import numpy as np

logging.basicConfig()
# logging.getLogger().setLevel(logging.DEBUG)

def randomRolloutPolicy(game):
    game.playAction(random.choice(game.getActions()))

class Node:
    def __init__(self, game, action=None, parent=None):
        self.game = game.copy()
        self.action = action
        self.parentNode = parent
        self.childNodes = []
        self.untriedActions = game.getActions()
        self.turn = game.getTurn()
        self.visits = 0
        self.reward = 0

    def selectChild(self): # UCT
        # TODO: add temperature
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
    def __init__(self, maxIters, maxTime, rolloutPolicy=randomRolloutPolicy):
        # TODO: rename iters to simulations
        self.maxIters = maxIters
        self.maxTime = maxTime
        self.rolloutPolicy = rolloutPolicy
        self.root = None
        self.log = logging.getLogger("MCTS")
        self.log.setLevel(logging.INFO)
        self.replayBuffer = []

    def search(self, game):
        # check if there is a child node in child nodes in root that is the same as the current game state
        found = False
        if self.root != None:
            for child in self.root.childNodes:
                # check if the two games are equal
                if (child.game.equals(game)):
                    self.root = child
                    self.root.parentNode = None # save memory
                    found = True
                    break
                if found:
                    break
        if not found:
            self.root = Node(game)
        start = time.time()
        iters = 0

        while (time.time() - start < self.maxTime) and iters < self.maxIters:
            iters += 1
            node = self.root
            gameCopy = game.copy()

            node = select(node, gameCopy)
            node = expand(node, gameCopy)

            reward = self.rollout(gameCopy)

            backpropagate(node, reward)

        actionNodes = sorted(self.root.childNodes, key=lambda c: c.visits)
        # print how many iterations were done
        print(iters, end=" ")

        # Create action distribution probabilities
        totalVisits = sum(node.visits for node in actionNodes)
        actionDist = {node.action: node.visits/totalVisits for node in actionNodes}
        actionDistNumpy = np.zeros(game.size * game.size)
        for action, prob in actionDist.items():
            actionDistNumpy[action] = prob

        if game.getTurn() == -1:
            actionDistNumpy = actionDistNumpy.reshape(game.size, game.size).T.reshape(-1)

        self.replayBuffer.append([game.getNNState(), actionDistNumpy])
        return actionNodes

    def rollout(self, gameCopy):
        while not gameCopy.isTerminal():
            self.rolloutPolicy(gameCopy)
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

if __name__ == "__main__":
    from player import RandomPlayer, MCTSPlayer
    from hex import HexGame
    from tournament import Tournament

    # fight mcts vs random player
    boardSize = 5
    numTournamentRounds = 1
    players = [
        MCTSPlayer(maxIters=500, maxTime=20),
        RandomPlayer()
    ]
    tournament = Tournament(HexGame, players, boardSize=boardSize, plot=True)
    tournament.run(numTournamentRounds)
    tournament.printResults()

    # fight mcts vs mcts with less iterations
    boardSize = 5
    numTournamentRounds = 1
    players = [
        MCTSPlayer(maxIters=500, maxTime=20, name="MCTS 500"),
        MCTSPlayer(maxIters=100, maxTime=20, name="MCTS 100")
    ]
    tournament = Tournament(HexGame, players, boardSize=boardSize, plot=True)
    tournament.run(numTournamentRounds)
    tournament.printResults()