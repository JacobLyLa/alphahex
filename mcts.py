import random
import time
from math import sqrt, log
import logging
logging.basicConfig()

# TODO: temp default policy, replace with neural network and continue from previous search
def rolloutPolicy(game):
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
        return "[MCTS Node] Action: " + str(self.action) + " Visits: " + str(self.visits) + " Expected Reward: " + str(round(self.reward/self.visits,3))

    def __str__(self):
        return self.__repr__()


class Mcts:
    def __init__(self, maxIters, maxTime):
        self.maxIters = maxIters
        self.maxTime = maxTime
        self.log = logging.getLogger("MCTS")
        self.log.setLevel(logging.INFO)

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
            reward = rollout(gameCopy)
            backpropagate(node, reward)

        if iters != self.maxIters:
            self.log.info(f"Search terminated after {iters} maxIters")

        actionNodes = sorted(root.childNodes, key=lambda c: c.visits)
        bestAction = actionNodes[-1].action

        for actionNode in actionNodes:
            self.log.debug(actionNode)

        return bestAction

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

def rollout(gameCopy):
    while not gameCopy.isTerminal():
        gameCopy.playAction(rolloutPolicy(gameCopy))
    return gameCopy.getResult()

def backpropagate(node, reward):
    while node != None:
        node.update(reward)
        node = node.parentNode

# test with nim game
if __name__ == "__main__":
    from game import Nim
    from player import HumanPlayer, MCTSPlayer, RandomPlayer
    from tournament import Tournament

    rounds = 20
    mctsPlayer = MCTSPlayer(maxIters=100, maxTime=1)
    tournament = Tournament(Nim, mctsPlayer, RandomPlayer())
    tournament.run(rounds)
    wins, losses, draws = tournament.getResults()
    print("MCTS won", wins, "out of", rounds, "games")

    humanPlayer = HumanPlayer()
    game = Nim(humanPlayer, mctsPlayer)
    game.playGame()