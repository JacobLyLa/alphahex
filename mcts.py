import random
import time
from math import sqrt, log
import graphviz as gv

# TODO: temp default policy, replace with neural network
# TODO: continue from previous search
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
        # if player is maximizing, select node with highest reward/visits + uct
        # if player is minimizing, select node with lowest reward/visits - uct
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
    def __init__(self, iterations=200, timeLimit=2, verbose=False):
        self.iterations = iterations
        self.timeLimit = timeLimit
        self.verbose = verbose

    def search(self, game, verbose=None):
        if verbose != None:
            self.verbose = verbose
        root = Node(game)
        start = time.time()
        iters = 0
        while (time.time() - start < self.timeLimit) and iters < self.iterations:
            iters += 1
            node = root
            gameCopy = game.copy()

            node = self.select(node, gameCopy)
            node = self.expand(node, gameCopy)
            reward = self.rollout(gameCopy)
            self.backpropagate(node, reward)

        actionNodes = sorted(root.childNodes, key=lambda c: c.visits)
        bestAction = actionNodes[-1].action

        if self.verbose:
            for actionNode in actionNodes:
                print(actionNode)

        return bestAction

    def select(self, node, gameCopy):
        while node.untriedActions == [] and node.childNodes != []:
            node = node.selectChild()
            gameCopy.playAction(node.action)
        return node

    def expand(self, node, gameCopy):
        if node.untriedActions != []:
            action = random.choice(node.untriedActions)
            gameCopy.playAction(action)
            node = node.addChild(gameCopy, action)
        return node

    def rollout(self, gameCopy):
        while not gameCopy.isTerminal():
            gameCopy.playAction(rolloutPolicy(gameCopy))
        return gameCopy.getResult()

    def backpropagate(self, node, reward):
        while node != None:
            node.update(reward)
            node = node.parentNode


# test with nim game
if __name__ == "__main__":
    from game import Nim
    from player import HumanPlayer, MCTSPlayer, RandomPlayer
    
    mctsPlayer = MCTSPlayer()
     
    wins = 0
    for i in range(100):
        if i % 2 == 0:
            game = Nim(mctsPlayer, RandomPlayer())
        else:
            game = Nim(RandomPlayer(), mctsPlayer)

        result = game.playGame()
        if i%2 != 0:
            result *= -1
        if result == 1:
            wins += 1
    print("MCTS won", wins, "out of 100 games vs random bot")

    humanPlayer = HumanPlayer()
    mctsPlayer.verbose = True
    game = Nim(humanPlayer, mctsPlayer)
    game.playGame()