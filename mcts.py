import random
import time
from math import sqrt, log
import logging
from neuralnet import getModel
logging.basicConfig()
# logging.getLogger().setLevel(logging.DEBUG)
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
            reward = self.rollout(gameCopy)
            backpropagate(node, reward)

        if iters != self.maxIters:
            self.log.info(f"Search terminated after {iters} maxIters")

        actionNodes = sorted(root.childNodes, key=lambda c: c.visits)
        bestAction = actionNodes[-1].action

        #  Create action distribution probabilities
        totalVisits = sum(node.visits for node in actionNodes)
        actionDist = {node.action: node.visits/totalVisits for node in actionNodes}
        self.replayBuffer.append((game.copy(), actionDist))
        # TODO: swap/reverse board for player -1

        return bestAction

    def rollout(self, gameCopy):
        while not gameCopy.isTerminal():
            gameCopy.playAction(self.rolloutPolicy(gameCopy))
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
    from hex import HexGame
    from player import RandomPlayer, NeuralNetPlayer, NeuralMCTSPlayer, MCTSPlayer
    from tournament import Tournament

    rounds = 10
    model = getModel()
    # nnMctsPlayer = NeuralMCTSPlayer(model=model, maxIters=30, maxTime=10)
    nnMctsPlayer = MCTSPlayer(maxIters=30, maxTime=10)
    randomPlayer = RandomPlayer()
    tournament = Tournament(HexGame, nnMctsPlayer, randomPlayer)
    tournament.run(rounds)
    wins, losses, draws = tournament.getResults()
    print(f"NN MCTS Player: {wins} wins, {losses} losses, {draws} draws")

    replay = nnMctsPlayer.mcts.replayBuffer
    print(replay)
    print(f'Length of replay buffer: {len(replay)}')