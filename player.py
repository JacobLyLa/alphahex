import random

import numpy as np

from mcts import Mcts


class Player:
    def __init__(self, name):
        self.name = name

    def playAction(self, game):
        raise NotImplementedError

class HumanPlayer(Player):
    def __init__(self, name="Human"):
        super().__init__(name)

    def playAction(self, game):
        print("Turn: " + str(game.getTurn()))
        print("Game State: " + str(game.getStringState()))
        print("Legal Actions: ", end="")
        legalActions = game.getActions()
        # enumerate the legal actions
        for i, action in enumerate(legalActions):
            print(f'{action}[{i+1}]', end=" ")
        print()
        actionIndex = int(input("Enter action index: "))
        if actionIndex < 1 or actionIndex > len(legalActions):
            print("Invalid action index")
            return self.playAction(game)
        action = legalActions[actionIndex - 1]
        print()
        game.playAction(action)

class RandomPlayer(Player):
    def __init__(self, name="Random"):
        super().__init__(name)

    def playAction(self, game):
        action = random.choice(game.getActions())
        game.playAction(action)

class NeuralNetPlayer(Player):
    def __init__(self, model, epsilon=1, epsilonMultiplier=0.99, argmax=False, name="NeuralNet"):
        super().__init__(name)
        self.epsilon = epsilon
        self.epsilonMultiplier = epsilonMultiplier
        self.model = model
        self.argmax = argmax

    def updateEpsilon(self):
        print(f"Epsilon = {self.epsilon * self.epsilonMultiplier:.5f}")
        self.epsilon *= self.epsilonMultiplier

    def playAction(self, game):
        # if not argmax, then use epsilon greedy
        if self.argmax == False:
            if random.random() < self.epsilon:
                action = random.choice(game.getActions())
                game.playAction(action)
                return

        # did not do random action, so do argmax
        actionProbs = self.model(game.getNNState()).numpy()[0]
        legalActionsMask = np.zeros(len(actionProbs))
        for action in game.getActions():
            legalActionsMask[action] = 1
        actionProbs = actionProbs * legalActionsMask
        actionProbs = actionProbs / np.sum(actionProbs)

        if self.argmax == "Probs":
            actionProbs = np.power(actionProbs, 1)
            actionProbs = actionProbs / np.sum(actionProbs)
            action = np.random.choice(len(actionProbs), p=actionProbs)
        else:
            action = np.argmax(actionProbs)
        
        game.playAction(action)

def argmaxPolicy(actionNodes):
    return max(actionNodes, key=lambda node: node.visits).action

def probabilsticPolicy(actionNodes):
    actionProbs = np.array([node.visits for node in actionNodes])
    actionProbs = actionProbs / np.sum(actionProbs)
    action = np.random.choice(len(actionProbs), p=actionProbs)
    return actionNodes[action].action

class MCTSPlayer(Player):
    def __init__(self, maxIters, maxTime, argmax=True, name="MCTS"):
        super().__init__(name)
        self.mcts = Mcts(maxIters, maxTime)
        self.argmax = argmax

    def playAction(self, game):
        actionNodes = self.mcts.search(game)
        if self.argmax:
            action = argmaxPolicy(actionNodes)
        else:
            action = probabilsticPolicy(actionNodes)
        game.playAction(action)

class NeuralMCTSPlayer(Player):
    def __init__(self, model, maxIters, maxTime, epsilonMultiplier=0.997, argmax=True, criticModel=None, name="NeuralMCTS"):
        super().__init__(name)
        self.model = model
        self.argmax = argmax
        self.epsilon = 1
        self.epsilonMultiplier = epsilonMultiplier
        self.neuralNetPlayer = NeuralNetPlayer(model, epsilonMultiplier=epsilonMultiplier, argmax=False, name="RolloutPolicy")
        rolloutPolicy = lambda game: self.neuralNetPlayer.playAction(game)
        self.mcts = Mcts(maxIters, maxTime, rolloutPolicy)

    def updateEpsilon(self):
        self.neuralNetPlayer.updateEpsilon()

    def playAction(self, game):
        actionNodes = self.mcts.search(game)
        if self.argmax:
            action = argmaxPolicy(actionNodes)
        else:
            action = probabilsticPolicy(actionNodes)
        game.playAction(action)
