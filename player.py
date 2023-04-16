import random
import numpy as np
from mcts import Mcts

def argmaxPolicy(actionNodes):
    return max(actionNodes, key=lambda node: node.visits).action

def epsilonGreedyPolicy(actionNodes, epsilon=0.1):
    if random.random() < epsilon:
        return random.choice(actionNodes).action
    else:
        return probabilsticPolicy(actionNodes)

def probabilsticPolicy(actionNodes):
    actionProbs = np.array([node.visits for node in actionNodes])
    actionProbs = actionProbs / np.sum(actionProbs)
    action = np.random.choice(len(actionProbs), p=actionProbs)
    return actionNodes[action].action

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
    def __init__(self, model=None, argmax=False, name="NeuralNet"):
        super().__init__(name)
        self.model = model
        self.argmax = argmax

    def getAction(self, game):
        actionProbs = self.model.predict(game.getNNState(), verbose=0)[0]
        legalActionsMask = np.zeros(len(actionProbs))
        for action in game.getActions():
            legalActionsMask[action] = 1
        actionProbs = actionProbs * legalActionsMask

        if self.argmax:
            action = np.argmax(actionProbs)
        else:
            # 90% chance of argmax, otherwise probabilistic
            if random.random() < 0.9:
                action = np.argmax(actionProbs)
            else:
                actionProbs = actionProbs / np.sum(actionProbs)
                action = np.random.choice(len(actionProbs), p=actionProbs)

        return action

    def playAction(self, game):
        action = self.getAction(game)
        game.playAction(action)

class MCTSPlayer(Player):
    def __init__(self, maxIters, maxTime, argmax=True, name="MCTS"):
        super().__init__(name)
        self.mcts = Mcts(maxIters, maxTime)
        self.argmax = argmax

    def selectAction(self, actionNodes):
        if self.argmax:
            return argmaxPolicy(actionNodes)
        else:
            return epsilonGreedyPolicy(actionNodes)

    def playAction(self, game):
        actionNodes = self.mcts.search(game)
        action = self.selectAction(actionNodes)
        game.playAction(action)
        # if the game is over now, update replaybuffer with winner
        if game.isTerminal():
            # divide by 2 and round up
            movesPlayed = (game.gameLength + 1) // 2
            winner = game.getResult()
            for i in range(movesPlayed):
                # update last element in relevant replaybuffer entry
                self.mcts.replayBuffer[-i-1][-1] = winner


class NeuralMCTSPlayer(Player):
    def __init__(self, model, maxIters, maxTime, argmax=True, name="NeuralMCTS", TM=None):
        super().__init__(name)
        self.model = model
        self.argmax = argmax
        self.rolloutPolicy = lambda game: NeuralNetPlayer(model=model, argmax=False, name="RolloutPolicy").getAction(game)
        self.mcts = Mcts(maxIters, maxTime, self.rolloutPolicy, TM)

    def selectAction(self, actionNodes):
        if self.argmax:
            return argmaxPolicy(actionNodes)
        else:
            return epsilonGreedyPolicy(actionNodes)

    def playAction(self, game):
        actionNodes = self.mcts.search(game)
        action = self.selectAction(actionNodes)
        game.playAction(action)