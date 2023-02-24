import random
import numpy as np

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
    def __init__(self, model=None, argmax=True, name="NeuralNet"):
        super().__init__(name)
        self.model = model
        self.argmax = argmax

    def getAction(self, game):
        actionProbs = self.model.predict(game.getNNState(), verbose=0)
        print(actionProbs)
        # if player was 2, flip the actions
        # replace illegal actions with 0
        legalActions = game.getActionsMask()
        for i in range(len(actionProbs[0])):
            if legalActions[i] == 0:
                actionProbs[0][i] = 0
        if self.argmax:
            action = np.argmax(actionProbs)
        else:
            # normalize the action probabilities
            actionProbs = actionProbs / np.sum(actionProbs)
            action = np.random.choice(len(actionProbs[0]), p=actionProbs[0])

        if game.turn == -1:
            action = game.flipAction(action)

        return action

    def playAction(self, game):
        action = self.getAction(game)
        game.playAction(action)

def getNormalNeuralNetRolloutPolicy(model):
    def rolloutPolicy(game):
        return lambda game: NeuralNetPlayer(model=model, argmax=False, name="RolloutPolicy").getAction(game)
    return rolloutPolicy