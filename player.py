import random
from mcts import Mcts

# Player to be used in a game, only needs to implement the play method.
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
        legalActions = game.getActions()
        # enumerate the legal actions
        print("Legal Actions: ", end="")
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

class MCTSPlayer(Player):
    def __init__(self, name="MCTS", maxIters=100, maxTime=1):
        super().__init__(name)
        self.mcts = Mcts(maxIters, maxTime)

    def playAction(self, game):
        action = self.mcts.search(game)
        game.playAction(action)
        # print("MCTS Action: " + str(action))