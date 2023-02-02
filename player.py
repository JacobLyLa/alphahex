import random
from mcts import Mcts

# Player to be used in a game, only needs to implement the play method.
class Player:
    def __init__(self, name, verbose=False):
        self.name = name
        self.verbose = verbose

    def playAction(self, game):
        raise NotImplementedError

class HumanPlayer(Player):
    def __init__(self, name="Human", verbose=True):
        super().__init__(name)
        self.verbose = verbose

    def playAction(self, game):
        if self.verbose:
            print("Turn: " + str(game.getTurn()))
            print("Game State: " + str(game.getState()))
            print("Legal Actions: " + str(game.getActions()))
        action = int(input("Enter action: "))
        print()
        game.playAction(action)

class RandomPlayer(Player):
    def __init__(self, name="Random"):
        super().__init__(name)

    def playAction(self, game):
        action = random.choice(game.getActions())
        game.playAction(action)

class MCTSPlayer(Player):
    def __init__(self, name="MCTS"):
        super().__init__(name)
        self.mcts = Mcts()

    def playAction(self, game):
        action = self.mcts.search(game, self.verbose)
        game.playAction(action)
        if self.verbose:
            print("MCTS Action: " + str(action))