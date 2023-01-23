# Player to be used in a game, only needs to implement the play method.
class Player:
    def __init__(self, name):
        self.name = name

    def play(self, game):
        raise NotImplementedError

class HumanPlayer(Player):
    def __init__(self, name="Human", verbose=True):
        self.verbose = verbose
        super().__init__(name)

    def play(self, game):
        if self.verbose:
            print("Turn: " + str(game.getTurn()))
            print("State: " + str(game.getState()))
            print("Actions: " + str(game.getActions()))
        action = int(input("Enter action: "))
        print()
        game.play(action)