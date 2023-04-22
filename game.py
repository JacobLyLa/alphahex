# Two player game class where opponents switch turns.
class Game:
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.turn = 1

    def playGame(self):
        while not self.isTerminal():
            if self.turn == 1:
                self.player1.playAction(self)
            else:
                self.player2.playAction(self)
        return self.getResult()

    # returns 1 if player 1 wins, -1 if player 2 wins, 0 if draw
    def getResult(self):
        raise NotImplementedError

    # plays action on the current state, updates the state
    def playAction(self, action):
        raise NotImplementedError

    # returns a list of all possible actions for the current player on the current state
    def getActions(self):
        raise NotImplementedError

    # returns true if the game is over
    def isTerminal(self):
        raise NotImplementedError

    # copies the game and returns the copy
    def copy(self):
        raise NotImplementedError

    # returns the current state of the game in a string for printing
    def getStringState(self):
        raise NotImplementedError

    # returns the current state of the game in a format that can be used by a neural network
    def getNNState(self):
        raise NotImplementedError

    def getTurn(self):
        return self.turn


class Nim(Game):
    def __init__(self, player1, player2):
        super().__init__(player1, player2)
        self.state = 20

    def getActions(self):
        if self.isTerminal():
            return []
        return [1, 2, 3, 4, 5]

    def playAction(self, action):
        if self.isTerminal():
            print("Game is already over")
            return
        if action not in self.getActions():
            print("Invalid action")
            return
        self.state -= action
        self.turn *= -1

    def isTerminal(self):
        return self.state <= 0

    def getResult(self):
        if self.isTerminal():
            return self.turn * -1
        else:
            return 0

    def copy(self):
        gameCopy = Nim(self.player1, self.player2)
        gameCopy.turn = self.turn
        gameCopy.state = self.state
        return gameCopy

    def getStringState(self):
        return f"There are {self.state} sticks left"


# test nim game
if __name__ == "__main__":
    from player import HumanPlayer, RandomPlayer
    p1 = HumanPlayer("Player 1")
    p2 = RandomPlayer("Player 2")
    nim = Nim(p1, p2)
    print("Result:", nim.playGame())
