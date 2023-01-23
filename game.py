# Two player game class where opponents switch turns.
class Game:
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.turn = 1
        self.state = self.createInitialState()
        self.playGame()

    def playGame(self):
        while not self.isTerminal():
            if self.turn == 1:
                self.player1.play(self)
            else:
                self.player2.play(self)
        return self.getResult()

    # returns 1 if player 1 wins, -1 if player 2 wins, 0 if draw
    def getResult(self):
        raise NotImplementedError

    # returns the initial state of the game
    def createInitialState(self):
        raise NotImplementedError

    # plays action on the current state, updates the state
    def play(self, action):
        raise NotImplementedError

    # returns a list of all possible actions for the current player on the current state
    def getActions(self):
        raise NotImplementedError

    # returns true if the game is over
    def isTerminal(self):
        raise NotImplementedError
 
    def getTurn(self):
        return self.turn

    def getState(self):
        return self.state


class Nim(Game):
    def __init__(self, player1, player2):
        super().__init__(player1, player2)

    def createInitialState(self):
        return 20

    def getActions(self):
        return [1, 2, 3, 4, 5]

    def play(self, action):
        if action not in self.getActions():
            return
        self.state -= action
        self.turn *= -1

    def isTerminal(self):
        return self.state <= 0

    def getResult(self):
        if self.state <= 0:
            return self.turn * -1
        else:
            return 0


# test nim game
if __name__ == "__main__":
    from player import HumanPlayer
    p1 = HumanPlayer("Player 1")
    p2 = HumanPlayer("Player 2")
    nim = Nim(p1, p2)