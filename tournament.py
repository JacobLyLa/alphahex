import random

'''
Restores best model from file and returns it as a player
'''
def restoreBestPlayer():
    pass

'''
Saves player to file as a model
'''
def savePlayer(player):
    pass

# TODO: tournment between M agents
class Tournament:
    def __init__(self, game, player1, player2):
        self.game = game
        self.player1 = player1
        self.player2 = player2
        self.player1Wins = 0
        self.player2Wins = 0
        self.draws = 0

    def run(self, games):
        # random player starts
        swapSide = random.randint(0, 1)
        for _ in range(games):
            if swapSide == 1:
                game = self.game(self.player2, self.player1)
            else:
                game = self.game(self.player1, self.player2)
            game.playGame()

            result = game.getResult()
            if result == 1 and swapSide == 0:
                self.player1Wins += 1
            elif result == 1 and swapSide == 1:
                self.player2Wins += 1
            elif result == -1 and swapSide == 0:
                self.player2Wins += 1
            elif result == -1 and swapSide == 1:
                self.player1Wins += 1
            else:
                self.draws += 1

            swapSide = 1 - swapSide

    def getResults(self):
        return self.player1Wins, self.player2Wins, self.draws
