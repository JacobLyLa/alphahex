import random

class Tournament:
    def __init__(self, game, players, boardSize=4, plot=False):
        self.game = game
        self.boardSize = boardSize
        self.players = players
        self.playerWins = [0] * len(players)
        self.playerLosses = [0] * len(players)
        self.playerDraws = [0] * len(players)
        self.plot = plot

    def run(self, games_per_permutation):
        for _ in range(games_per_permutation):
            for i, player1 in enumerate(self.players):
                for j, player2 in enumerate(self.players):
                    if player1 is player2:
                        continue

                    game = self.game(player1, player2, self.boardSize, plot=self.plot)
                    game.playGame()
                    result = game.getResult()

                    if result == 1:
                        self.playerWins[i] += 1
                        self.playerLosses[j] += 1
                    elif result == -1:
                        self.playerWins[j] += 1
                        self.playerLosses[i] += 1
                    else:
                        self.playerDraws[i] += 1
                        self.playerDraws[j] += 1

    def getResults(self):
        return self.playerWins, self.playerLosses, self.playerDraws

    def printResults(self):
        for i, player in enumerate(self.players):
            print(f'{player.name} won {self.playerWins[i]} times, lost {self.playerLosses[i]} times, and drew {self.playerDraws[i]} times')