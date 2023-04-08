import matplotlib.pyplot as plt
from math import sqrt

class Tournament:
    def __init__(self, game, players, boardSize=4, plot=False):
        self.game = game
        self.boardSize = boardSize
        self.players = players
        self.playerWins = [0] * len(players)
        self.playerLosses = [0] * len(players)
        self.playerDraws = [0] * len(players)
        self.plot = plot

        if plot:
            self.initializePlotting()

    def run(self, games_per_permutation=1):
        for _ in range(games_per_permutation):
            for i, player1 in enumerate(self.players):
                for j, player2 in enumerate(self.players):
                    if player1 is player2:
                        continue

                    game = self.game(
                        player1,
                        player2,
                        self.boardSize,
                        plot=self.plot,
                        ax=self.axes[i][j] if self.plot else None
                    )
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

    def initializePlotting(self):
        self.fig, self.axes = plt.subplots(len(self.players), len(self.players))

        for row in range(len(self.players)):
            for col in range(len(self.players)):
                ax: plt.Axes = self.axes[row, col]
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)

                distCorner = 1
                distSide = distCorner * sqrt(3) / 2

                ax.set_aspect('equal')
                padding = 0.05
                ax.set_xlim(-distSide - padding, (3 * self.boardSize - 2) * distSide + padding)
                ax.set_ylim(-(1.5 * self.boardSize - 0.5) * distCorner - padding, distCorner + padding)

                player1Name = self.players[row].name
                player2Name = self.players[col].name
                if row == 0:
                    pad = 15
                    ax.annotate(
                        player2Name,
                        xy=(0.5, 1),
                        xytext=(0, pad),
                        xycoords='axes fraction',
                        textcoords='offset points',
                        size='large',
                        ha='center',
                        va='baseline'
                    )
                if col == 0:
                    pad = 5
                    ax.annotate(
                        player1Name,
                        xy=(0, 0.5),
                        xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label,
                        textcoords='offset points',
                        size='large',
                        ha='right',
                        va='center',
                        rotation=90
                    )

        self.fig.tight_layout()