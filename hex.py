import random
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from math import sqrt
from game import Game
logging.basicConfig()
# logging.getLogger().setLevel(logging.DEBUG)


NEIGHBORS = [(0, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1)]

class HexGame(Game):
    def __init__(self, player1, player2, size=4, plot=False):
        self.player1 = player1
        self.player2 = player2
        self.size = size
        self.board = np.zeros(shape=(size,size), dtype=np.int8)
        self.turn = 1
        self.log = logging.getLogger(__name__)

        self.plot = plot
        if plot:
            self.plotter = HexPlotter(self)

    def getStringState(self):
        return self.board.copy()

    def getNNState(self):
        board = self.board * self.turn
        if self.turn == -1:
            board = board.T
        return board.reshape((1, -1))

    def flipAction(self, action):
        action = (action // self.size, action % self.size)
        action = (action[1], action[0])
        return action[0] * self.size + action[1]

    def getActionsMask(self):
        mask = np.zeros(shape=(self.board.shape[0] * self.board.shape[1]), dtype=np.int8)
        # flatten the board
        if self.turn == -1:
            flatBoard = self.board.T.reshape(-1)
        else:
            flatBoard = self.board.reshape(-1)
        for i in range(len(flatBoard)):
            if flatBoard[i] == 0:
                mask[i] = 1

        return mask

    def getActions(self):
        actions = []
        for x in range(self.board.shape[0]):
            for y in range(self.board.shape[1]):
                if self.board[x, y] == 0:
                    actions.append((x, y))
        return actions

    def playAction(self, action):
        # TODO: move to network class
        # if action is int, convert to tuple
        if isinstance(action, int) or isinstance(action, np.int64):
            action = (action // self.board.shape[1], action % self.board.shape[1])

        # check if the coordinates are within the board
        if not self.withinBounds(action):
            self.log.warning(f'{action} is out of bounds\n')
            return

        # check if the coordinates are already occupied
        if self.board[action] != 0:
            self.log.warning(f'{action} is already occupied\n')
            return

        self.board[action] = self.turn
        self.turn = self.turn * -1

        self.log.debug(f'player {self.turn} made move {action} resulting in board state:\n{self.board}\n')

        if self.plot:
            self.plotter.plot_action(action)

    def isTerminal(self):
        return self.hasPlayerWon(1) or self.hasPlayerWon(-1)

    def getResult(self):
        if self.hasPlayerWon(1):
            return 1
        elif self.hasPlayerWon(-1):
            return -1
        else:
            return 0

    def copy(self):
        gameCopy = HexGame(self.player1, self.player2, size=self.size)
        gameCopy.turn = self.turn
        gameCopy.board = self.board.copy()
        return gameCopy

    def withinBounds(self, position) -> bool:
        return 0 <= position[0] < self.board.shape[0] and 0 <= position[1] < self.board.shape[1]

    def searchForWin(self, player, position, visited):
        if not self.withinBounds(position):
            return False

        if position in visited:
            return False
        visited.add(position)

        if self.board[position] != player:
            return False

        # check if player 1 has won
        if player == 1 and position[0] == self.board.shape[0] - 1:
            return True

        # check if player -1 has won
        if player == -1 and position[1] == self.board.shape[1] - 1:
            return True

        for neighbor in NEIGHBORS:
            neighborPos = (position[0] + neighbor[0], position[1] + neighbor[1])

            if self.searchForWin(player, neighborPos, visited):
                return True

        return False

    def hasPlayerWon(self, player):
        if player == 1:
            for x in range(self.board.shape[0]):
                if self.board[0, x] == player:
                    visited = set()
                    if self.searchForWin(player, (0, x), visited):
                        return True
        else:
            for y in range(self.board.shape[1]):
                if self.board[y, 0] == player:
                    visited = set()
                    if self.searchForWin(player, (y, 0), visited):
                        return True

        return False


class HexPlotter():
    def __init__(self, game: HexGame, R=0.5, pause_after_plot=0.001):
        self.game = game
        self.pause_after_plot = pause_after_plot

        # distance between center and corner
        self.dist_corner = R
        # distance between center and middle of side
        self.dist_side = R * sqrt(3) / 2

        fig, ax = plt.subplots(1, figsize=(10, 8))
        ax.set_title(f'Red player is {game.player1.name}, blue player is {game.player2.name}')
        ax.axis('off')
        ax.set_aspect('equal')
        padding = 0.02
        ax.set_xlim(-self.dist_side - padding, (3 * game.size - 2) * self.dist_side + padding)
        ax.set_ylim(-(1.5 * game.size - 0.5) * self.dist_corner - padding, self.dist_corner + padding)
        self.ax = ax

        self.colors = {1: 'red', -1: 'blue', 0: 'white'}

        self.plot_board()

    def plot_board(self):
        for x in range(self.game.size):
            for y in range(self.game.size):
                self.plot_hexagon(x, y, self.colors[self.game.board[x, y]])
        plt.pause(self.pause_after_plot)

    def plot_action(self, action):
        self.plot_hexagon(action[0], action[1], self.colors[-1 * self.game.turn])
        plt.pause(self.pause_after_plot)

    def plot_hexagon(self, row, col, color='white'):
        x = 2 * self.dist_side * col + self.dist_side * row
        y = - 1.5 * self.dist_corner * row
        hex = RegularPolygon((x, y), numVertices=6, radius=self.dist_corner, orientation=0, facecolor=color, edgecolor='black', alpha=0.8, linewidth=2)
        self.ax.add_patch(hex)


if __name__ == '__main__':
    from player import RandomPlayer
    from tournament import Tournament
    r1, r2 = RandomPlayer("Random1"), RandomPlayer("Random2")
    tournament = Tournament(HexGame, r1, r2, plot=True)
    tournament.run(5)
    wins, losses, draws = tournament.getResults()
    print(f'{r1.name} won {wins} times, {r2.name} won {losses} times, and there were {draws} draws')
    plt.show() # to not close the plot after the game is finished