import logging
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.patches import RegularPolygon

from game import Game

logging.basicConfig()
# logging.getLogger().setLevel(logging.DEBUG)


NEIGHBORS = [(0, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1)]

class HexGame(Game):
    def __init__(self, player1, player2, size=4, plot=False, ax=None):
        self.player1 = player1
        self.player2 = player2
        self.size = size
        self.board = np.zeros(shape=(size,size), dtype=np.int8)
        self.turn = 1
        self.gameLength = 0
        self.log = logging.getLogger(__name__)

        self.plot = plot
        if plot:
            self.plotter = HexPlotter(self, ax=ax)

    def equals(self, other):
        return np.array_equal(self.board, other.board) and self.turn == other.turn and self.gameLength == other.gameLength

    def getStringState(self):
        return self.board.copy()

    # add turn as the last element
    def getNNState(self):
        # first bitmap is current players stones, second bitmap is opponents stones, third bitmap is turn empty space, fourth bitmap is turn
        '''
        firstMap = np.where(self.board == self.turn, 1, 0)
        secondMap = np.where(self.board == self.turn * -1, 1, 0)
        thirdMap = np.where(self.board == 0, 1, 0)
        fourthMap = np.ones(shape=self.board.shape) * self.turn
        board = np.stack((firstMap, secondMap, thirdMap, fourthMap), axis=2)
        '''

        # here the bitmaps are the represent player1 and player2, not turn
        firstMap = np.where(self.board == 1, 1, 0)
        secondMap = np.where(self.board == -1, 1, 0)
        thirdMap = np.where(self.board == 0, 1, 0)
        fourthMap = np.ones(shape=self.board.shape) * self.turn
        board = np.stack((firstMap, secondMap, thirdMap, fourthMap), axis=0)

        '''
        board = self.board.copy()
        # print(f'turn: {self.turn}')
        # print(f'board before: {board}')
        if self.turn == -1:
            board = board.T * -1
            # print(f'board after: {board}')
        '''
        # TODO: check that the representation is correct

        tensor = tf.convert_to_tensor(board, dtype=tf.float32)
        tensor = tf.expand_dims(tensor, 0)
        return tensor

    def getActions(self):
        board = self.board
        actions = []
        for x in range(board.shape[0]):
            for y in range(board.shape[1]):
                if board[x, y] == 0:
                    actions.append(x * self.board.shape[1] + y)

        return actions

    def playAction(self, action):
        # TODO: move to network class if action is int, convert to tuple
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
        self.gameLength += 1

        self.log.debug(f'player {self.turn * -1} made move {action} resulting in board state:\n{self.board}\n')

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
        gameCopy.gameLength = self.gameLength
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
    def __init__(self, game: HexGame, distCorner=1, pauseAfterPlot=0.001, ax=None):
        self.game = game
        self.pauseAfterPlot = pauseAfterPlot

        # distance between center and corner
        self.distCorner = distCorner
        # distance between center and middle of side
        self.distSide = distCorner * sqrt(3) / 2

        if ax is None:
            _, self.ax = plt.subplots(1, figsize=(10, 8))
        else:
            self.ax = ax

        self.ax.set_aspect('equal')
        padding = 0.05
        self.ax.set_xlim(-self.distSide - padding, (3 * game.size - 2) * self.distSide + padding)
        self.ax.set_ylim(-(1.5 * game.size - 0.5) * self.distCorner - padding, self.distCorner + padding)

        self.colors = {1: 'red', -1: 'blue', 0: 'white'}

        self.plot_board()

    def plot_board(self):
        for x in range(self.game.size):
            for y in range(self.game.size):
                self.plot_hexagon(x, y, self.colors[self.game.board[x, y]])
        plt.pause(self.pauseAfterPlot)

    def plot_action(self, action):
        self.plot_hexagon(action[0], action[1], self.colors[-1 * self.game.turn])
        plt.pause(self.pauseAfterPlot)

    def plot_hexagon(self, row, col, color='white'):
        x = 2 * self.distSide * col + self.distSide * row
        y = - 1.5 * self.distCorner * row
        hex = RegularPolygon((x, y), numVertices=6, radius=self.distCorner, orientation=0, facecolor=color, edgecolor='black', linewidth=2)
        self.ax.add_patch(hex)


if __name__ == '__main__':
    from player import RandomPlayer
    from tournament import Tournament
    '''
    numPlayers = 3
    players = [RandomPlayer(f"Random{i}") for i in range(numPlayers)]
    tournament = Tournament(HexGame, players, boardSize=5, plot=True)
    tournament.run(3)
    tournament.printResults()

    # To not close the plot after the game is finished.
    # Press ctrl+c to close all plots.
    try:
        plt.show()
    except KeyboardInterrupt:
        plt.close()
    '''
    game = HexGame(None, None, 4)
    game.playAction(1)
    print(game.getNNState())
