import numpy as np
import random
import logging
from game import Game

NEIGHBORS = [(0, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1)]

class HexGame(Game):
    def __init__(self, player1, player2, size=4):
        self.player1 = player1
        self.player2 = player2
        self.size = size
        self.turn = 1
        self.board_shape = (size, size)
        self.board = np.zeros(shape=self.board_shape, dtype=np.int8)
        self.log = logging.getLogger(__name__)

    def createInitialState(self):
        return self.board

    def getState(self):
        return self.board

    def getActions(self):
        actions = []
        for x in range(self.board.shape[0]):
            for y in range(self.board.shape[1]):
                if self.board[x, y] == 0:
                    actions.append((x, y))
        return actions

    def playAction(self, action):
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
        game_copy = HexGame(self.player1, self.player2, size=self.size)
        game_copy.turn = self.turn
        game_copy.board = self.board.copy()
        return game_copy

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
            neighbor_pos = (position[0] + neighbor[0], position[1] + neighbor[1])

            if self.searchForWin(player, neighbor_pos, visited):
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

if __name__ == '__main__':
    game = HexGame()
    while not game.isTerminal():
        actions = game.getActions()
        print(actions)
        action = random.choice(actions)
        game.playAction(action)
        print(game.board, end='\n\n')

    print(f'{game.getResult() = }')