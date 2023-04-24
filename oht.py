from math import sqrt
import argparse
from pathlib import Path
from time import time, sleep

import numpy as np

from client import ActorClient
from player import Player, NeuralNetPlayer
from hex import HexGame
from neuralnet import loadModel


class MyClient(ActorClient):
    def __init__(self, player: Player, qualify=False):
        assert not qualify, 'Just want to be sure that we are not qualifying by accident'
        super().__init__(qualify=qualify)
        self.player = player

    def handle_series_start(self, unique_id, series_id, player_map, num_games, game_params):
        super().handle_series_start(unique_id, series_id, player_map, num_games, game_params)
        self.gameSize = game_params[0]

    def handle_game_start(self, start_player):
        super().handle_game_start(start_player)
        self.game = HexGame(
            player1=self.player,
            player2=None,
            size=self.gameSize
        )
        self.lastBoard = self.game.board.copy()

        # In the OHT implementation player 1 always goes from top to bottom
        # and player 2 always goes from left to right.
        # This matches our implementation in hex.py
        # However the OHT implementation of Hex both players may start
        # this does not match out implementation of hex, so we need to
        # swap the turn if player 2 starts.
        if start_player == 2:
            self.game.turn = -1

        self.lastActionDone = time()

    def handle_get_action(self, state):
        print(f'Opponent took {time() - self.lastActionDone:.2f}s')
        start = time()
        assert state[0] == 1, 'Only support for local player playing as player 1'

        board = stateToBoard(state)
        opponentAction = actionFromBoardDiff(board, self.lastBoard)
        if opponentAction is not None:
            self.game.playAction(opponentAction)
            self.lastBoard = self.game.board.copy()

        self.game.player1.playAction(self.game)
        action = actionFromBoardDiff(self.game.board, self.lastBoard)
        self.lastBoard = self.game.board.copy()
        print(f'Chose action: {action} ({time() - start:.2f}s)')
        self.lastActionDone = time()
        return action

    def handle_game_end(self, result):
        super().handle_game_end(result)

    def handle_series_end(self, results):
        super().handle_series_end(results)

    def handle_tournament_over(self, score):
        super().handle_tournament_over(score)


def actionFromBoardDiff(board, lastBoard):
    diff = board - lastBoard

    if np.sum(diff) == 0:
        return None

    action = np.where(diff != 0)
    return (int(action[0][0]), int(action[1][0]))


def stateToBoard(state):
    size = int(sqrt(len(state) - 1))
    board = np.array(state[1:]).reshape(size, size)
    board[board == 2] = -1
    return board


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--qualify', action='store_true', help='Run in qualification mode')
    args = parser.parse_args()

    modelPath = Path(args.model)
    model = loadModel(modelPath)
    player = NeuralNetPlayer(model, argmax=True)

    client = MyClient(player, qualify=args.qualify)
    client.run()


if __name__ == "__main__":
    main()
