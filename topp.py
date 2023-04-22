import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from hex import HexGame
from neuralnet import createModel, loadModel
from player import NeuralNetPlayer
from reinforcement import ReinforcementLearner
from tournament import Tournament


class TournamentOfProgressivePolicies:
    def __init__(self, boardSize: int, models: dict, plot=False):
        self.boardSize = boardSize
        self.models = models
        self.players = [NeuralNetPlayer(model, name=f'nn {iteration}', epsilon=0.05, argmax="Probs") for iteration, model in self.models.items()]
        self.tournament = Tournament(HexGame, self.players, boardSize=self.boardSize, plot=plot)

    @classmethod
    def FromTraining(cls, gameTime, iterations, numPolicies, boardSize, saveInterval, miniBatchSize, plot=False):

        initialModel = createModel(size=boardSize)
        replayBufferSize = boardSize*boardSize*50

        learner = ReinforcementLearner(
            epsilonMultiplier=0.99,
            avgGameTime=gameTime,
            saveInterval=saveInterval,
            miniBatchSize=miniBatchSize,
            boardSize=boardSize,
            model=initialModel,
            replayBufferSize=replayBufferSize
        )

        tournamentIterations = [(iterations - 1) * i // (numPolicies - 1) for i in range(numPolicies)]
        print(f'Tournament iterations: {tournamentIterations}')

        models = {}
        for iteration in range(iterations):
            learner.oneIteration()
            if iteration in tournamentIterations:
                models[iteration] = tf.keras.models.clone_model(learner.model)

        return cls(boardSize, models, plot=plot)

    @classmethod
    def LoadTournament(cls, path, plot=False):
        path = Path(path)
        tournamentInfo = json.loads((path / 'tournamentInfo.json').read_text())

        boardSize = tournamentInfo['boardSize']
        models = {iteration: loadModel(path / f'model{iteration}') for iteration in tournamentInfo['iterations']}

        return cls(boardSize, models, plot=plot)

    def save(self, path):
        path = Path(path)

        if not os.path.exists(path):
            os.makedirs(path)

        tournamentInfo = {
            'boardSize': self.boardSize,
            'iterations': list(self.models.keys()),
        }

        with open(path / 'tournamentInfo.json', 'w') as f:
            f.write(json.dumps(tournamentInfo))

        for iteration, model in self.models.items():
            model.save(path / f'model{iteration}')

    def run(self, tournamentRounds=1):
        self.tournament.run(tournamentRounds)
        for player in self.players:
            result = self.tournament.getPlayerResults(player)
            print(f'{player.name}: wins {result[0]}, losses {result[1]}')
            print(f'win rate: {result[0] / (result[0] + result[1])}')


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', type=str, help='Train a model and save to path (e.g. --train topp/tournament1)')
    group.add_argument('--load', type=str, help='Load a model from path (e.g. --load topp/tournament1)')

    parser.add_argument('--game_time', type=int, default=30, help='Average game time')
    parser.add_argument('--iterations', type=int, default=20, help='Number of iterations to train for')
    parser.add_argument('--num_policies', type=int, default=6, help='Number of policies to train')
    parser.add_argument('--board_size', type=int, default=4, help='Board size')
    parser.add_argument('--save_interval', type=int, default=10, help='Save interval')
    parser.add_argument('--mini_batch_size', type=int, default=8, help='Mini batch size')
    parser.add_argument('--tournament_rounds', type=int, default=10, help='Number of rounds to play in tournament')
    parser.add_argument('--plot', action='store_true', help='Plot the tournament')

    args = parser.parse_args()

    if args.train:
        path = args.train
        print(f'Training and saving to {path}...')
        topp = TournamentOfProgressivePolicies.FromTraining(
            gameTime=args.game_time,
            iterations=args.iterations,
            numPolicies=args.num_policies,
            boardSize=args.board_size,
            saveInterval=args.save_interval,
            miniBatchSize=args.mini_batch_size,
            plot=args.plot
        )
        topp.save(path)

    if args.load:
        path = args.load
        print(f'Loading from {path}...')
        topp = TournamentOfProgressivePolicies.LoadTournament(path, plot=args.plot)

    topp.run(args.tournament_rounds)
    try:
        plt.show()
    except KeyboardInterrupt:
        plt.close()


if __name__ == "__main__":
    main()
