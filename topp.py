import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from reinforcement import ReinforcementLearner
from neuralnet import createModel, loadModel
from tournament import Tournament
from hex import HexGame
from player import NeuralNetPlayer


class TournamentOfProgressivePolicies:
    def __init__(self, boardSize: int, models: dict):
        self.boardSize = boardSize
        self.models = models
        self.players = [NeuralNetPlayer(model, name=f'nn {iteration}', epsilon=0.05, argmax=False) for iteration, model in self.models.items()]
        self.tournament = Tournament(HexGame, self.players, boardSize=self.boardSize, plot=False)

    @classmethod
    def FromTraining(cls, iterations, numPolicies, boardSize, saveInterval, miniBatchSize):
        initialModel = createModel(size=boardSize)
        replayBufferSize = boardSize*boardSize*50

        learner = ReinforcementLearner(
            epsilonMultiplier=0.995,
            avgGameTime=10,
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
            print(f'Iteration {iteration}')
            learner.oneIteration()
            if iteration in tournamentIterations:
                models[iteration] = tf.keras.models.clone_model(learner.model)

        return cls(boardSize, models)

    @classmethod
    def LoadTournament(cls, path):
        path = Path(path)
        tournamentInfo = json.loads((path / 'tournamentInfo.json').read_text())

        boardSize = tournamentInfo['boardSize']
        models = {iteration: loadModel(path / f'model{iteration}') for iteration in tournamentInfo['iterations']}

        return cls(boardSize, models)

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

    def run(self):
        self.tournament.run(10)
        for player in self.players:
            # TESTING:
            game = HexGame(None, None, self.boardSize)
            state = game.getNNState()
            print(player.name)
            print(player.model(state))
            result = self.tournament.getPlayerResults(player)
            print(f'{player.name}: wins {result[0]}, losses {result[1]}')
            print(f'win rate: {result[0] / (result[0] + result[1])}')


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', type=str, help='Train a model and save to path (e.g. --train topp/tournament1)')
    group.add_argument('--load', type=str, help='Load a model from path (e.g. --load topp/tournament1)')

    parser.add_argument('--iterations', type=int, default=20, help='Number of iterations to train for')
    parser.add_argument('--num_policies', type=int, default=6, help='Number of policies to train')
    parser.add_argument('--board_size', type=int, default=3, help='Board size')
    parser.add_argument('--save_interval', type=int, default=10, help='Save interval')
    parser.add_argument('--mini_batch_size', type=int, default=32, help='Mini batch size')

    args = parser.parse_args()

    if args.train:
        path = args.train
        print(f'Training and saving to {path}...')
        topp = TournamentOfProgressivePolicies.FromTraining(
            iterations=args.iterations,
            numPolicies=args.num_policies,
            boardSize=args.board_size,
            saveInterval=args.save_interval,
            miniBatchSize=args.mini_batch_size,
        )
        topp.save(path)

    if args.load:
        path = args.load
        print(f'Loading from {path}...')
        topp = TournamentOfProgressivePolicies.LoadTournament(path)

    topp.run()
    try:
        plt.show()
    except KeyboardInterrupt:
        plt.close()


if __name__ == "__main__":
    main()
