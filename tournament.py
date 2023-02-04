from hex import HexGame

'''
Restores best model from file and returns it as a player
'''
def restore_best_player():
    pass

'''
Saves player to file as a model
'''
def save_player(player):
    pass

class Tournament:
    def __init__(self, player1, player2, rounds):
        self.player1 = player1
        self.player2 = player2
        self.player1_wins = 0
        self.player2_wins = 0
        self.draws = 0

    def play_round(self):
        pass

    def get_results(self):
        pass