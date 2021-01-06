import threading

from run_game import GameRunner
from matplotlib import pyplot as plt

from checkers.consts import RED_PLAYER, BLACK_PLAYER, TIE

class ExperimentManager(object):
    def __init__(self):
        self.scores_path = './scores_table.csv'
        self.experiment_path = './experiment.csv'
        self.times = [2, 10, 50]
        self.players = ['simple_player', 'better_h_player', 'improved_player', 'improved_better_h_player']
        self.games = []
        self.player_time_recorder = {player: {t: 0 for t in self.times} for player in self.players}

    def execute(self):
        for time in self.times:
            for i in range(len(self.players)):
                for j in range(len(self.players)):
                    if i == j:
                        continue

                    self.run_game(self.players[i], self.players[j], time)

        self.create_chart()

    def execute_multithreaded(self):
        threads = []

        for time in self.times:
            for i in range(len(self.players)):
                for j in range(len(self.players)):
                    if i == j:
                        continue

                    t = threading.Thread(target=self.run_game, args=[self.players[i], self.players[j], time])
                    threads.append(t)
                    t.start()

        for t in threads:
            t.join()
        self.create_chart()

    def output_text_to_file(self, filename, single_game):
        joined_str = ','.join(map(lambda value: str(value), single_game))
        with open(filename, 'a') as f:
            f.write(joined_str)
            f.write("\n")

    def calculate_scores(self, winner):
        if winner == TIE:
            red_score, black_score = 0.5, 0.5
        elif winner[0] == RED_PLAYER:
            red_score, black_score = 1, 0
        elif winner[0] == BLACK_PLAYER:
            red_score, black_score = 0, 1
        return red_score, black_score

    def run_game(self, player1, player2, time):
        winner = GameRunner(2, time, 5, 'n', player1, player2).run()

        red_score, black_score = self.calculate_scores(winner)
        single_game = [player1, player2, time, red_score, black_score]

        self.player_time_recorder[player1][time] += float(red_score)
        self.player_time_recorder[player2][time] += float(black_score)

        self.games.append(single_game.copy())
        self.output_text_to_file(self.experiment_path, single_game)

    def create_chart(self):
        scores_table = []
        x = [int(t) for t in self.times]

        plt.figure()
        plt.title('Scores as a function of the time.')

        for player in self.players:
            y = [self.player_time_recorder[player][t] for t in self.times]
            scores_table.append(player + ',' + ','.join(map(lambda value: str(value), y)) + '\n')
            plt.plot(x, y, '.-', label=player)

        plt.legend()
        plt.show()

        with open(self.scores_path, 'w') as f:
            headers = 'player_name,' + ','.join([f't={time}' for time in self.times]) + '\n'
            f.write(headers)
            for line in scores_table:
                f.write(line)


if __name__ == '__main__':
    experiment_manager = ExperimentManager()
    experiment_manager.execute_multithreaded()