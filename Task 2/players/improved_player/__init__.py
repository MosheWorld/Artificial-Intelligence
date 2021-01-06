# ===============================================================================
# Imports
# ===============================================================================

import time
import abstract
from players import simple_player

from utils import MiniMaxWithAlphaBetaPruning, INFINITY, run_with_limited_time, ExceededTimeError

# ===============================================================================
# Globals
# ===============================================================================

PAWN_WEIGHT = 1
KING_WEIGHT = 1.5

# ===============================================================================
# Player
# ===============================================================================

class Player(simple_player.Player):
    def __init__(self, setup_time, player_color, time_per_k_turns, k):
        simple_player.Player.__init__(self, setup_time, player_color, time_per_k_turns, k)

    def get_move(self, game_state, possible_moves):
        self.clock = time.process_time()
        # We want the first turns of the k turns to get more time to think,
        # so it would go deeper on the minmax tree
        # and the rest of the k turns won't have to think more
        self.time_for_current_move = self.time_remaining_in_round / 2 - 0.05
        if len(possible_moves) == 1:
            return possible_moves[0]

        current_depth = 1
        prev_alpha = -INFINITY

        # Choosing an arbitrary move in case Minimax does not return an answer:
        best_move = possible_moves[0]

        # Initialize Minimax algorithm, still not running anything
        minimax = MiniMaxWithAlphaBetaPruning(self.utility, self.color, self.no_more_time, self.selective_deepening_criterion)

        # Iterative deepening until the time runs out.
        while True:
            print('going to depth: {}, remaining time: {}, prev_alpha: {}, best_move: {}'.format(current_depth, self.time_for_current_move - (time.process_time() - self.clock), prev_alpha, best_move))

            try:
                (alpha, move), run_time = run_with_limited_time(minimax.search, (game_state, current_depth, -INFINITY, INFINITY, True), {}, self.time_for_current_move - (time.process_time() - self.clock))
            except (ExceededTimeError, MemoryError):
                print('no more time, achieved depth {}'.format(current_depth))
                break

            if self.no_more_time():
                print('no more time')
                break

            prev_alpha = alpha
            best_move = move

            if alpha == INFINITY:
                print('the move: {} will guarantee victory.'.format(best_move))
                break

            if alpha == -INFINITY:
                print('all is lost')
                break

            current_depth += 1

        if self.turns_remaining_in_round == 1:
            self.turns_remaining_in_round = self.k
            self.time_remaining_in_round = self.time_per_k_turns
        else:
            self.turns_remaining_in_round -= 1
            self.time_remaining_in_round -= (time.process_time() - self.clock)

        return best_move

    # In this function, we choose when to dive into the minmax tree.
    # Here, if we have more than one possible jump, we want the minmax to choose which jump is better
    # for the rest of the game
    def selective_deepening_criterion(self, state):
        counter = 0
        for move in state.get_possible_moves():
            if len(move.jumped_locs) > 0:
                counter += 1
        return counter > 1

    def __repr__(self):
        return '{} {}'.format(abstract.AbstractPlayer.__repr__(self), 'improved')