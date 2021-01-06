#===============================================================================
# Imports
#===============================================================================

import abstract
from players import simple_player

from math import sqrt
from collections import defaultdict

from utils import INFINITY
from checkers.consts import EM, PAWN_COLOR, KING_COLOR, OPPONENT_COLOR, MAX_TURNS_NO_JUMP, BACK_ROW, RP, RK, BK, BP, MY_COLORS

# ===============================================================================
# Globals
# ===============================================================================

PAWN_WEIGHT = 10
KING_WEIGHT = 15

SIDE_WEIGHT = 4
BACK_ROWֹֹ_WEIGHT = 5
CLOSENESS_WEIGHT = 5

# ===============================================================================
# Helper Functions
# ===============================================================================

# We want our pawns to stay on the back row as long as possible so it won't allow the opponent to get kings.
# We will give bonus for this pawns and if we have full row we will give extra bonus.
def back_row_score(state, player):
    score = 0
    for (row, col), piece in state.board.items():
        if row == BACK_ROW[OPPONENT_COLOR[player]] and piece in MY_COLORS[player]:
            score += BACK_ROWֹֹ_WEIGHT

    if score == BACK_ROWֹֹ_WEIGHT * 4:
        score += BACK_ROWֹֹ_WEIGHT

    return score

# We want our pawns to stay on the sides of the board so it we'll get less chance to get jumped on.
# This function returns the number of pieces on the last and first colums (the sides) of our player.
# Then we take this number (from previous line calculation) and subtract it of the opponent result.
# We want this result to be as large as possible, which means it's better state for us.
def pieces_on_sides_scores(board, color):
    my_pieces, opponent_pieces = 0, 0
    opponent_color = OPPONENT_COLOR[color]

    for square in board:
        if square[1] == 0 or square[1] == 7:
            if board[square] == PAWN_COLOR[color] or board[square] == KING_COLOR[color]:
                my_pieces += 1

            if board[square] == PAWN_COLOR[opponent_color] or board[square] == KING_COLOR[opponent_color]:
                opponent_pieces += 1

    return my_pieces - opponent_pieces

# A function that calculates the mass of the board for each player.
def calculate_mass(board):
    red_pieces, black_pieces = 0, 0
    red_mass_counter = [0, 0]
    black_mass_counter = [0, 0]

    for square in board:
        if board[square] == RP or board[square] == RK:
            red_mass_counter[0] += square[0]
            red_mass_counter[1] += square[1]
            red_pieces += 1
        if board[square] == BP or board[square] == BK:
            black_mass_counter[0] += square[0]
            black_mass_counter[1] += square[1]
            black_pieces += 1

    return red_pieces, black_pieces, red_mass_counter, black_mass_counter

# This function returns the euclidian distance between the two mass centers of pawns.
# We want our pawns to stay close.
def group_centers_mass(board):
    red_pieces, black_pieces, red_mass_counter, black_mass_counter = calculate_mass(board)

    red_mass_counter[0] /= red_pieces
    red_mass_counter[1] /= red_pieces
    black_mass_counter[0] /= black_pieces
    black_mass_counter[1] /= black_pieces

    return sqrt((black_mass_counter[0] - red_mass_counter[0]) ** 2 + (black_mass_counter[1] - red_mass_counter[1]) ** 2)

# ===============================================================================
# Player
# ===============================================================================

class Player(simple_player.Player):
    def __init__(self, setup_time, player_color, time_per_k_turns, k):
        simple_player.Player.__init__(self, setup_time, player_color, time_per_k_turns, k)

    def utility(self, state):
        if len(state.get_possible_moves()) == 0:
            return INFINITY if state.curr_player != self.color else -INFINITY
        if state.turns_since_last_jump >= MAX_TURNS_NO_JUMP:
            return 0

        piece_counts = defaultdict(lambda: 0)
        for loc_val in state.board.values():
            if loc_val != EM:
                piece_counts[loc_val] += 1

        opponent_color = OPPONENT_COLOR[self.color]
        my_u = ((PAWN_WEIGHT * piece_counts[PAWN_COLOR[self.color]]) + (KING_WEIGHT * piece_counts[KING_COLOR[self.color]]))
        op_u = ((PAWN_WEIGHT * piece_counts[PAWN_COLOR[opponent_color]]) + (KING_WEIGHT * piece_counts[KING_COLOR[opponent_color]]))

        if my_u == 0:
            # I have no tools left
            return -INFINITY
        elif op_u == 0:
            # The opponent has no tools left
            return INFINITY
        else:
            back_row_weight = back_row_score(state, self.color) 
            distance = group_centers_mass(state.board)
            sides = pieces_on_sides_scores(state.board, self.color)
            return my_u - op_u + back_row_weight - CLOSENESS_WEIGHT * distance + SIDE_WEIGHT * sides

    def __repr__(self):
        return '{} {}'.format(abstract.AbstractPlayer.__repr__(self), 'better_h')