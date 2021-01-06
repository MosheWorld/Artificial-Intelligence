#===============================================================================
# Imports
#===============================================================================

import abstract
from players import better_h_player, improved_player

# ===============================================================================
# Player
# ===============================================================================

class Player(improved_player.Player, better_h_player.Player):
    def __init__(self, setup_time, player_color, time_per_k_turns, k):
        improved_player.Player.__init__(self, setup_time, player_color, time_per_k_turns, k)
        better_h_player.Player.__init__(self, setup_time, player_color, time_per_k_turns, k)

    def __repr__(self):
        return '{} {}'.format(abstract.AbstractPlayer.__repr__(self), 'improved_better_h')