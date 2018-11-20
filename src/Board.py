class Board(object):
    def __init__(self):
        self.is_player_1s_turn = True
        self.board_state = self.start()

    # State data structure must be hashable such as flat tuple
    def start(self):
        # Returns a representation of the starting state of the game
        #   4  4  4 | 4  4  4        12 11 10 9  8  7
        # 0                   0   13                  6 1st players home
        #   4  4  4 | 4  4  4        0  1  2  3  4  5
        return [4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0]

    def __str__(self):
        return "Board object for game"

    def current_player(self, state):
        #  Takes the game state and returns the current player's number.
        pass

    def next_state(self, state, play):
        # Takes the game state, and the move to be applied.
        # Returns the new game state.
        pebbles = state[play]
        print("Pebbles to move {}".format(pebbles))
        for x in range(pebbles):
            position_to_increment = 2  # todo play += 1
            holes_value = state[position_to_increment]
            state[position_to_increment] = holes_value
        pass

    def legal_plays(self, state_history, ):
        # Takes a sequence of game states representing the full
        # game history, and returns the full list of moves that
        # are legal plays for the current player.
        pass

    def winner(self, state_history):
        # Takes a sequence of game states representing the full
        # game history.  If the game is now won, return the player
        # number.  If the game is still ongoing, return zero.  If
        # the game is tied, return a different distinct value, e.g. -1.
        pass
