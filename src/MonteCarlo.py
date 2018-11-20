from __future__ import division
import datetime
from random import choice
from cmath import log, sqrt


class MonteCarlo(object):
    def __init__(self, board, **kwargs):
        # Takes an instance of a Board and optionally some keyword
        # arguments.  Initializes the list of game states and the
        # statistics tables.
        self.board = board
        self.states = []  # Authoritative record of the moves so far.
        seconds = kwargs.get('time', 30)
        self.calculation_time = datetime.timedelta(seconds=seconds)
        self.max_moves = kwargs.get('max_moves', 100)
        self.wins = {}  # dictionaries
        self.plays = {}
        # Larger values of C encourage more exploration of possibilities
        # Smaller values cause the AI to concentrate on known good moves
        self.C = kwargs.get('C', 1.4)

    def print_variables(self):
        print('Seconds: \n{}\nBoard: \n{}\nMax Moves: \n{}\n'.format(
            self.calculation_time, self.board, self.max_moves))

    def update(self, state):
        # Takes a game state, and appends it to the history.
        self.states.append(state)

    def get_play(self):
        # Causes the AI to calculate the best move from the
        # current game state and return it.
        self.max_depth = 0
        state = self.states[-1]
        player = self.board.current_player(state)
        legal = self.board.legal_plays(self.states[:])

        # Bail out early if there is no real choice to be made.
        if not legal:
            return
        if len(legal) == 1:
            return legal[0]

        games = 0
        begin = datetime.datetime.utcnow()
        while datetime.datetime.utcnow() - begin \
                < self.calculation_time:
            self.run_simulation()
            games += 1

        moves_states = [(p, self.board.next_state(state, p)) for p in
                        legal]

        # Display the number of calls of `run_simulation` and the
        # time elapsed.
        print(games, datetime.datetime.utcnow() - begin)

        # Pick the move with the highest percentage of wins.
        percent_wins, move = max(
            (self.wins.get((player, S), 0) /
             self.plays.get((player, S), 1),
             p)
            for p, S in moves_states
        )

        # Display the stats for each possible play.
        for x in sorted(
                ((100 * self.wins.get((player, S), 0) /
                  self.plays.get((player, S), 1),
                  self.wins.get((player, S), 0),
                  self.plays.get((player, S), 0), p)
                 for p, S in moves_states),
                reverse=True
        ):
            print("{3}: {0:.2f}% ({1} / {2})".format(*x))

        print("Maximum depth searched:", self.max_depth)

        return move

    def run_simulation(self):
        # Plays out a "random" game from the current position,
        # then updates the statistics tables with the result.
        # A bit of an optimization here, so we have a local
        # variable lookup instead of an attribute access each loop.
        plays, wins = self.plays, self.wins

        visited_states = set()
        states_copy = self.states[:]
        state = states_copy[-1]  # -1 means the last element in the list
        player = self.board.current_player(state)

        expand = True
        for t in range(1, self.max_moves + 1):
            legal = self.board.legal_plays(states_copy)
            moves_states = [(p, self.board.next_state(state, p)) for p
                            in legal]

            if all(plays.get((player, S)) for p, S in moves_states):
                # If we have stats on the legal moves here, use them.
                log_total = log(
                    sum(plays[(player, S)] for p, S in moves_states))
                value, move, state = max(
                    ((wins[(player, S)] / plays[(player, S)]) +
                     self.C * sqrt(log_total / plays[(player, S)]), p,
                     S)
                    for p, S in moves_states
                )
            else:
                # Otherwise, just make an arbitrary decision.
                move, state = choice(moves_states)

            states_copy.append(state)

            # `player` here and below refers to the player
            # who moved into that particular state.
            if expand and (player, state) not in plays:
                expand = False
                plays[(player, state)] = 0
                wins[(player, state)] = 0
                if t > self.max_depth:
                    self.max_depth = t

            visited_states.add((player, state))

            player = self.board.current_player(state)
            winner = self.board.winner(states_copy)
            if winner:
                break

        for player, state in visited_states:
            if (player, state) not in plays:
                continue
            plays[(player, state)] += 1
            if player == winner:
                wins[(player, state)] += 1
