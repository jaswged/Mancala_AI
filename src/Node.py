import numpy as np
import math
import copy


class Node:
    def __init__(self, game, move, parent=None):
        self.game = game  # state s
        self.move = move  # action index
        self.has_children = False
        self.children = {}
        self.parent = parent
        self.player_turn = game.player
        self.policy = np.zeros([14], dtype=np.float32)
        self.child_total_value = np.zeros([14], dtype=np.float32)
        self.child_number_visits = np.zeros([14], dtype=np.float32)
        self.legal_moves = []

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.move] \
            if self.parent is not None else 0

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.move]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move] = value

    def child_reward_q(self):
        return self.child_total_value / (1 + self.child_number_visits)

    def child_upper_confidence_bound(self):
        return math.sqrt(self.number_visits) * (
                abs(self.policy) / (1 + self.child_number_visits))

    def best_child(self):
        if self.legal_moves != []:
            bestmove = self.child_reward_q() + \
                       self.child_upper_confidence_bound()
            bestmove = self.legal_moves[
                np.argmax(bestmove[self.legal_moves])]
        else:
            bestmove = np.argmax(self.child_reward_q() +
                                 self.child_upper_confidence_bound())
        return bestmove

    def select_leaf(self):
        current = self
        while current.has_children:
            best_move = current.best_child()
            current = current.maybe_add_child(best_move)
        return current

    def add_dirichlet_noise(self, action_idxs, child_priors):
        # select only legal moves entries in child_priors array
        valid_child_priors = child_priors[action_idxs]
        valid_child_priors = 0.75 * valid_child_priors + 0.25 * np.random.dirichlet(
            np.zeros([len(valid_child_priors)], dtype=np.float32) + 192)

        child_priors[action_idxs] = valid_child_priors
        return child_priors

    def expand(self, policy):
        """Expand only nodes that result from legal moves, mask illegal
        moves and add Dirichlet noise to prior probabilities of root."""
        legal_moves = self.game.get_legal_moves()
        self.has_children = not legal_moves == []
        child_priors = policy

        self.legal_moves = legal_moves
        # mask all illegal actions
        child_priors[[i for i in range(len(policy)) if
                     i not in legal_moves]] = 0.000000000

        # add dirichlet noise to child_priors in root node
        if self.parent is not None and self.parent.parent is None:
            child_priors = self.add_dirichlet_noise(legal_moves,
                                                    child_priors)

        self.policy = child_priors

    def maybe_add_child(self, move):
        if move not in self.children:
            # make a copy of the board
            copy_board = copy.deepcopy(self.game)
            # take the action on the copied board
            copy_board.process_move(move)
            self.children[move] = Node(copy_board, move, parent=self)
        return self.children[move]

    def backup(self, value_estimate: float):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            if current.game.player == 1:
                # value estimate +1 = O wins
                current.total_value += (1 * value_estimate)
            else:
                current.total_value += (-1 * value_estimate)
            current = current.parent
