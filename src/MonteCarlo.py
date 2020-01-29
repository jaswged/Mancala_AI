from __future__ import division
import datetime
from random import choice
from cmath import log, sqrt
import pickle
import numpy as np
import torch
import os
import logging
import copy
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


def policy_value_fn(board):
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state"""
    # return uniform probabilities and 0 score for pure MCTS
    action_probs = np.ones(len(board.availables))/len(board.availables)
    return zip(board.availables, action_probs), 0


class Node():
    def __init__(self, prior_policy, parent=None):
        self.parent = parent
        self.children = {}
        self.child_priors = np.zeros([6], dtype=np.float32)
        self.child_total_value = np.zeros([6], dtype=np.float32)
        self.child_number_visits = np.zeros([6], dtype=np.float32)
        self.action_indexes = []
        self.expected_reward = 0
        self.n_visits = 0
        self.upper_confidence_bound = 0
        self.policy = prior_policy

    def expand(self, action_priors):
        """Expand tree by and create new child nodes
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function."""
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = Node(self, prob)

    def select(self, c_puct):
        """Select action from children that has highest expected reward Q
        Return: A tuple of (action, next_node)"""
        return max(self.children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective. """
        # Increment visit.
        self.n_visits += 1
        # Update expectedReward Q, a running average of values for all visits.
        self.expected_reward += 1.0 * (
                    leaf_value - self.expected_reward) / self.n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score."""
        self.upper_confidence_bound = (c_puct * self.policy *
                                       np.sqrt(self.parent.n_visits) / (
                                                   1 + self.n_visits))
        return self.expected_reward + self.upper_confidence_bound

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self.children == {}

    def is_root(self):
        return self.parent is None

    # @property #TD
    # def number_visits(self):
    #     return self.parent.child_number_visits[self.move]
    #
    # @number_visits.setter # TD
    # def number_visits(self, value):
    #     self.parent.child_number_visits[self.move] = value
    #
    # @property
    # def total_value(self):
    #     return self.parent.child_total_value[self.move]
    #
    # @total_value.setter
    # def total_value(self, value):
    #     self.parent.child_total_value[self.move] = value
    #
    # def child_Q(self):
    #     return self.child_total_value / (1 + self.child_number_visits)
    #
    # def child_U(self):
    #     return math.sqrt(self.number_visits) * (
    #             abs(self.child_priors) / (1 + self.child_number_visits))
    #
    # def best_child(self):
    #     if self.action_idxes != []:
    #         bestmove = self.child_Q() + self.child_U()
    #         bestmove = self.action_idxes[
    #             np.argmax(bestmove[self.action_idxes])]
    #     else:
    #         bestmove = np.argmax(self.child_Q() + self.child_U())
    #     return bestmove
    #
    # def select_leaf(self):
    #     current = self
    #     while current.is_expanded:
    #         best_move = current.best_child()
    #         current = current.add_child(best_move)
    #     return current
    #
    # def add_dirichlet_noise(self, action_idxs, child_priors):
    #     # select only legal moves entries in child_priors array
    #     valid_child_priors = child_priors[action_idxs]
    #     valid_child_priors = 0.75 * valid_child_priors + 0.25 * np.random.dirichlet(
    #         np.zeros([len(valid_child_priors)], dtype=np.float32) + 192)
    #
    #     child_priors[action_idxs] = valid_child_priors
    #     return child_priors
    #
    #
    # def expand(self, child_priors):
    #     """Expand only nodes that result from legal moves, mask illegal
    #     moves and add Dirichlet noise to prior probabilities of root node."""
    #     action_idxs = self.game.actions();
    #     self.is_expanded = False if action_idxs == [] else True
    #
    #     childPriors = child_priors
    #
    #     self.action_indexes = action_idxs
    #     childPriors[[i for i in range(len(child_priors)) if
    #                 i not in action_idxs]] = 0.000000000  # mask all illegal actions
    #
    #     if self.parent.parent is None:  # add dirichlet noise to child_priors in root node
    #         childPriors = self.add_dirichlet_noise(action_idxs, childPriors)
    #     self.child_priors = childPriors
    #
    # def add_child(self, move):
    #     if move not in self.children:
    #         copy_board = copy.deepcopy(self.game)  # make copy of board
    #         # take the action on the copied board
    #         copy_board = board.process_static_move(copy_board, move)
    #         self.children[move] = UCTNode(copy_board, move, parent=self)
    #     return self.children[move]
    #
    # def backup(self, value_estimate: float):
    #     current = self
    #     while current.parent is not None:
    #         current.number_visits += 1
    #         if current.game.player == 1:  # same as current.parent.game.player = 0
    #             current.total_value += (1 * value_estimate)  # value estimate +1 = O wins
    #         elif current.game.player == 0:  # same as current.parent.game.player = 1
    #             current.total_value += (-1 * value_estimate)
    #         current = current.parent


class MonteCarlo(object):
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """ policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
            c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.  """
        self.root = Node(None, 1.0)
        self.policy = policy_value_fn
        self.c_puct = c_puct
        self.n_playout = n_playout

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self.root
        while 1:
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self.c_puct)
            state.do_move(action)

        action_probs, leaf_value = self.policy(state)
        # Check for end of game
        end, winner = state.get_winner()
        if not end:
            node.expand(action_probs)
        else:
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie TODO
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_whose_turn() else -1.0
                )

            # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

        # Evaluate the leaf node by random rollout
        # leaf_value = self.evaluate_rollout(state)
        # Update value and visit count of nodes in this traversal.
        # node.update_recursive(-leaf_value)

    # def _evaluate_rollout(self, state, limit=1000):
    #     """Use the rollout policy to play until the end of the game,
    #     returning +1 if the current player wins, -1 if the opponent wins,
    #     and 0 if it is a tie.  """
    #     player = state.get_current_player()
    #     for i in range(limit):
    #         end, winner = state.game_end()
    #         if end:
    #             break
    #         action_probs = rollout_policy_fn(state)
    #         max_action = max(action_probs, key=itemgetter(1))[0]
    #         state.do_move(max_action)
    #     else:
    #         # If no break from the loop, issue a warning.
    #         print("WARNING: rollout reached move limit")
    #     if winner == -1:  # TODO tie
    #         return 0
    #     else:
    #     return 1 if winner == player else -1

    def get_move(self, state, temp=1e-3):
        """Run all playouts sequentially and return the available
        actions and their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1]
        controls the level of exploration """
        for n in range(self.n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node.n_visits)
                      for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree. """
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = Node(None, 1.0)

    def __str__(self):
        return "Monte Carlo Tree Search"


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MonteCarlo(policy_value_function, c_puct, n_playout)
        self.is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.width*board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            if self.is_selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)
#                location = board.move_to_location(move)
#                print("AI move: %d,%d\n" % (location[0], location[1]))

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)


class MCTSPurePlayer(object):
    """AI player based on MCTS"""

    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = MonteCarlo(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "Monte Carlo Player {}".format(self.player)


# def run_Monte_Carlo(args, start_idx=0, iteration=0):
#     net_to_play = "%s_iter%d.pth.tar" % (
#         args.neural_net_name, iteration)
#     net = NeuralNet()
#     cuda = torch.cuda.is_available()
#     if cuda:
#         net.cuda()
#
#     if args.MCTS_num_processes > 1:
#         logger.info("Preparing model for multi-process MCTS...")
#         mp.set_start_method("spawn", force=True)
#         net.share_memory()
#         net.eval()
#
#         current_net_filename = os.path.join("./model_data/", \
#                                             net_to_play)
#         if os.path.isfile(current_net_filename):
#             checkpoint = torch.load(current_net_filename)
#             net.load_state_dict(checkpoint['state_dict'])
#             logger.info("Loaded %s model." % current_net_filename)
#         else:
#             torch.save({'state_dict': net.state_dict()},
#                        os.path.join("./model_data/", \
#                                     net_to_play))
#             logger.info("Initialized model.")
#
#         processes = []
#         if args.MCTS_num_processes > mp.cpu_count():
#             num_processes = mp.cpu_count()
#             logger.info(
#                 "Required number of processes exceed number of CPUs! Setting MCTS_num_processes to %d" % num_processes)
#         else:
#             num_processes = args.MCTS_num_processes
#
#         logger.info("Spawning %d processes..." % num_processes)
#         with torch.no_grad():
#             for i in range(num_processes):
#                 p = mp.Process(target=MCTS_self_play, args=(
#                     net, args.num_games_per_MCTS_process, start_idx, i,
#                     args, iteration))
#                 p.start()
#                 processes.append(p)
#             for p in processes:
#                 p.join()
#         logger.info("Finished multi-process MCTS!")
#
#     elif args.MCTS_num_processes == 1:
#         logger.info("Preparing model for MCTS...")
#         net.eval()
#
#         current_net_filename = os.path.join("./model_data/", \
#                                             net_to_play)
#         if os.path.isfile(current_net_filename):
#             checkpoint = torch.load(current_net_filename)
#             net.load_state_dict(checkpoint['state_dict'])
#             logger.info("Loaded %s model." % current_net_filename)
#         else:
#             torch.save({'state_dict': net.state_dict()},
#                        os.path.join("./model_data/", net_to_play))
#             logger.info("Initialized model.")
#
#         with torch.no_grad():
#             MCTS_self_play(net, args.num_games_per_MCTS_process,
#                            start_idx, 0, args, iteration)
#         logger.info("Finished MCTS!")


def save_as_pickle(filename, data):
    complete_name  = os.path.join("./datasets/", filename)
    with open(complete_name, 'wb') as output:
        pickle.dump(data, output)


def load_pickle(filename):
    complete_name = os.path.join("./datasets/", filename)
    with open(complete_name, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data
