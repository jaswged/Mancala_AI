from __future__ import division
import datetime
from random import choice
from cmath import log, sqrt
import pickle
import numpy as np
import torch
import datetime
import datetime
import logging
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)


class Node():
    def __init__(self, game, move, parent=None):
        self.game = game  # state s
        self.move = move  # action index
        self.is_expanded = False
        self.parent = parent
        self.children = {}
        self.child_priors = np.zeros([6], dtype=np.float32)
        self.child_total_value = np.zeros([6], dtype=np.float32)
        self.child_number_visits = np.zeros([6], dtype=np.float32)
        self.action_indexes = []

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.move]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move] = value

    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        return math.sqrt(self.number_visits) * (
                abs(self.child_priors) / (1 + self.child_number_visits))

    def best_child(self):
        if self.action_idxes != []:
            bestmove = self.child_Q() + self.child_U()
            bestmove = self.action_idxes[
                np.argmax(bestmove[self.action_idxes])]
        else:
            bestmove = np.argmax(self.child_Q() + self.child_U())
        return bestmove

    def select_leaf(self):
        current = self
        while current.is_expanded:
            best_move = current.best_child()
            current = current.add_child(best_move)
        return current

    def add_dirichlet_noise(self, action_idxs, child_priors):
        # select only legal moves entries in child_priors array
        valid_child_priors = child_priors[action_idxs]
        valid_child_priors = 0.75 * valid_child_priors + 0.25 * np.random.dirichlet(
            np.zeros([len(valid_child_priors)], dtype=np.float32) + 192)

        child_priors[action_idxs] = valid_child_priors
        return child_priors


    def expand(self, child_priors):
        """Expand only nodes that result from legal moves, mask illegal
        moves and add Dirichlet noise to prior probabilities of root node."""
        action_idxs = self.game.actions();
        self.is_expanded = False if action_idxs == [] else True

        childPriors = child_priors

        self.action_indexes = action_idxs
        childPriors[[i for i in range(len(child_priors)) if
                    i not in action_idxs]] = 0.000000000  # mask all illegal actions

        if self.parent.parent is None:  # add dirichlet noise to child_priors in root node
            childPriors = self.add_dirichlet_noise(action_idxs, childPriors)
        self.child_priors = childPriors

    def add_child(self, move):
        if move not in self.children:
            copy_board = copy.deepcopy(self.game)  # make copy of board
            # take the action on the copied board
            copy_board = board.process_static_move(copy_board, move)
            self.children[move] = UCTNode(copy_board, move, parent=self)
        return self.children[move]

    def backup(self, value_estimate: float):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            if current.game.player == 1:  # same as current.parent.game.player = 0
                current.total_value += (1 * value_estimate)  # value estimate +1 = O wins
            elif current.game.player == 0:  # same as current.parent.game.player = 1
                current.total_value += (-1 * value_estimate)
            current = current.parent

class DummyNode(object):
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)


def run_Monte_Carlo(args, start_idx=0, iteration=0):
    net_to_play = "%s_iter%d.pth.tar" % (
    args.neural_net_name, iteration)
    net = NeuralNet()
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()

    if args.MCTS_num_processes > 1:
        logger.info("Preparing model for multi-process MCTS...")
        mp.set_start_method("spawn", force=True)
        net.share_memory()
        net.eval()

        current_net_filename = os.path.join("./model_data/", \
                                            net_to_play)
        if os.path.isfile(current_net_filename):
            checkpoint = torch.load(current_net_filename)
            net.load_state_dict(checkpoint['state_dict'])
            logger.info("Loaded %s model." % current_net_filename)
        else:
            torch.save({'state_dict': net.state_dict()},
                       os.path.join("./model_data/", \
                                    net_to_play))
            logger.info("Initialized model.")

        processes = []
        if args.MCTS_num_processes > mp.cpu_count():
            num_processes = mp.cpu_count()
            logger.info(
                "Required number of processes exceed number of CPUs! Setting MCTS_num_processes to %d" % num_processes)
        else:
            num_processes = args.MCTS_num_processes

        logger.info("Spawning %d processes..." % num_processes)
        with torch.no_grad():
            for i in range(num_processes):
                p = mp.Process(target=MCTS_self_play, args=(
                net, args.num_games_per_MCTS_process, start_idx, i,
                args, iteration))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        logger.info("Finished multi-process MCTS!")

    elif args.MCTS_num_processes == 1:
        logger.info("Preparing model for MCTS...")
        net.eval()

        current_net_filename = os.path.join("./model_data/", \
                                            net_to_play)
        if os.path.isfile(current_net_filename):
            checkpoint = torch.load(current_net_filename)
            net.load_state_dict(checkpoint['state_dict'])
            logger.info("Loaded %s model." % current_net_filename)
        else:
            torch.save({'state_dict': net.state_dict()},
                       os.path.join("./model_data/", net_to_play))
            logger.info("Initialized model.")

        with torch.no_grad():
            MCTS_self_play(net, args.num_games_per_MCTS_process,
                           start_idx, 0, args, iteration)
        logger.info("Finished MCTS!")

def save_as_pickle(filename, data):
    completeName = os.path.join("./datasets/", filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)


def load_pickle(filename):
    completeName = os.path.join("./datasets/", filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data
