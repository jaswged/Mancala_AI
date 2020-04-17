import logging
import torch
import numpy as np
from tqdm import tqdm

from Mcts import Tree
from rules.Mancala import Board
from MonteCarlo import search, get_policy

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)


class Arena:
    def __init__(self, best_net, new_net):
        logger.debug("Setup arena")
        self.best_net = best_net
        self.new_net = new_net

    def battle(self, episodes, depth, mcts):
        """ Battle the 2 NN against each other and
            return the winner if they win 55% of the matches """
        logger.info("Battle the nets to the death in the Arena")
        new_wins = 0

        for _ in tqdm(range(episodes)):
            with torch.no_grad():
                winner = self.play_match(depth, mcts)
                logger.debug("%s wins!" % winner)
            if winner == "new":
                new_wins += 1

        winning_ratio = new_wins / episodes
        logger.info("Winning ratio is: {}".format(winning_ratio))
        if winning_ratio >= 0.55:
            logger.info("New neural net is better!")
            return self.new_net
        else:
            logger.info("Reigning champion lives on!")
            return self.best_net

    def play_match(self, depth, mcts):
        # Switch which net goes first randomly
        if np.random.uniform(0, 1) <= 0.5:
            first = self.new_net
            second = self.best_net
            f = "new"
            s = "best"
        else:
            first = self.best_net
            second = self.new_net
            f = "best"
            s = "new"

        game = Board()
        winner = 0
        temp = 0.1
        game_over = False

        while game_over is False:
            if game.player == 1:
                if mcts:
                    policy = get_mcts_policy(game, depth, first, temp)
                else:
                    policy = get_policy_moves(game, depth, first, temp)
            else:
                if mcts:
                    policy = get_mcts_policy(game, depth, second, temp)
                else:
                    policy = get_policy_moves(game, depth, second, temp)

            # Process best move
            legal_moves = game.get_legal_moves()
            policy = game.policy_for_legal_moves(legal_moves, policy)
            move = np.random.choice(legal_moves, p=policy)
            game.process_move(move)

            if game.is_game_over():
                game_over = True
                winner = game.get_winner()

        if winner == 1:
            return f
        elif winner == -1:
            return s
        else:
            return None


def get_policy_moves(game, depth, net, temp):
    root = search(game, depth, net)
    policy = get_policy(root, temp)
    logger.debug(policy)
    return policy


def get_mcts_policy(game, depth, net, temp):
    root = Tree(net)
    policy = root.think(game, depth, temp, show=False)
    logger.debug(policy)
    return policy
