import logging
import torch
import numpy as np
from rules.Mancala import Board
from JasonMonteCarlo import search, get_policy

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)


class Arena:
    def __init__(self, best_net, new_net):
        logger.debug("Setup arena")
        self.best_net = best_net
        self.new_net = new_net

    def battle(self, episodes):
        logger.info("Battle the nets to the death")
        new_wins = 0

        for _ in range(episodes):
            with torch.no_grad():
                winner = self.play_match()
                logger.debug("%s wins!" % winner)
            if winner == "new":
                new_wins += 1

        winning_ratio = new_wins / episodes
        if winning_ratio >= 0.55:
            return self.new_net
        else:
            return self.best_net

    def play_match(self):
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
            print("game is still going")
            if game.player == 1:
                root = search(game, 777, first)
                policy = get_policy(root, temp)
            else:
                root = search(game, 777, second)
                policy = get_policy(root, temp)

            # Process best move
            legal_moves = game.get_legal_moves()
            policy = game.policy_for_legal_moves(legal_moves, policy)
            policy.argmax(axis=0)
            move = np.random.choice(legal_moves, p=policy)
            move = 0
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
