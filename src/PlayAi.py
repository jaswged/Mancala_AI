import torch
import os
import numpy as np

from Mcts import Tree
from NeuralNet import JasonNet
from rules.Mancala import Board
from MonteCarlo import search, get_policy
from argparse import ArgumentParser


def play_match_against_ai(network, depth, mcts):
    net_is_player1 = np.random.uniform(0, 1) <= 0.5
    if net_is_player1:
        print("You are player 2!")
    else:
        print("You are player 1!")

    game = Board()
    game.is_printing = True
    game_over = False
    moves_count = 0

    while game_over is False:
        # set exploration factor
        temp = 1. if moves_count <= 5 else 0.1
        moves_count += 1

        game.print_current_board()
        # Get move from player or ai depending on whose turn it is
        move = process_ai_move(game, depth, network, temp, mcts) \
            if net_is_player1 == (game.player == 1) \
            else get_move_from_player()

        game.process_move(move)

        if game.is_game_over():
            game_over = True

    return game


def get_move_from_player():
    while True:
        try:
            move = input("Please enter your move.")
            int_move = int(move)
            print("You chose move: {}".format(int_move))
            return int_move
        except ValueError:
            print("That's not an int! Try again.")


def process_ai_move(game, depth, net, temp, mcts):
    print("AI is thinking...")
    # turn off printing for AI's thinking
    game.is_printing = False
    policy = get_policy(game, depth, net, temp, mcts)

    # turn printing back on
    game.is_printing = True

    legal_moves = game.get_legal_moves()
    policy = game.policy_for_legal_moves(legal_moves, policy)
    # Choose best move from position
    move = legal_moves[policy.index(max(policy))]
    return move


def get_policy(game, depth, net, temp, mcts):
    if mcts:
        root = Tree(net)
        policy = root.think(game, depth, temp, show=False)
    else:
        root = search(game, depth, net)
        policy = get_policy(root, temp)
    return policy


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", default='net_iter0.pth.tar',
                        help="Model pickle you wish to play against")
    parser.add_argument("--search_depth", type=int, default=777,
                        help="How deep in tree to search")
    parser.add_argument("--mcts", type=int, default=True,
                        help="Use Mcts.py instead of MonteCarlo.py")
    args = parser.parse_args()

    best_net = args.model
    best_net_filename = os.path.join("./model_data/", best_net)
    net = JasonNet()

    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    checkpoint = torch.load(best_net_filename)
    net.load_state_dict(checkpoint)

    play_again = True
    while play_again:
        board = play_match_against_ai(net, args.search_depth, args.mcts)
        winner = board.get_winner()
        print(F"Winner is: {board.get_winner_string()}")
        while True:
            again = input("Do you want to play again? (Y/N)\n")
            if again.lower() in ["y", "n"]:
                if again.lower() == "n":
                    play_again = False
                    break
                else:
                    break
    print("Thanks for playing!")
