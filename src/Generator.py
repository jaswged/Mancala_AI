import os
import torch
import pickle
from tqdm import tqdm
import logging
import copy
import numpy as np
import datetime
import torch.multiprocessing as mp
from rules.Mancala import Board
from Mcts import Tree


logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)


def generate_data(net, episodes, depth, iteration):
    print("Generate data for training")
    make_training_directory(iteration)

    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # Load or save the neural network
    net_name = "net_iter%d.pth.tar" % iteration
    net_filename = os.path.join("./model_data/", net_name)
    if os.path.isfile(net_filename):
        checkpoint = torch.load(net_filename)
        net.load_state_dict(checkpoint)
        logger.info("Loaded %s model." % net_filename)
    else:
        torch.save(net.state_dict(), net_filename)
        logger.info("Saved initial model.")
    
    self_play(net, episodes, 1.1, iteration, depth)


def self_play(net, episodes, temp, iteration, depth):
    for ind in tqdm(range(episodes)):
        game = Board()  # new game to play with
        is_game_over = False
        replay_buffer = []  # (state, policy, value) for NN training
        value = 0  # winning player. 0 means tie
        move_count = 0  # number of moves so far in the game

        # While no winner
        while is_game_over is False:
            # Choose best policy after 8 moves.
            t = temp if move_count < 8 else 0.1
            state_copy = copy.deepcopy(game.current_board)
            game_copy = copy.deepcopy(game)

            # In each turn:
            #  Perform a fixed # of MCTS simulations for State at t
            #  pick move by sampling policy(state, reward) from net
            root = Tree(net)
            policy = root.think(game_copy, depth, t, show=False)
            legal_moves = game_copy.get_legal_moves()

            mv = game_copy.policy_for_legal_moves(legal_moves, policy)
            move = np.random.choice(legal_moves, p=mv)
            game.process_move(move)

            # Add game_state and choice to replay buffer to train NN
            replay_buffer.append([state_copy, policy])

            if game.game_over is True:
                value = game.get_winner()
                is_game_over = True
            move_count += 1

        save_game_data(replay_buffer, value, iteration, ind)


def make_training_directory(iteration):
    if not os.path.isdir("./datasets/iter_%d" % iteration):
        if not os.path.isdir("datasets"):
            os.mkdir("datasets")
        os.mkdir("datasets/iter_%d" % iteration)


def save_game_data(replay_buffer, value, itr, ind):
    dataset = []
    # replay_buffer is [board_state, policy]
    for idx, data in enumerate(replay_buffer):
        state, pol = data
        dataset.append([state, pol, value])
    del replay_buffer
    filename = "iter_%d/" % itr + "dataset_iter%d_%i_%s.pkl" % (
        itr, ind, datetime.datetime.today().strftime("%Y-%m-%d"))
    complete_name = os.path.join("./datasets/", filename)
    save_as_pickle(complete_name, dataset)


def save_as_pickle(complete_name, data):
    with open(complete_name, 'wb') as output:
        pickle.dump(data, output)
