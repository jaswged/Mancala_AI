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
from Node import Node

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)


def run_monte_carlo(net, start_ind, iteration, episodes, depth):
    if torch.cuda.is_available():
        net.cuda()

    logger.info("Prepare network for multi threaded Tree search")
    mp.set_start_method("spawn", force=True)
    net.share_memory()
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

    # Spawn processes to self play the game
    processes = []
    num_processes = 1 # mp.cpu_count()

    logger.info(F"Spawning {num_processes} processes")
    with torch.no_grad():
        for i in range(num_processes):
            p = mp.Process(target=self_play, args=(net, episodes,
                                                   start_ind, i, 1.1,
                                                   iteration, depth))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    logger.info("Finished multi-process MCTS!")


def self_play(net, episodes, start_ind, core, temp, iteration, depth):
    logger.info("[Core: %d]: Starting MCTS self-play..." % core)

    # Make directory for training iteration data to be stored
    make_training_directory(iteration)

    # tqdm is a progress bar
    for ind in tqdm(range(start_ind, episodes + start_ind)):
        logger.info(F"[Core: {core}]")
        game = Board()  # new game to play with
        is_game_over = False
        replay_buffer = []  # (state, policy, value) for NN training
        value = 0  # winning player
        move_count = 0  # number of moves so far in the game

        # While no winner and actions you can do
        while is_game_over is False:
            # Choose best policy after 11 moves.
            t = temp if move_count < 11 else 0.1

            state_copy = copy.deepcopy(game.current_board)

            # In each turn:
            #  Perform a fixed # of MCTS simulations for State at t
            #  pick move by sampling policy(state, reward) from net
            root = search(state_copy, depth, net)
            policy = get_policy(root, t)

            # Get policy only for legal moves
            legal_moves = game.get_legal_moves()
            legal_pol = game.policy_for_legal_moves(legal_moves, policy)

            logger.debug(f"[Core: {core}]: Game {ind} POLICY:\n {policy}")

            # Pick a random choice based off of the probability policy
            move = np.random.choice(legal_moves, p=legal_pol)
            game.process_move(move)

            # Add game_state and choice to replay buffer to train NN
            replay_buffer.append([state_copy, policy])
            logger.debug("[Iteration: %d Core: %d]: Game %d "
                         "CURRENT BOARD:\n" % (iteration, core, ind),
                         game.current_board_str())
            logger.debug(" ")

            if game.game_over is True:  # if somebody won
                value = game.get_winner()
                is_game_over = True
            move_count += 1

        save_game_data(replay_buffer, value, iteration, core, ind)


def search(game, sim_nbr, net):
    """ Create tree to find the best policy """
    # Create root node
    root = Node(game, move=None, parent=None)

    # For number of simulations find a leaf and evaluate the board with
    # the neural network. if the game is won, backup the winning value
    for _ in range(sim_nbr):  # number of simulations
        leaf = root.select_leaf()

        # Convert current_board array to tensor for pytorch
        current_board_t_sqzd = board_to_tensor(leaf.game.current_board)

        # Use neural net to predict policy and value
        estimated_policy, estimated_val = net(current_board_t_sqzd)
        policy_numpy = estimated_policy.detach().cpu().numpy()[0]
        estimated_val = estimated_val.item()

        # Check if game over
        if leaf.game.is_game_over() is True:
            # If game is over, backup actual value
            leaf.backup(leaf.game.get_winner())
            continue

        leaf.expand(policy_numpy)  # need to make sure valid moves
        leaf.backup(estimated_val)
    return root


def board_to_tensor(board):
    if torch.cuda.is_available():
        current_board_t = torch.tensor(board, dtype=torch.float).cuda()
    else:
        current_board_t = torch.tensor(board, dtype=torch.float32)
    # return a new tensor with a 1 dimension added at provided index
    current_board_t_sqzd = current_board_t.unsqueeze(0).unsqueeze(0)
    return current_board_t_sqzd


def get_policy(root, temp=1):
    return (root.child_number_visits ** (1 / temp)) / \
           sum(root.child_number_visits ** (1 / temp))


def save_game_data(replay_buffer, value, itr, core, ind):
    dataset = []
    # replay_buffer is [board_state, policy]
    for idx, data in enumerate(replay_buffer):
        state, pol = data
        dataset.append([state, pol, value])
    del replay_buffer
    filename = "iter_%d/" % itr + "dataset_iter%d_core%i_%i_%s.pkl" % (
        itr, core, ind, datetime.datetime.today().strftime
                       ("%Y-%m-%d"))
    complete_name = os.path.join("./datasets/", filename)
    save_as_pickle(complete_name, dataset)


def save_as_pickle(complete_name, data):
    with open(complete_name, 'wb') as output:
        pickle.dump(data, output)


def load_pickle(filename):
    with open(filename, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data


def make_training_directory(iteration):
    if not os.path.isdir("./datasets/iter_%d" % iteration):
        if not os.path.isdir("datasets"):
            os.mkdir("datasets")
        os.mkdir("datasets/iter_%d" % iteration)
