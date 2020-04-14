import logging
import os
import pickle
from argparse import ArgumentParser

from Mancala import Board
from Train import train_net
from Arena import Arena
#from MonteCarlo import run_monte_carlo
from Mcts import Tree
from NeuralNet import JasonNet

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)


def save_as_pickle(iteration, nn):
    net_name = "winner_iter%d.pkl" % iteration
    file_name = os.path.join("./model_data/", net_name)
    with open(file_name, 'wb') as output:
        pickle.dump(nn, output)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--iteration", type=int, default=0,
                        help="Current iteration number to resume from")
    parser.add_argument("--total_iterations", type=int, default=100,
                        help="Total number of iterations to run")
    parser.add_argument("--MCTS_num_processes", type=int, default=5,
                        help="Nbr of processes to run MCTS self-plays")
    parser.add_argument("--episodes", type=int, default=150,
                        help="Nbr of games to play")
    parser.add_argument("--search_depth", type=int, default=300,
                        help="How deep in tree to search")
    parser.add_argument("--bs", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=int, default=1e-4,
                        help="Learning Rate")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of epochs to train")

    args = parser.parse_args()

    episodes = args.episodes
    search_depth = args.search_depth
    logger.debug("Number of episodes: {} and search depth {}"
                 .format(episodes, search_depth))

    # Setup NN
    net = JasonNet()
    current_NN = net
    best_NN = net

    if not os.path.isdir("model_data"):
        os.mkdir("model_data")

    logger.info("Starting to train...")
    for i in range(args.iteration, args.total_iterations):
        logger.info(F"Iteration {i}")
        # Play a number of Episodes (games) of self play
        tree = Tree(current_NN)
        state = Board()
        tree.think(state, 1000, show=True)
        # run_monte_carlo(current_NN, 0, i, episodes, search_depth)

        # Train NN from dataset of monte carlo tree search above
        train_net(current_NN, i, args.lr, args.bs, args.epochs)

        # Fight new version against reigning champion in the Arena
        # Even with first iteration just battle against yourself
        arena = Arena(best_NN, current_NN)
        best_NN = arena.battle(episodes//2, search_depth)
        # Save the winning net as a Pickle for battle later
        save_as_pickle(i, best_NN)

    print("End of the main driver program. Training has completed!")
