import logging
import os
from argparse import ArgumentParser

from Arena import Arena
from JasonMonteCarlo import run_monte_carlo
from NeuralNet import JasonNet

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--iteration", type=int, default=0,
                        help="Current iteration number to resume from")
    parser.add_argument("--total_iterations", type=int, default=100,
                        help="Total number of iterations to run")
    parser.add_argument("--MCTS_num_processes", type=int, default=5,
                        help="Nbr of processes to run MCTS self-plays")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Nbr of games to play")
    parser.add_argument("--search_depth", type=int, default=200,
                        help="How deep in tree to search")
    args = parser.parse_args()

    episodes = args.episodes
    search_depth = args.search_depth
    logger.debug("Number of episodes: {} and search depth {}"
                 .format(episodes, search_depth))

    # Setup NN
    net = JasonNet()
    current_NN = net
    best_NN = net

    if not os.path.isdir("datasets"):
        os.mkdir("datasets")

    print('Begin the game.')
    logger.info("Starting to train...")

    for i in range(args.iteration, args.total_iterations):
        logger.info("Iteration {}".format(i))
        # Play a number of Episodes (games) of self play
        run_monte_carlo(current_NN, 0, i, episodes, search_depth)

        # Fight new version against reigning champion in the Arena
        # Take new one if it wins 55% of matches
        if i > 0:
            # Battle the models to the death!
            logger.info("Cast them into the arena()")
            arena = Arena(best_NN, current_NN)
            best_NN = arena.battle(episodes, search_depth)
        else:
            best_NN = current_NN

    print("End of the main driver program. Training has completed!")
