from Train import Train
from Arena import Arena
from argparse import ArgumentParser
import logging
import os
from ConnectNet import ConnectNet
from JasonMonteCarlo import run_monte_carlo

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
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
    args = parser.parse_args()

    episodes = args.episodes
    logger.debug("Number of episodes: {}".format(episodes))

    # Setup NN
    net = ConnectNet()
    current_NN = net
    best_NN = net

    # setup Train and Arena
    train = Train()
    arena = Arena()

    if not os.path.isdir("datasets"):
        os.mkdir("datasets")

    print('Begin the game.')
    logger.info("Starting training...")

    for i in range(args.iteration, args.total_iterations):
        logger.info("Iteration {}".format(i))
        # Play a number of Episodes (games) of self play
        run_monte_carlo(current_NN, 0, i, episodes)

        # In each turn:
        #  Perform a fixed # of MCTS simulations for State at t
        #  pick a move by sampling policy(state, policy, reward from net

        # Add game to replay buffer

        # Train NN on games from replay buffer
        # Save new version pickle
        train.run()

        # pit new version against reigning champion in the Arena
        # Take new one if it wins 55% of matches
        if i > 1:
            # Battle the models to the death!
            logger.info("cast them into the arena()")
            arena.battle(best_NN, current_NN)
        else:
            best_NN = current_NN

    print("End of the main driver program.")
