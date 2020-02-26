from ConnectNet import BoardData
from NeuralNet import PolicyValueNet, AlphaLoss
from rules.Mancala import Board
from collections import defaultdict, deque
from MonteCarlo import MonteCarlo, MCTSPlayer, MCTSPurePlayer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
from random import shuffle
import os
import datetime
import logging
import pickle
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from JasonMonteCarlo import load_pickle, save_as_pickle, board_to_tensor

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)


class Trainer:
    def __init__(self, init_model=None):
        # params of the board and the game
        self.board_size = 15
        self.n_in_row = 4
        self.board = Board()

        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0

        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_size,
                                                   model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_size)
        self.mcts_player = MonteCarlo(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout)

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_size)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array(
                    [np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(
                self.mcts_player,
                temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(
            state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(
                state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(
                new_probs + 1e-10)),
                                axis=1)
                         )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(
                                 winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(
                                 winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout)
        pure_mcts_player = MCTSPurePlayer(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            self.pure_mcts_playout_num,
            win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                    i + 1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                # check the performance of the current model,
                # and save the model params
                if (i + 1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i + 1))
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model(
                        './current_policy.model')
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model(
                            './best_policy.model')
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def train_net(net, iter, lr, bs, epochs):
    data_path = "./datasets/iter_%d/" % iter
    datasets = []
    for idx, file in enumerate(os.listdir(data_path)):
        filename = os.path.join(data_path, file)
        with open(filename, 'rb') as fo:
            datasets.extend(pickle.load(fo, encoding='bytes'))
    datasets = np.array(datasets)
    logger.info("Loaded data from %s." % data_path)

    if torch.cuda.is_available():
        net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.8, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=
                                    [50, 100, 150, 200, 250, 300, 400],
                                               gamma=0.77)
    train(net, datasets, optimizer, scheduler, iter, bs, epochs)


def train(net, dataset, optim, scheduler, iter, bs, epochs):
    torch.manual_seed(0)
    net.train()
    loss_function = AlphaLoss()

    train_set = BoardData(dataset)
    train_loader = DataLoader(train_set, batch_size=bs,
                              shuffle=True, num_workers=0,
                              pin_memory=False)
    losses_per_epoch = load_results(iter + 1)

    logger.info("Starting training process...")
    update_size = len(train_loader) // 10

    # From Torch example
    loss_fn = torch.nn.MSELoss(reduction='sum')
    # learning_rate = 1e-4
    # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    for t in range(500):
        y_pred_p, y_pred_v = net(dataset[0][0])
        y_act_v = dataset[0][2]
        y_act_p = dataset[0][1]
        loss_p = loss_fn(y_pred_p, y_act_p)
        loss = loss_fn(y_pred_v, y_act_v)
        if t % 100 == 99:
            print(t, loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()

    for epoch in range(epochs):
        total_loss = 0.0
        batch_loss = []

        shuffle(dataset)
        for i, data in enumerate(dataset, 1):
            value = data[2]
            policy = data[1]
            board_t = board_to_tensor(data[0])

            policy_pred, value_pred = net(board_t)
            policy_t = torch.tensor(policy, dtype=torch.float32)

            # Calculate loss. sub array may fail
            loss = loss_function(value_pred[:, 0], value, policy_pred,
                                 policy_t)

            optim.zero_grad()
            loss.backward()
            # clip_grad_norm_(net.parameters(), 1.0)
            optim.step()

            total_loss += loss.item()

            if i % update_size == (update_size - 1):
                batch_loss.append(1 * total_loss / update_size)
                logger.debug(f"Iteration: {iter}, Epoch: {epoch + 1}.")
                logger.debug(f"Loss per batch: {batch_loss[-1]}")
                logger.debug(" ")
                total_loss = 0.0
        # End of for loop

        scheduler.step()
        if len(batch_loss) >= 1:
            losses_per_epoch.append(sum(batch_loss) / len(batch_loss))
        if (epoch % 2) == 0:
            filename = "losses_per_epoch_iter%d.pkl" % (iter + 1)
            complete_name = os.path.join("./model_data/", filename)
            save_as_pickle(complete_name, losses_per_epoch)
            torch.save({'epoch': epoch + 1,
                        'state_dict': net.state_dict(),
                        'optimizer': optim.state_dict(),
                        'scheduler': scheduler.state_dict(), },
                       os.path.join("./model_data/",
                                    "trn_net_iter%d.pth.tar" %
                                    (iter + 1)))
    logger.info("Finished Training!")
    fig = plt.figure()
    ax = fig.add_subplot(222)
    ax.scatter([e for e in range(start_epoch, (
                len(losses_per_epoch) + start_epoch))],
               losses_per_epoch)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss per batch")
    ax.set_title("Loss vs Epoch")
    plt.savefig(os.path.join("./model_data/",
                             "Loss_vs_Epoch_iter%d_%s.png" % (
                             (iter + 1),
                             datetime.datetime.today().strftime(
                                 "%Y-%m-%d"))))
    plt.show()


def load_results(iteration):
    """ Loads saved results if exists """
    losses_path = "./model_data/losses_per_epoch_iter%d.pkl" % iteration
    if os.path.isfile(losses_path):
        filename = "losses_per_epoch_iter%d.pkl" % iteration
        filename = os.path.join("./model_data/", filename)
        losses_per_epoch = load_pickle(filename)
        logger.info("Loaded results buffer")
    else:
        losses_per_epoch = []
    return losses_per_epoch
