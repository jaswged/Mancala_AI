import datetime
import logging
import os
import pickle
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ConnectNet import BoardData
from MonteCarlo import load_pickle, save_as_pickle, board_to_tensor
from NeuralNet import AlphaLoss

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)


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

    for epoch in tqdm(range(epochs)):
        logger.info(F"Training Epoch: {epoch}")
        total_loss = 0.0
        batch_loss = []

        shuffle(dataset)
        for i, data in enumerate(dataset, 1):
            value = data[2]
            policy = data[1]
            board_t = board_to_tensor(data[0])

            policy_pred, value_pred = net(board_t)
            policy_t = torch.tensor(policy, dtype=torch.float32)
            value_t = torch.tensor(value, dtype=torch.float32)
            # Set cuda on if using a gpu to avoid ASSERT FAILED error.
            if torch.cuda.is_available():
                value_t = value_t.cuda()
                policy_t = policy_t.cuda()

            # Calculate loss. sub array may fail
            loss = loss_function(value_pred[:, 0], value_t, policy_pred,
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
    ax.scatter([e for e in range(len(losses_per_epoch))],
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
