import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BoardData(Dataset):
    def __init__(self, dataset):  # dataset is np.array of (s, p, v)
        self.x_board_state = dataset[:, 0]
        self.y_policy = dataset[:, 1]
        self.y_value = dataset[:, 2]

    def __len__(self):
        return len(self.x_board_state)

    def __getitem__(self, idx):
        return self.x_board_state[idx], \
               self.y_policy[idx], \
               self.y_value[idx]


class JasonNet(torch.nn.Module):
    def __init__(self):
        super(JasonNet, self).__init__()
        self.board_size = 14

        # common layers
        self.cnn1d_1 = torch.nn.Conv1d(in_channels=1, out_channels=3,
                                       kernel_size=3, stride=1)
        self.conv1 = nn.Conv1d(3, 13, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(13, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv1d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(52, self.board_size)
        # state value layers
        self.val_conv1 = nn.Conv1d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(26, 32)
        self.val_fc2 = nn.Linear(32, 1)
        self.val_fc3 = nn.Linear(1, 1)

    def forward(self, x):
        x = torch.relu(self.cnn1d_1(x))
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # action policy layers
        x_act = torch.relu(self.act_conv1(x))
        x_act = torch.relu(self.act_fc1(x_act.view(-1, 52)))
        x_act = torch.log_softmax(x_act, dim=1).exp()

        # state value layers
        x_val = torch.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.board_size - 2)
        x_val = torch.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val


class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()
        self.mse_loss_fn = torch.nn.MSELoss(reduction='sum')

    def forward(self, y_value, value, y_policy, policy):
        # Does the below line still throw a warning?
        value_error = self.mse_loss_fn(y_value[0], value)

        # sum of mean-squared error value and cross-entropy policy loss
        policy_error = torch.sum((-policy * (1e-8 + y_policy).log()), 1)
        ttl_error = (value_error.view(-1).float() + policy_error).mean()

        return ttl_error
