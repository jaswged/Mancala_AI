#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib
import numpy as np
from rules.Mancala import Board

matplotlib.use("Agg")


class BoardData(Dataset):
    def __init__(self, dataset):  # dataset = np.array of (s, p, v)
        self.x_board_state = dataset[:, 0]
        self.y_policy = dataset[:, 1]
        self.y_value = dataset[:, 2]

    def __len__(self):
        return len(self.x_board_state)

    def __getitem__(self, idx):
        print("Get item")
        #return np.int64(self.x_board_states[idx].transpose(2, 0, 1)), \
        return self.x_board_state[idx], \
               self.y_policy[idx], \
               self.y_value[idx]


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.action_size = 14
        self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

    def forward(self, s):
        # batch_size x channels x board_x x board_y
        s = s.view(-1, 3, 6, 7)
        s = F.relu(self.bn1(self.conv1(s.view(15))))
        return s


class ResBlock(nn.Module):
    def __init__(self, inplanes=128, planes=128, stride=1,
                 downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(128, 3, kernel_size=1)  # value head
        self.bn = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(3 * 6 * 7, 32)  # TODO Different math here
        self.fc2 = nn.Linear(32, 1)

        self.conv1 = nn.Conv2d(128, 32, kernel_size=1)  # policy head
        self.bn1 = nn.BatchNorm2d(32)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(6 * 7 * 32, 7)

    def forward(self, s):
        v = F.relu(self.bn(self.conv(s)))  # value head
        v = v.view(-1,
                   3 * 6 * 7)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))

        p = F.relu(self.bn1(self.conv1(s)))  # policy head
        p = p.view(-1, 6 * 7 * 32)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p, v


class ConnectNet(nn.Module):
    def __init__(self):
        super(ConnectNet, self).__init__()
        self.conv = ConvBlock()
        for block in range(19):
            setattr(self, "res_%i" % block, ResBlock())
        self.outBlock = OutBlock()

    def forward(self, s):
        s = self.conv(s)
        for block in range(19):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outBlock(s)
        return s


class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        # sum of mean-squared error value and cross-entropy policy loss
        value_error = (value - y_value) ** 2
        policy_error = torch.sum((-policy * (1e-8 + y_policy.float())
                                  .float().log()
                                  ),
                                 1)
        ttl_error = (value_error.view(-1).float() + policy_error).mean()
        return ttl_error

# From Aplha zero notebook example
num_filters = 16
num_blocks = 6

class Conv(nn.Module):
    def __init__(self, filters0, filters1, kernel_size, bn=False):
        super().__init__()
        self.conv = nn.Conv1d(filters0, filters1, kernel_size, stride=1,
                              padding=kernel_size//2, bias=False)
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm1d(filters1)

    def forward(self, x):
        h = self.conv(x)
        if self.bn is not None:
            h = self.bn(h)
        return h

class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv = Conv(filters, filters, 3, True)

    def forward(self, x):
        return F.relu(x + (self.conv(x)))


class Net(nn.Module):
    '''ニューラルネット計算を行うクラス'''
    def __init__(self):
        super().__init__()
        state = Board()
        self.input_shape = torch.tensor(state.current_board).shape
        self.board_size = 15  # self.input_shape[1] * self.input_shape[2]

        self.layer0 = Conv(self.input_shape[0], num_filters, 3, bn=True)
        self.blocks = nn.ModuleList([ResidualBlock(num_filters)
                                     for _ in range(num_blocks)])

        self.conv_p1 = Conv(num_filters, 4, 1, bn=True)
        self.conv_p2 = Conv(4, 1, 1)

        self.conv_v = Conv(num_filters, 4, 1, bn=True)
        self.fc_v = nn.Linear(self.board_size * 4, 1, bias=False)

    def forward(self, x):
        h = F.relu(self.layer0(x))
        for block in self.blocks:
            h = block(h)

        h_p = F.relu(self.conv_p1(h))
        h_p = self.conv_p2(h_p).view(-1, self.board_size)

        h_v = F.relu(self.conv_v(h))
        h_v = self.fc_v(h_v.view(-1, self.board_size * 4))

        # value(状態価値)にtanhを適用するので負け -1 ~ 勝ち 1
        return F.softmax(h_p, dim=-1), torch.tanh(h_v)

    def predict(self, state):
        # 探索中に呼ばれる推論関数
        self.eval()
        x = torch.from_numpy(state.feature()).unsqueeze(0)
        with torch.no_grad():
            p, v = self.forward(x)
        return p.cpu().numpy()[0], v.cpu().numpy()[0][0]