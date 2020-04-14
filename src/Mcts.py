import numpy as np
from rules.Mancala import Board
import time
import copy
import torch


class Node:
    """Search result of one abstract (or root) state"""

    def __init__(self, p, v):
        self.p = p
        self.v = v
        self.n = np.zeros_like(p)
        self.q_sum = np.zeros_like(p)
        self.n_all = 1
        self.q_sum_all = v / 2  # prior

    def update(self, action, q_new):
        # update
        self.n[action] += 1
        self.q_sum[action] += q_new

        # Update overall stats
        self.n_all += 1
        self.q_sum_all += q_new


class Tree:
    """Monte Carlo Tree"""

    def __init__(self, net):
        self.net = net
        self.nodes = {}

    def search(self, state, depth):
        # Return predicted value from new state: because it is recursive
        if state.is_game_over():
            return state.get_winner()

        # Get key for the current state
        key = state.board_key()

        # if current state is not in the nodes add it and return value
        if key not in self.nodes:
            current_board_t_sqzd = board_to_tensor(state.current_board)

            # Use neural net to predict policy and value
            estimated_policy, estimated_val = self.net(current_board_t_sqzd)
            # p, v = self.net.predict(state)

            policy_numpy = estimated_policy.detach().cpu().numpy()[0]
            val = estimated_val.item()

            self.nodes[key] = Node(policy_numpy, val)
            return val

        # State transition by an action selected from bandit
        node = self.nodes[key]
        p = node.p
        if depth == 0:
            # Add noise to policy on the root node
            p = 0.75 * p + 0.25 * np.random.dirichlet([0.15] * len(p))

        best_action, best_ucb = None, -float('inf')
        # For each legal action find the best choice
        for action in state.get_legal_moves():
            n, q_sum = 1 + node.n[action], node.q_sum_all / node.n_all + \
                       node.q_sum[action]
            ucb = q_sum / n + 2.0 * np.sqrt(node.n_all) * p[
                action] / n  # PUCB

            if ucb > best_ucb:
                best_action, best_ucb = action, ucb

        # Search next state by recursively calling this function
        state.process_move(best_action)
        # With the assumption of changing player by turn
        q_new = -self.search(state, depth + 1)
        node.update(best_action, q_new)

        return q_new

    def think(self, state, num_simulations, temperature=0, show=False):
        # End point of MCTS
        if show:
            print(state)
        start, prev_time = time.time(), 0
        for _ in range(num_simulations):
            self.search(copy.deepcopy(state), depth=0)

            # Display search result on every second
            if show:
                tmp_time = time.time() - start
                if int(tmp_time) > int(prev_time):
                    prev_time = tmp_time
                    root, pv = self.nodes[
                                   state.board_key()], self.pv(state)
                    #print('%.2f sec. best %s. q = %.4f. n = %d / %d. pv = %s'
                    #    % (tmp_time, state.action2str(pv[0]),
                    #       root.q_sum[pv[0]] / root.n[pv[0]],
                    #       root.n[pv[0]], root.n_all,
                    #       ' '.join([state.action2str(a) for a in pv])))

        #  Return probability distribution weighted by the number of simulations
        n = root = self.nodes[state.board_key()].n + 1
        n = (n / np.max(n)) ** (1 / (temperature + 1e-8))
        return n / n.sum()

    def pv(self, state):
        # Return principal variation (action sequence which is considered as the best)
        s, pv_seq = copy.deepcopy(state), []
        while True:
            key = s.board_key()
            if key not in self.nodes or self.nodes[key].n.sum() == 0:
                break
            best_action = sorted(
                [(a, self.nodes[key].n[a]) for a in s.get_legal_moves()],
                key=lambda x: -x[1])[0][0]
            pv_seq.append(best_action)
            s.process_move(best_action)
        return pv_seq


def board_to_tensor(board):
    if torch.cuda.is_available():
        current_board_t = torch.tensor(board, dtype=torch.float).cuda()
    else:
        current_board_t = torch.tensor(board, dtype=torch.float32)
    # return a new tensor with a 1 dimension added at provided index
    current_board_t_sqzd = current_board_t.unsqueeze(0).unsqueeze(0)
    return current_board_t_sqzd
