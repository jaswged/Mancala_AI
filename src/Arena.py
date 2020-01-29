class Arena:
    def __init__(self):
        print("Setup arena")

    def battle(self, best_net, rookie_net):
        print("Battle the nets to the death")
    #     if (i + 1) % self.check_freq == 0:
    #         print("current self-play batch: {}".format(i + 1))
    #         win_ratio = self.policy_evaluate()
    #         self.policy_value_net.save_model('./current_policy.model')
    #         if win_ratio > self.best_win_ratio:
    #             print("New best policy!!!!!!!!")
    #             self.best_win_ratio = win_ratio
    #             # update the best_policy
    #             self.policy_value_net.save_model('./best_policy.model')
    #             if (self.best_win_ratio == 1.0 and
    #                     self.pure_mcts_playout_num < 5000):
    #                 self.pure_mcts_playout_num += 1000
    #                 self.best_win_ratio = 0.0