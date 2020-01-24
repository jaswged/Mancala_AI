class Board(object):
    def __init__(self):
        self.is_player_1s_turn = True
        # self.board_state = [0,1,2,3,4,5, 0, 7,8,9,10,11,12, 0]
        # self.board_state = [2, 3, 1, 0, 9, 0, 20, 1, 1, 0, 2, 8, 3, 4]
        # self.board_state = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
        self.board_state = self.initial_board()
        self.player_1_pit = 6
        self.player_2_pit = 13
        self.game_over = False
        self.winner = None
        self.pairs = {0: 12, 1: 11, 2: 10,  3: 9,  4: 8,  5: 7,
                      7:  5, 8:  4, 9:  3, 10: 2, 11: 1, 12: 0}

    def __str__(self):
        return "Board object for game Mancala"

    @staticmethod
    def initial_board():
        # Returns a representation of the starting state of the game
        #   4  4  4 | 4  4  4       12 11 10 | 9  8  7
        # 0                   0  13                   6 1st players home
        #   4  4  4 | 4  4  4       0  1  2  | 3  4  5
        return [4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0]

    @staticmethod
    def process_static_move(board, move):
        board.process_move(move)

    def process_move(self, move):
        # Check that the chosen move is a legal move
        print("Processing move {} for player {}"
              .format(move, "1" if self.is_player_1s_turn else "2"))
        if move not in self.get_legal_moves():
            # legal moves is the values of the moves not the indexes
            print("Not a valid move.")
            return

        # Get the marbles from the hole.
        marbles = self.board_state[move]
        self.board_state[move] = 0
        print("Marbles in pit {} is {}".format(move, marbles))

        # Place the marbles around the board. Skipping opponent home
        pit_to_add = move
        print("pit to add {}".format(pit_to_add))
        for m in range(marbles):
            pit_to_add = self.get_pit_to_add(pit_to_add)
            print("in for loop for placing marbles. Adding to {}"
                  .format(pit_to_add))
            self.board_state[pit_to_add] += 1

        # Check if pit was empty, steal from opponent only on your side
        if self.board_state[pit_to_add] == 1 \
                and self.own_side_pit(pit_to_add):
            print("Pit was empty. Steal the opponent marbles")
            self.board_state[pit_to_add] = 0
            amount_to_add = 1
            # Get the opposing side pit when zero
            opponent_pit = self.get_opposite_pit(pit_to_add)
            amount_to_add += self.board_state[opponent_pit]
            current_home = self.player_1_pit if self.is_player_1s_turn \
                else self.player_2_pit
            self.board_state[current_home] += amount_to_add
            self.board_state[opponent_pit] = 0

            # Add marbles to players pit
            own_home = self.player_1_pit if self.is_player_1s_turn \
                else self.player_2_pit
            self.board_state[own_home] += amount_to_add

        # if pit_to_add is own home. then free turn
        if not self.own_home(pit_to_add):
            self.switch_player()

        # Check for the win conditions.
        if self.board_state[self.player_1_pit] > 24 or \
                self.board_state[self.player_2_pit] > 24:
            self.game_over = True

        if self.marbles_gone_on_one_side():
            # Clean up all marbles by moving them to that players home
            self.clean_up_winning_marbles()
            self.game_over = True

    def get_legal_moves(self):
        filtered = list(map(lambda x: x[0],
                            filter(lambda x: x[1] != 0,
                                   enumerate(self.board_state))))

        # Remove player homes from legal moves
        if self.player_1_pit in filtered:
            filtered.remove(self.player_1_pit)
        if self.player_2_pit in filtered:
            filtered.remove(self.player_2_pit)

        board_side = list(filter(lambda x: x < 6, filtered)) \
            if self.is_player_1s_turn \
            else list(filter(lambda x: x > 6, filtered))

        return board_side

    def switch_player(self):
        self.is_player_1s_turn = not self.is_player_1s_turn

    def get_whose_turn(self):
        return 1 if self.is_player_1s_turn else 2

    def get_pit_to_add(self, pit_to_increment):
        pit = (pit_to_increment + 1) % 14
        print("Pit to increment is: {}".format(pit_to_increment))

        if self.enemy_home(pit):
            print("\t\tPit is enemy home. Skip it")
            pit += 1
            pit = pit % 14

        print("Previous pit is {}, new pit is {}"
              .format(pit_to_increment, pit))

        return pit

    def enemy_home(self, pit_to_add):
        enemy_home = pit_to_add == self.player_2_pit \
            if self.is_player_1s_turn \
            else pit_to_add == self.player_1_pit
        return enemy_home

    def own_home(self, pit_to_add):
        own_home = pit_to_add == self.player_1_pit \
            if self.is_player_1s_turn \
            else pit_to_add == self.player_2_pit
        return own_home

    def is_tie(self):
        return self.board_state[6] == self.board_state[13]

    # TODO Check if game is over. even if it is a tie
    def get_winner(self):
        winning_player = self.board_state[6] > self.board_state[13]
        a = self.board_state[6]
        b = self.board_state[13]
        print("Player 1: {}, Player 2: {}".format(a, b))
        compare = (a > b) - (a < b)
        print("Compare to is: {}".format(compare))

        print("Winner is: {}".format(winning_player))
        return winning_player

    def marbles_gone_on_one_side(self):
        side1 = self.board_state[:6]
        if len(list(filter(lambda x: x == 0, side1))) == 6:
            return True

        side2 = self.board_state[7:13]
        if len(list(filter(lambda x: x == 0, side2))) == 6:
            return True

        return False

    def print_board_state(self):
        print("\n\n")
        print("         12:{}  11:{}  10:{}  9:{}  8:{}  7:{}".format(
                                                self.board_state[12],
                                                self.board_state[11],
                                                self.board_state[10],
                                                self.board_state[9],
                                                self.board_state[8],
                                                self.board_state[7]))
        print("2P Home:{}                                1P Home:{}".
              format(self.board_state[13], self.board_state[6]))
        print("         0:{}   1:{}   2:{}   3:{}  4:{}  5:{}".format(
                                                self.board_state[0],
                                                self.board_state[1],
                                                self.board_state[2],
                                                self.board_state[3],
                                                self.board_state[4],
                                                self.board_state[5]))

    def clean_up_winning_marbles(self):
        print("Total before {}".format(self.board_state[6]))
        for x in range(6):
            print("x: {}".format(x))
            to_add = self.board_state[x]
            self.board_state[x] = 0
            self.board_state[6] += to_add
        print("Total after {}".format(self.board_state[6]))

        print("Total coins for second player")
        print("Total before {}".format(self.board_state[13]))
        for x in range(7, 13):
            print("x: {}".format(x))
            to_add = self.board_state[x]
            self.board_state[x] = 0
            self.board_state[13] += to_add
        print("Total after {}".format(self.board_state[13]))

    def get_opposite_pit(self, pit):
        return self.pairs.get(pit)

    def own_side_pit(self, pit_to_add):
        own_pit = pit_to_add < 6 if self.is_player_1s_turn \
            else 6 < pit_to_add < 13
        return own_pit

# From Board.py file
    def next_state(self, state, play):
        # Takes the game state, and the move to be applied.
        # Returns the new game state.
        pebbles = state[play]
        print("Pebbles to move {}".format(pebbles))
        for x in range(pebbles):
            position_to_increment = 2  # todo play += 1
            holes_value = state[position_to_increment]
            state[position_to_increment] = holes_value
        print("Pass")
        pass