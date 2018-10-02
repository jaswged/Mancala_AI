from Mancala import Board


def play_game(board):
    print("Begin the real game logic.")

    # Start with player 2 as the beginner
    board.switch_player()

    while not board.game_over:
        board.print_board_state()
        print("It is player {}'s turn".format(board.get_whose_turn()))
        move = input("Please enter your move.")
        int_move = int(move)
        print("You chose move: {}".format(int_move))
        board.process_move(int_move)

    winning_player = board.get_winner()
    if winning_player == 0:
        print("The game is a tie.")
    else:
        print("The winner is: {}".format("Player 1"
                                         if winning_player > 0
                                         else "Player 2"))


if __name__ == "__main__":
    mancala = Board()
    play_game(mancala)
    print("End of the main driver program.")

def test_get_winner():
    mancala = Board()
    print(mancala.board_state)

    # tie
    win = mancala.get_winner()
    print("win is {}".format(win))
    win = 0
    # player 1: 1 vs 0
    mancala.board_state = [0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 11, 12, 0]
    print(mancala.get_winner())
    # tie:   4 vs 4
    mancala.board_state = [1, 1, 0, 3, 0, 5, 4, 7, 8, 0, 0, 11, 12, 4]
    print(mancala.get_winner())
    # player 2:   4 vs 54
    mancala.board_state = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 54]
    print(mancala.get_winner())
