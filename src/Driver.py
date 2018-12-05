from MonteCarlo import MonteCarlo
from Board import Board

if __name__ == "__main__":
    time = 10
    board = Board()
    carlo = MonteCarlo(board, time=time, max_moves=10)
    print('Begin the game.')
    carlo.print_variables()

    print("get play about to be called.")
    new_move = carlo.get_play()
    print("The new move is: {}".format(new_move))
    print("After get play called.")

    winner = False
    while not winner:
        print("continue")

    print("End of the main driver program.")
