import argparse

from game import Board, Game, MoveResult
from decider import HumanDecider

def finished_turn_callback(board, move_result):
    if move_result == MoveResult.CHEESE:
        print('You collected a cheese!')
    elif move_result == MoveResult.COMPLETE:
        print('You have completed the game!')
    elif move_result == MoveResult.HOLE:
        print('You have fallen into a hole...')
    elif move_result == MoveResult.INVALID:
        print('Invalid move!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--board_size', action='store', default=5)
    parser.add_argument('--n_cheese', action='store', default=3)
    parser.add_argument('--n_holes', action='store', default= 4)
    args = parser.parse_args()

    the_game = Game(Board(args.board_size, args.n_cheese, args.n_holes))
    the_game.play(move_decider=HumanDecider(), finished_turn_callback=finished_turn_callback)