from game import Arrows

class Decider:
    def decide(self, board):
        raise NotImplementedError

class HumanDecider(Decider):
    def decide(self, board):
        print('Score:', board.get_score())
        print(board)
        while True:
            i = input('Make a move (press arrow key and then enter)...')
            if i == Arrows.UP:
                return Arrows.UP
            elif i == Arrows.DOWN:
                return Arrows.DOWN
            elif i == Arrows.RIGHT:
                return Arrows.RIGHT
            elif i == Arrows.LEFT:
                return Arrows.LEFT
