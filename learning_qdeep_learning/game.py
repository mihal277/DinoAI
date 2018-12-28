import numpy as np

class Arrows:
    UP = '\x1b[A'
    DOWN = '\x1b[B'
    RIGHT = '\x1b[C'
    LEFT = '\x1b[D'

    def from_int(number):
        if number == 0:
            return Arrows.UP
        elif number == 1:
            return Arrows.DOWN
        elif number == 2:
            return Arrows.RIGHT
        else:
            return Arrows.LEFT

    def to_int(arrow):
        if arrow == Arrows.UP:
            return 0
        elif arrow == Arrows.DOWN:
            return 1
        elif arrow == Arrows.RIGHT:
            return 2
        else:
            return 3

    def to_string(arrow):
        if arrow == Arrows.UP:
            return 'UP'
        elif arrow == Arrows.DOWN:
            return 'DOWN'
        elif arrow == Arrows.RIGHT:
            return 'RIGHT'
        elif arrow == Arrows.LEFT:
            return 'LEFT'
        else:
            return ''

class MoveResult:
    OK = 0
    HOLE = 1
    CHEESE = 2
    COMPLETE = 3
    INVALID = 4

class Direction:
    UP = lambda x, y: (x - 1, y)
    DOWN = lambda x, y: (x + 1, y)
    RIGHT = lambda x, y: (x, y + 1)
    LEFT = lambda x, y: (x, y - 1)

    def from_arrow(arrow):
        if arrow == Arrows.UP:
            return Direction.UP
        elif arrow == Arrows.DOWN:
            return Direction.DOWN
        elif arrow == Arrows.RIGHT:
            return Direction.RIGHT
        else:
            return Direction.LEFT

class Board:
    def __init__(self, size, n_cheese, n_holes):
        self._size = size
        self._mouse_pos = (0, 0)
        self._cheese = self._init_cheese(n_cheese)
        self._holes = self._init_holes(n_holes)
        self._score = 0

    def _init_cheese(self, n):
        cheese = []
        while len(cheese) < n:
            x, y = tuple(np.random.randint(self._size, size=2))
            if (x, y) != self._mouse_pos \
                    and (x, y) not in cheese:
                cheese.append((x, y))
        return cheese

    def _init_holes(self, n):
        holes = []
        while len(holes) < n:
            x, y = tuple(np.random.randint(self._size, size=2))
            if (x, y) != self._mouse_pos \
                    and (x, y) not in self._cheese \
                    and (x, y) not in holes:
                holes.append((x, y))
        return holes

    def get_score(self):
        return self._score

    def move(self, direction):
        x, y = direction(self._mouse_pos[0], self._mouse_pos[1])
        if x < 0 or x >= self._size \
            or y < 0 or y >= self._size:
            self._score -= 1
            return MoveResult.INVALID
        if (x, y) in self._holes:
            self._score -= 100
            return MoveResult.HOLE
        self._mouse_pos = (x, y)
        if self._mouse_pos in self._cheese:
            self._cheese.remove(self._mouse_pos)
            self._score += 100
            if len(self._cheese) == 0:
                return MoveResult.COMPLETE
            else:
                return MoveResult.CHEESE
        return MoveResult.OK

    def get_state(self):
        # 1 is an empty cell
        # 2 is a cell with cheese
        # -1 is a cell with a hole
        # 3 is a mouse
        board = np.empty((self._size, self._size))
        board[:] = 1
        board[self._mouse_pos] = 3
        for c in self._cheese:
            board[c] = 2
        for h in self._holes:
            board[h] = -1
        return board

    def __str__(self):
        # O is an empty cell
        # C is a cell with cheese
        # H is a cell with a hole
        # M is a mouse
        byteboard = np.chararray((self._size, self._size))
        byteboard[:] = 'O'
        byteboard[self._mouse_pos] = 'M'
        for c in self._cheese:
            byteboard[c] = 'C'
        for h in self._holes:
            byteboard[h] = 'H'
        real_board = [x.tostring().decode('utf-8') for x in byteboard]
        real_board = ['#' + x + '#' for x in real_board]
        frame = ''.join(['#' for x in real_board[0]])
        real_board = [frame] + real_board + [frame]
        real_board = [' '.join(list(x)) for x in real_board]
        return '\n'.join(real_board)

class Game:
    def __init__(self, board):
        self._board = board

    def get_board(self):
        return self._board

    def make_move(self, move):
        return self._board.move(Direction.from_arrow(move))

    def play(self, move_decider, decision_callback=None, finished_turn_callback=None):
        while True:
            move = move_decider.decide(self._board)
            if decision_callback is not None:
                decision_callback(self._board, move)
            move_result = self._board.move(Direction.from_arrow(move))
            if finished_turn_callback is not None:
                finished_turn_callback(self._board, move_result)
            if move_result != MoveResult.OK and move_result != MoveResult.CHEESE:
                break
