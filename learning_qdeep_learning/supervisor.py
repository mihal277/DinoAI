from game import Direction, MoveResult

class GameSupervisor:
    def order_move(self, board, move):
        prev_score = board.get_score()
        move_result = board.move(Direction.from_arrow(move))
        reward = board.get_score() - prev_score
        done = (move_result == MoveResult.HOLE or move_result == MoveResult.COMPLETE)
        return reward, done, move_result
