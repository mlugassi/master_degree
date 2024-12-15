class Player:
    RED = 1
    YELLOW = -1

    @staticmethod
    def replace_player(cur_player) -> 'Player':
        return Player.RED if cur_player == Player.YELLOW else Player.YELLOW

class CellStatus:
    # Some constants that I want to use to fill the board.
    RED = 1
    YELLOW = -1
    EMPTY = 0

class GameStatus:
    # Game status constants
    RED_WIN = 1
    YELLOW_WIN = -1
    DRAW = 0
    ONGOING = -17  # Completely arbitrary