from enum import Enum

class Colors:
    White = (255, 255, 255)
    Black = (0, 0, 0)
    Gray = (191, 191, 191)
    LightGray = (127, 127, 127)
    Green = (0, 191, 0)
    Red = (127, 15, 15)    


class Player:
    Black = -1
    White = 1

class GameState:
    OnGoing = 0
    BlackWon = Player.Black
    WhiteWon = Player.White

class MoveDirection(Enum):
    Left = -1
    Forward = 0
    Right = 1

class Position:
    def __init__(self, pos):
        self.x = pos[0]
        self.y = pos[1]

class Move:
    def __init__(self, from_pos, to_pos):
        self.from_pos: Position = from_pos
        self.to_pos: Position = to_pos