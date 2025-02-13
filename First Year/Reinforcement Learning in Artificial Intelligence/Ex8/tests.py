from breakthrough import Breakthrough
import random

def random_board(size):
    board = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(random.choice([1,0,-1]))
        board.append(row)
    return board

def random_vector(size):
    return [random.choice([-1,0,1]) for _ in range(size*size)]


def boards_are_equals(board1, board2):
    for i in range(len(board1)):
        for j in range(len(board2[i])):
            if board1[i][j] != board2[i][j]:
                return False
    return True

def vectors_are_equals(vector1, vector2):
    for i in range(len(vector1)):
        if vector1[i] != vector2[i]:
            return False
    return True

breakthrough = Breakthrough(board_size=5)
# breakthrough.player = breakthrough.change_player()

board = random_board(breakthrough.board_size)
breakthrough.board = board

vercor = breakthrough.encode()
breakthrough.unencode(vercor)

if boards_are_equals(board, breakthrough.board):
    print("Boards are equals!")
else:
    print("Boards aren't equals!")

# vectors
white_vector = [random.choice([0,0,1]) for _ in range(breakthrough.board_size*breakthrough.board_size)]
black_vector = [random.choice([0,0,1]) if not white_vector[i] else 0 for i in range(breakthrough.board_size*breakthrough.board_size)]
verctor2 = white_vector + black_vector + [1]
breakthrough.unencode(verctor2)
verctor3 = breakthrough.encode()
if vectors_are_equals(verctor2, verctor3):
    print("Verctors are equals!")
else:
    print("Verctors aren't equals!")


# index = random.randint(0, breakthrough.board_size ** 2 *3)
breakthrough.player = -1
index = 48
From, To = breakthrough.decode(index)
index = 27
From, To = breakthrough.decode(index)
res_index = breakthrough.undecode(From, To)
if index == res_index:
    print("Indexes are equals!")
else:
    print("Indexes aren't equals!")

breakthrough.player = breakthrough.change_player()
index = random.randint(0, breakthrough.board_size ** 2 *3)
From, To = breakthrough.decode(index)
res_index = breakthrough.undecode(From, To)
if index == res_index:
    print("Indexes are equals!")
else:
    print("Indexes aren't equals!")