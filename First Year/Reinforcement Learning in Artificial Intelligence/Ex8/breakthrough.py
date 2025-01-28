import pygame
import sys
from pygame.locals import *
import copy
from breakthrough_types import *

# Game class
class Breakthrough:
    def __init__(self, board_size=8, title_size=80):
        self.board_size = board_size
        self.title_size = title_size
        self.window_size = title_size * board_size
        self.board = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        for y in range(self.board_size):
            if y <= 1:
                for x in range(self.board_size):
                    self.board[y][x] = Player.Black
            elif y >= 6:
                for x in range(self.board_size):
                    self.board[y][x] = Player.White
        self.state = GameState.OnGoing
        self.player = Player.White
        self.selection = None

    def get_state(self):
        for x in range(self.board_size):
            if self.board[0][x] == Player.White:
                return GameState.WhiteWon
            if self.board[self.board_size - 1][x] == Player.Black:
                return GameState.BlackWon
        return GameState.OnGoing

    def valid_moves(self, pos):
        x, y = pos
        player = self.player
        moves = []
        for dx in [MoveDirection.Left, MoveDirection.Forward, MoveDirection.Right]:
            nx, ny = x + dx, y - player
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[ny][nx] != player:
                if dx != MoveDirection.Forward or self.board[ny][nx] == 0:
                    moves.append((nx, ny))
        return moves

    def change_player(self):
        return Player.Black if self.player == Player.White else Player.White
    
    def make_move(self, start, end):
        x0, y0 = start
        x1, y1 = end
        self.board[y0][x0] = 0
        self.board[y1][x1] = self.player
        self.state = self.get_state()
        if self.state == GameState.OnGoing:
            self.player = self.change_player()
        self.selection = None

    def unmake_move(self, start, end, captured):
        x0, y0 = start
        x1, y1 = end
        self.board[y1][x1] = captured
        self.board[y0][x0] = self.player
        self.player = self.change_player()
        self.state = GameState.OnGoing
        self.selection = None

    def clone(self):
        return copy.deepcopy(self)

    def encode(self):
        encoded_state = []
        for row in self.board:
            encoded_state.extend(row)
        encoded_state.append(self.player)
        return encoded_state

    def decode(self, action_index):
        x0, y0 = divmod(action_index[0], self.board_size)
        x1, y1 = divmod(action_index[1], self.board_size)
        return ((x0, y0), (x1, y1))

    def status(self):
        winner = self.get_state()
        if winner:
            return "Won" if winner == self.player else "Lost"
        return "On Going"

    def legal_moves(self):
        moves = []
        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.board[y][x] == self.player:
                    moves.extend([((x, y), move) for move in self.valid_moves((x, y))])
        return moves

    # Pygame functions
    def draw_board(self):
        for y in range(self.board_size):
            for x in range(self.board_size):
                color = Colors.White if (x + y) % 2 == 0 else Colors.Gray
                pygame.draw.rect(self.screen, color, (x * self.title_size, y * self.title_size, self.title_size, self.title_size))
                piece = self.board[y][x]
                if piece != 0:
                    piece_color = Colors.White if piece == 1 else Colors.Black
                    pygame.draw.circle(self.screen, piece_color, (x * self.title_size + self.title_size // 2, y * self.title_size + self.title_size // 2), self.title_size // 3)
                    if piece == 1:  # Add black border for white pieces
                        pygame.draw.circle(self.screen, Colors.Black, (x * self.title_size + self.title_size // 2, y * self.title_size + self.title_size // 2), self.title_size // 3, 2)

    def draw_selection(self):
        if self.selection:
            x, y = self.selection
            pygame.draw.rect(self.screen, Colors.LightGray, (x * self.title_size, y * self.title_size, self.title_size, self.title_size), 3)
            for move in self.valid_moves((x, y)):
                mx, my = move
                pygame.draw.rect(self.screen, Colors.Green, (mx * self.title_size, my * self.title_size, self.title_size, self.title_size), 3)
    
    def run(self):
        clock = pygame.time.Clock()

        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == MOUSEBUTTONDOWN and self.state == GameState.OnGoing:
                    mx, my = pygame.mouse.get_pos()
                    x, y = mx // self.title_size, my // self.title_size

                    if self.selection is None and self.board[y][x] == self.player:
                        self.selection = (x, y)
                    elif self.selection:
                        if (x, y) in self.valid_moves(self.selection):
                            self.make_move(self.selection, (x, y))
                            self.selection = None
                        else:
                            self.selection = None

                # Draw
                self.screen.fill(Colors.Black)
                self.draw_board()
                self.draw_selection()

                if self.state != GameState.OnGoing:
                    winner = "White" if self.state == GameState.WhiteWon else "Black"
                    text = self.font.render(f"{winner} Won!!", True, Colors.Green)
                    self.screen.blit(text, (10, 10))

                    pygame.display.flip()
                    clock.tick(30)        


# Main loop
if __name__ == "__main__":
    breakthrogh = Breakthrough()
    breakthrogh.run()
