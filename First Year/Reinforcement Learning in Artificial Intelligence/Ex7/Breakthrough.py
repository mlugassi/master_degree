import pygame
import sys
from pygame.locals import *

# Constants
BOARD_SIZE = 8
TILE_SIZE = 80
WINDOW_SIZE = BOARD_SIZE * TILE_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (191, 191, 191)
LIGHT_GRAY = (127, 127, 127)
GREEN = (0, 191, 0)
RED = (127, 15, 15)

# Game state
class Game:
    def __init__(self):
        self.board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        for y in range(BOARD_SIZE):
            if y <= 1:
                for x in range(BOARD_SIZE):
                    self.board[y][x] = -1
            elif y >= 6:
                for x in range(BOARD_SIZE):
                    self.board[y][x] = 1
        self.state = "Play"
        self.player = 1
        self.selection = None
        self.anim = None
        self.last_move = None

    def is_winner(self):
        for x in range(BOARD_SIZE):
            if self.board[0][x] == 1:
                return 1
            if self.board[BOARD_SIZE - 1][x] == -1:
                return -1
        return 0

    def valid_moves(self, pos):
        x, y = pos
        player = self.player
        moves = []
        for dx in [-1, 0, 1]:
            nx, ny = x + dx, y - player
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[ny][nx] != player:
                if dx != 0 or self.board[ny][nx] == 0:
                    moves.append((nx, ny))
        return moves

    def move(self, start, end):
        x0, y0 = start
        x1, y1 = end
        self.board[y0][x0] = 0
        self.board[y1][x1] = self.player
        if self.is_winner():
            self.state = "Won"
        else:
            self.player = -self.player

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Breakthrough")
font = pygame.font.SysFont(None, 36)

def draw_board(game):
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            color = WHITE if (x + y) % 2 == 0 else GRAY
            pygame.draw.rect(screen, color, (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE))
            
            piece = game.board[y][x]
            if piece != 0:
                piece_color = WHITE if piece == 1 else BLACK
                pygame.draw.circle(screen, BLACK, 
                                   (x * TILE_SIZE + TILE_SIZE // 2, y * TILE_SIZE + TILE_SIZE // 2), 
                                   TILE_SIZE // 3 + 2)
                pygame.draw.circle(screen, piece_color, 
                                   (x * TILE_SIZE + TILE_SIZE // 2, y * TILE_SIZE + TILE_SIZE // 2), 
                                   TILE_SIZE // 3)
def draw_selection(game):
    if game.selection:
        x, y = game.selection
        pygame.draw.rect(screen, LIGHT_GRAY, (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE), 3)
        for move in game.valid_moves((x, y)):
            mx, my = move
            pygame.draw.rect(screen, GREEN, (mx * TILE_SIZE, my * TILE_SIZE, TILE_SIZE, TILE_SIZE), 3)

# Main loop
def main():
    game = Game()
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

            if event.type == MOUSEBUTTONDOWN and game.state == "Play":
                mx, my = pygame.mouse.get_pos()
                x, y = mx // TILE_SIZE, my // TILE_SIZE

                if game.selection is None and game.board[y][x] == game.player:
                    game.selection = (x, y)
                elif game.selection:
                    if (x, y) in game.valid_moves(game.selection):
                        game.move(game.selection, (x, y))
                        game.selection = None
                    else:
                        game.selection = None

        # Draw
        screen.fill(BLACK)
        draw_board(game)
        draw_selection(game)

        if game.state == "Won":
            winner = "White" if game.player == 1 else "Black"
            text = font.render(f"{winner} wins!", True, GREEN)
            screen.blit(text, (10, 10))

        pygame.display.flip()
        clock.tick(30)

if __name__ == "__main__":
    main()


