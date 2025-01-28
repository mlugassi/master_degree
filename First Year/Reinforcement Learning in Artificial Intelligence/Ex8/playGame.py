from breakthrough import Breakthrough
from breakthrough_types import *
from mcts_node import MCTSNode
from mcts_player import MCTSPlayer
import pygame
import sys
from pygame.locals import *



def my_move(game: Breakthrough, screen, clock):
    # game.selection = None
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()            
            elif event.type == MOUSEBUTTONDOWN and game.state == GameState.OnGoing:
                mx, my = pygame.mouse.get_pos()
                x, y = mx // game.tile_size, my // game.tile_size

                if game.selection is None and game.board[y][x] == game.player:
                    game.selection = (x, y)
                elif game.selection:
                    if (x, y) in game.valid_moves(game.selection):
                        return (game.selection, (x, y))
                    else:
                        game.selection = None
                refresh(game,screen)
                clock.tick(30)
    

# Pygame functions
def draw_board(game:Breakthrough, screen):
    for y in range(game.board_size):
        for x in range(game.board_size):
            color = Colors.White if (x + y) % 2 == 0 else Colors.Gray
            pygame.draw.rect(screen, color, (x * game.tile_size, y * game.tile_size, game.tile_size, game.tile_size))
            piece = game.board[y][x]
            if piece != 0:
                piece_color = Colors.White if piece == 1 else Colors.Black
                pygame.draw.circle(screen, piece_color, (x * game.tile_size + game.tile_size // 2, y * game.tile_size + game.tile_size // 2), game.tile_size // 3)
                if piece == 1:  # Add black border for white pieces
                    pygame.draw.circle(screen, Colors.Black, (x * game.tile_size + game.tile_size // 2, y * game.tile_size + game.tile_size // 2), game.tile_size // 3, 2)

def draw_selection(game, screen):
    if game.selection:
        x, y = game.selection
        pygame.draw.rect(screen, Colors.LightGray, (x * game.tile_size, y * game.tile_size, game.tile_size, game.tile_size), 3)
        for move in game.valid_moves((x, y)):
            mx, my = move
            pygame.draw.rect(screen, Colors.Green, (mx * game.tile_size, my * game.tile_size, game.tile_size, game.tile_size), 3)
            
def refresh(game, screen):
    screen.fill(Colors.Black)
    draw_board(game, screen)
    draw_selection(game, screen)
    pygame.display.flip()

def main():
    game = Breakthrough()
    white_mcts_player = MCTSPlayer(Player.White)
    black_mcts_player = MCTSPlayer(Player.Black)
    play_against_me = True
    
    
    pygame.init()
    pygame.display.set_caption("Breakthrough")
    
    screen = pygame.display.set_mode((game.window_size, game.window_size))
    font = pygame.font.SysFont(None, 36)    
    clock = pygame.time.Clock()

    while True:
        refresh(game, screen)
        clock.tick(30) 
        
        if play_against_me and game.player == Player.White:
            move = my_move(game, screen, clock)
        elif not play_against_me and game.player == Player.White:
            move = white_mcts_player.choose_move(game, num_iterations=2500)
        else:
            move = black_mcts_player.choose_move(game, num_iterations=2500)

        game.make_move(move[0], move[1])

        if game.state != GameState.OnGoing:
            winner = "White" if game.state == GameState.WhiteWon else "Black"
            text = font.render(f"{winner} Won!!", True, Colors.Green)
            screen.blit(text, (10, 10))
            pygame.quit()
            sys.exit()

if __name__ == "__main__":
    main()
