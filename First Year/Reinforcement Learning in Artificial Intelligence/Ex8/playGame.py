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
                pygame.display.flip()
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

def main():
    game = Breakthrough()
    white_mcts_player = MCTSPlayer(Player.White)
    black_mcts_player = MCTSPlayer(Player.Black)
    play_against_me = False
    
    
    pygame.init()
    pygame.display.set_caption("Breakthrough")
    
    screen = pygame.display.set_mode((game.window_size, game.window_size))
    font = pygame.font.SysFont(None, 36)    
    clock = pygame.time.Clock()
    
    refresh(game, screen)
    pygame.display.flip()
    clock.tick(30) 

    while game.state == GameState.OnGoing:       
        if play_against_me and game.player == Player.White:
            move = my_move(game, screen, clock)
        elif not play_against_me and game.player == Player.White:
            move = white_mcts_player.choose_move(game, num_iterations=1)
        else:
            move = black_mcts_player.choose_move(game, num_iterations=100)

        game.make_move(move[0], move[1])
        
        refresh(game, screen) 
        pygame.display.flip()
        clock.tick(30)
        
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
                          
        winner = "White" if game.state == GameState.WhiteWon else "Black"
        text = font.render(f"{winner} Won!!", True, Colors.Green)
        screen.blit(text, (10, 10)) 
        pygame.display.flip()
        clock.tick(30)


# # Example usage
# if __name__ == "__main__":
#     # Placeholder for the neural network and game state
#     input_dim = 42  # Example input dimension
#     num_actions = 7  # Example number of actions
#     network = GameNetwork(input_dim, num_actions)

#     # Create a PUCT player
#     c_puct = 1.0
#     num_simulations = 100
#     player = PUCTPlayer(network, c_puct, num_simulations)

#     # Example initial state
#     initial_state = np.zeros(input_dim)

#     # Select an action
#     selected_action = player.select_action(initial_state)
#     print("Selected Action:", selected_action)



if __name__ == "__main__":
    main()
