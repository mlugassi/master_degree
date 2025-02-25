from breakthrough import Breakthrough
from breakthrough_types import *
from mcts_node import MCTSNode
from mcts_player import MCTSPlayer
from puct_player import PUCTPlayer
from game_network import GameNetwork
import pygame
import sys
from pygame.locals import *
import json
import os

def my_move(game: Breakthrough, screen=None, clock=None, use_gui=True):
    if not use_gui:
        # If GUI is disabled, just return a move from the console
        print("Enter move (e.g., x1 y1 x2 y2):")
        x1, y1, x2, y2 = map(int, input().split())
        return ((x1, y1), (x2, y2))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()            
            elif event.type == pygame.MOUSEBUTTONDOWN and game.state == GameState.OnGoing:
                mx, my = pygame.mouse.get_pos()
                x, y = mx // game.tile_size, my // game.tile_size

                if game.selection is None and game.board[y][x] == game.player:
                    game.selection = (x, y)
                elif game.selection:
                    if (x, y) in game.valid_moves(game.selection):
                        return (game.selection, (x, y))
                    else:
                        game.selection = None
                refresh(game, screen)
                pygame.display.flip()
                clock.tick(30)

# Pygame Functions (Only used if GUI is enabled)
def draw_board(game: Breakthrough, screen):
    for y in range(game.board_size):
        for x in range(game.board_size):
            color = Colors.White if (x + y) % 2 == 0 else Colors.Gray
            pygame.draw.rect(screen, color, (x * game.tile_size, y * game.tile_size, game.tile_size, game.tile_size))
            piece = game.board[y][x]
            if piece != 0:
                piece_color = Colors.White if piece == 1 else Colors.Black
                pygame.draw.circle(screen, piece_color, 
                    (x * game.tile_size + game.tile_size // 2, y * game.tile_size + game.tile_size // 2), 
                    game.tile_size // 3)
                if piece == 1:
                    pygame.draw.circle(screen, Colors.Black, 
                        (x * game.tile_size + game.tile_size // 2, y * game.tile_size + game.tile_size // 2), 
                        game.tile_size // 3, 2)
                    
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

def export_game(records, winner, size):
    for record in records:
        records[record]["winner"] = winner
    with open(f"play_book_{size}.json", "a") as f:
        json.dump(records, f)
        f.write("\n")

    # Load from file
    # with open("data.json", "r") as file:
    #     loaded_data = json.load(file)

def main():
    # inputs
    board_size = 5
    iteration = 1*1000
    exploration = 0.8
    play_against_me = False
    exit_on_finish = False
    use_gui = True
    record = False
    is_training = True
    c_puct = 0.2
    run_puct = False
    records = {}
    if play_against_me and not use_gui:
        exit("Error: You must Gui to play by yourself.")

    game = Breakthrough(board_size=board_size)

    if not run_puct:
        white_player = MCTSPlayer(Player.White, exploration_weight=exploration)
        black_player = MCTSPlayer(Player.Black, exploration_weight=exploration)
    else:
        white_network = GameNetwork(board_size)
        if os.path.isfile(f"game_network_weights_{board_size}_batch_11000.pth"):
            white_network.load_weights(f"game_network_weights_{board_size}_batch_11000.pth", train=False)
        if is_training:
            black_network = white_network # GameNetwork(board_size)
    
        white_player = PUCTPlayer(white_network,c_puct, is_training)
        black_player = PUCTPlayer(black_network, c_puct, is_training)

    if use_gui:
        pygame.init()
        pygame.display.set_caption("Breakthrough")
        screen = pygame.display.set_mode((game.window_size, game.window_size))
        font = pygame.font.SysFont(None, 36)    
        clock = pygame.time.Clock()
        refresh(game, screen)
        pygame.display.flip()
        clock.tick(30)

    else:
        screen = None
        clock = None

    while game.state == GameState.OnGoing:
        if play_against_me and game.player == Player.White:
            move = my_move(game, screen, clock, use_gui)
        elif not play_against_me and game.player == Player.Black:
            move = white_player.choose_move(game, num_iterations=iteration)
        else:
            move = black_player.choose_move(game, num_iterations=iteration)

        if record:
            records["move_" + str(len(records))] = {
                "state": game.encode(),
                "move": game.undecode(move[0], move[1]),
                "player": game.player
            }
        game.make_move(move[0], move[1])
        
        TIMER_EVENT = pygame.USEREVENT + 1
        pygame.time.set_timer(TIMER_EVENT, 10000)  # 10 שניות
        if use_gui:
            refresh(game, screen)
            pygame.display.flip()
            clock.tick(30)

    if record:
        export_game(records, game.state, game.board_size)

    if use_gui and not exit_on_finish:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                            
            winner = "White" if game.state == GameState.WhiteWon else "Black"
            text = font.render(f"{winner} Won!!", True, Colors.Green)
            screen.blit(text, (10, 10)) 
            pygame.display.flip()
            clock.tick(30)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for _ in range(5*1000):
        main()  # Change to False to run without GUI
