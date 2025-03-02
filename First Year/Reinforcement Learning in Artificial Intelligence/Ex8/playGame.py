from breakthrough import Breakthrough
from breakthrough_types import *
from mcts_node import MCTSNode
from mcts_player import MCTSPlayer
from puct_player import PUCTPlayer
from game_network import GameNetwork, GameDataset
import game_network
import pygame
import sys
from pygame.locals import *
import json
import os
import time
from datetime import datetime

seconds = time.time()

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

def export_game_records(records, winner, size):
    for record in records:
        records[record]["winner"] = winner
    with open(f"play_book_{size}_v2_{seconds}.json", "a") as f:
        json.dump(records, f)
        f.write("\n")

    # Load from file
    # with open("data.json", "r") as file:
    #     loaded_data = json.load(file)
class PlayerType(Enum):
    USER = 0
    PUCTv1 = 1
    PUCTv2 = 2
    PUCTv2_2 = 2.2
    MCTS = 3


def build_player(player_type: PlayerType, player: Player, board_size: int, batch_size:int, exploration: int, train_model=False):
    if player_type == PlayerType.USER:
        return None, None
    if player_type.name.startswith("PUCT"):
        game_model = GameNetwork(board_size)
        weights_name = f"game_network_weights_{board_size}_batch_{batch_size}_v{player_type.value}.pth"
        if os.path.isfile(weights_name):
            game_model.load_weights(weights_name, train=False)
        else:
            print("Error: weights file:", weights_name, "didn't found")
            exit(1)
        return PUCTPlayer(game_model, exploration, training=train_model), game_model
    else:
        return MCTSPlayer(player, exploration_weight=exploration), None

def main(game_num: int):
    # inputs
    board_size  = 5
    batch_size = 512
    iteration   = 1*1000
    exploration = 0.8
    use_gui         = False
    train_model     = True
    export_game     = False
    white_player_type = PlayerType.PUCTv2_2
    black_player_type = PlayerType.PUCTv2_2


    if (white_player_type == PlayerType.USER or black_player_type == PlayerType.USER) and not use_gui:
        exit("Error: You must Gui to play by yourself.")

    records = {}
    game = Breakthrough(board_size=board_size)

    white_player, white_player_model = build_player(white_player_type, Player.White, board_size, batch_size, exploration, train_model)
    black_player, black_player_model = build_player(black_player_type, Player.Black, board_size, batch_size, exploration, train_model)

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
        if game.player == Player.White:
            if white_player is None:
                move = my_move(game, screen, clock, use_gui)
            else:
                move = white_player.choose_move(game, num_iterations=iteration)
        else:
            if black_player is None:
                move = my_move(game, screen, clock, use_gui)
            else:
                move = black_player.choose_move(game, num_iterations=iteration)            

        if train_model:
            records["move_" + str(len(records))] = {
                "state": game.encode(),
                "move": game.undecode(move[0], move[1]),
                "player": game.player
            }
        game.make_move(move[0], move[1])
        
        if use_gui:
            refresh(game, screen)
            pygame.display.flip()
            clock.tick(30)
    print(f"Game #{game_num} - White: {white_player_type.name}, Black: {black_player_type.name} - {game.state.name}, winning: {game.state}, steps: {len(records)}", flush=True)
    if use_gui:
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

    if train_model:
        if white_player_model:
            train_game(white_player_model, records, game.state, f"game_network_weights_{board_size}_batch_{batch_size}_v{white_player_type.value}.pth")
        if black_player_model and white_player_type != black_player_type:
            train_game(black_player_model, records, game.state, f"game_network_weights_{board_size}_batch_{batch_size}_v{black_player_type.value}.pth")
        if not export_game:
            records.clear()

    if export_game:
        export_game_records(records, game.state.value, game.board_size)
        records.clear()



def train_game(model:GameNetwork, records, winner, weights_name):
    from torch.utils.data import DataLoader
    learning_rate = 0.0001

    records_list = [records[key] for key in records]
    for record in records_list:
        record["winner"] = winner
    
    dataset = GameDataset(records_list)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    game_network.train(model, train_loader, val_loader=None, epochs=1, lr=learning_rate)
    model.save_weights(weights_name)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    start_time = datetime.now()
    print(f"Training started at: {start_time}", flush=True)
    
    for i in range(1000):
        print(f"Time: {datetime.now()}, iteration: {i+1}", flush=True)
        main(i + 1)  # Change to False to run without GUI

    end_time = datetime.now()
    print(f"Training completed at: {end_time}", flush=True)
    print(f"Total training time: {end_time - start_time}", flush=True)    
