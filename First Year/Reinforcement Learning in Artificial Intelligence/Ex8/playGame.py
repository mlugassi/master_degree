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
from torch.utils.data import DataLoader
import json
import os
import time
from datetime import datetime
import warnings
from elo_rating import EloRating
import random
import copy
import torch

warnings.simplefilter(action='ignore', category=FutureWarning)

seconds = time.time()

def my_move(game: Breakthrough, screen=None, clock=None, use_gui=True):
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN and game.state == GameState.OnGoing:
                mx, my = pygame.mouse.get_pos()
                
                # בדיקה האם נלחץ כפתור Undo
                if undo_button_rect.collidepoint(mx, my):
                    game.undo()
                    return None
                
                # חישוב קואורדינטות של הלוח עם השוליים
                x, y = (mx - game.MARGIN) // game.tile_size, (my - game.MARGIN) // game.tile_size
                
                if 0 <= x < game.board_size and 0 <= y < game.board_size:
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
            pygame.draw.rect(screen, color, 
                             (game.MARGIN + x * game.tile_size, game.MARGIN + y * game.tile_size, 
                              game.tile_size, game.tile_size))

            piece = game.board[y][x]
            if piece != 0:
                piece_color = Colors.White if piece == 1 else Colors.Black
                pygame.draw.circle(screen, piece_color, 
                                   (game.MARGIN + x * game.tile_size + game.tile_size // 2, 
                                    game.MARGIN + y * game.tile_size + game.tile_size // 2), 
                                   game.tile_size // 3)
                if piece == 1:
                    pygame.draw.circle(screen, Colors.Black, 
                                       (game.MARGIN + x * game.tile_size + game.tile_size // 2, 
                                        game.MARGIN + y * game.tile_size + game.tile_size // 2), 
                                       game.tile_size // 3, 2)
                    
def draw_selection(game:Breakthrough, screen):
    if game.selection:
        x, y = game.selection
        pygame.draw.rect(screen, Colors.LightGray, (game.MARGIN + x * game.tile_size, game.MARGIN + y * game.tile_size, game.tile_size, game.tile_size), 3)
        for move in game.valid_moves((x, y)):
            mx, my = move
            pygame.draw.rect(screen, Colors.Green, (game.MARGIN + mx * game.tile_size, game.MARGIN + my * game.tile_size, game.tile_size, game.tile_size), 3)

def draw_undo_button(game:Breakthrough, screen):
    button_rect = pygame.Rect(game.MARGIN, game.window_size - 50, 100, 40)  # הזזה למיקום טוב
    pygame.draw.rect(screen, Colors.Gray, button_rect)
    font = pygame.font.SysFont(None, 30)
    text = font.render("Undo", True, Colors.White)
    screen.blit(text, (game.MARGIN + 30, game.window_size - 40))
    return button_rect
      
def refresh(game, screen):
    screen.fill(Colors.Black)
    draw_board(game, screen)
    draw_selection(game, screen)
    global undo_button_rect
    undo_button_rect = draw_undo_button(game, screen)

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
    PUCTv1_1 = 1.1
    PUCTv1_2 = 1.2 #againts itslef
    PUCTv1_3 = 1.3
    PUCTv1_4 = 1.4

    PUCTv2 = 2
    PUCTv2_1 = 2.1 # exploration 1.2 + 2k iter lr 0.001
    PUCTv2_2 = 2.2 # exploration 0.8 + 2k iter lr 0.0001
    MCTS = 3


def build_player(player_type: PlayerType, player: Player, board_size: int, batch_size:int, train_model=False):
    if player_type == PlayerType.USER:
        return None
    if player_type.name.startswith("PUCT"):
        weights_name = f"game_network_weights_{board_size}_batch_{batch_size}_v{player_type.value}.pth"
        game_model = GameNetwork(board_size=board_size, weights_name=weights_name)
        if os.path.isfile(weights_name):
            game_model.load_weights(train=False)
        else:
            print("Error: weights file:", weights_name, "didn't found")
            exit(1)
        return PUCTPlayer(game_model, training=train_model)
    else:
        return MCTSPlayer(player)

def main(game_num: int, board_size, batch_size, iteration, exploration, learning_rate, use_gui, train_model, export_game, white_player_type, black_player_type, add_randomizion):
    if (white_player_type == PlayerType.USER or black_player_type == PlayerType.USER) and not use_gui:
        exit("Error: You must Gui to play by yourself.")

    records = {}
    game = Breakthrough(board_size=board_size)

    white_player = build_player(white_player_type, Player.White, board_size, batch_size, train_model)
    black_player = build_player(black_player_type, Player.Black, board_size, batch_size, train_model)
    moves_counter = 0

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
        policy_distribution = None
        q_value = None        
        avg_loss = None

        if game.player == Player.White:
            if white_player is None:
                move = my_move(game, screen, clock, use_gui)
            else:
                num_of_iteration = iteration
                exploration_weight = exploration
                if add_randomizion:
                    num_of_iteration = num_of_iteration if not add_randomizion else int(num_of_iteration * (random.randrange(80, 121)/100))
                    exploration_weight = exploration_weight if not add_randomizion else (exploration_weight * (random.randrange(80, 121)/100))
                move, q_value, policy_distribution = white_player.choose_move(game, num_of_iteration, exploration_weight)
        else:
            if black_player is None:
                move = my_move(game, screen, clock, use_gui)
            else:
                num_of_iteration = iteration
                exploration_weight = exploration
                if add_randomizion:
                    num_of_iteration = num_of_iteration if not add_randomizion else int(num_of_iteration * (random.randrange(80, 121)/100))
                    exploration_weight = exploration_weight if not add_randomizion else (exploration_weight * (random.randrange(80, 121)/100))
                
                if black_player_type == PlayerType.MCTS:
                    move = black_player.choose_move(game, num_of_iteration, exploration_weight)            
                else:
                    move, q_value, policy_distribution = black_player.choose_move(game, num_of_iteration, exploration_weight)            
        
        if move is None:
            refresh(game, screen)
            pygame.display.flip()
            clock.tick(30)
            continue
        
        if train_model:
            records[f"game_{game_num}_move_{len(records)}"] = {
                "state": game.encode(),
                "move": game.undecode(move[0], move[1]) if policy_distribution is None else policy_distribution,
                # "move": game.undecode(move[0], move[1]),
                "player": game.player,
                "winner": q_value
                # "winner": None
            }
        game.make_move(move[0], move[1])
        moves_counter += 1
        
        if use_gui:
            refresh(game, screen)
            pygame.display.flip()
            clock.tick(30)
    
    
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

    # if train_model:
        # records = create_data_loader(records, game.state, batch_size)
        # if white_player_type.name.startswith("PUCT") and "_" in white_player_type.name:
        #     avg_loss = train_game(white_player.model, game_data_loader, learning_rate, batch_size)
        # if black_player_type.name.startswith("PUCT") and "_" in  black_player_type.name:
        #     if black_player_type.name != white_player_type.name:
        #         avg_loss = train_game(black_player.model, game_data_loader, learning_rate, batch_size)
        #     else:
        #         black_player.model.load_weights()

    # if export_game:
    #     export_game_records(records, game.state.value, game.board_size)
    #     records.clear()
    
    print(f"Game #{game_num} - White: {white_player_type.name}, Black: {black_player_type.name} - winning: {game.state.name}, steps: {moves_counter}", flush=True)
    return game.state, moves_counter, records, white_player.model, black_player.model

def update_winner(records, winner):
    records_list = [records[key] for key in records.keys()]
    for record in records_list:
        if record["winner"] is None:
            record["winner"] = winner
    return records_list

def create_data_loader(records: dict, batch_size):
    records_list = list()
    left_recs = copy.deepcopy(records)
    for i, (key, val) in enumerate(records.items()):
        if i < batch_size:
            records_list.append(val)
            del left_recs[key]
        else:
            break
    dataset = GameDataset(records_list)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False), left_recs

def train_game(model:GameNetwork, data_loader, learning_rate):
    avg_loss = game_network.train(model, data_loader, val_loader=None, epochs=1, lr=learning_rate)
    model.save_weights()
    return avg_loss

def evaluate_game(game_num, model, data_loader, player_type, winner, total_loss, total_policy_acc, total_value_acc, moves_counter):
    avg_val_loss, val_policy_acc, val_value_acc = game_network.evaluate(model, data_loader)
    print(f"\nGame #{game_num} - Black Model - {player_type} - {winner} - Loss: {avg_val_loss:.6f}, Policy Accuracy: {val_policy_acc:.2%}, Value Accuracy: {val_value_acc:.2%}, Moves: {moves_counter}", flush=True)    
    return total_loss + avg_val_loss, total_policy_acc + val_policy_acc, total_value_acc + val_value_acc

def print_game_results(wins_counter, moves_counter):
    print("\n############### GAME RESULTS ###############")
    for player in wins_counter:
        print(f"####  {player:<10} wins: {wins_counter[player]:<4} avg: {0 if wins_counter[player] == 0 else moves_counter[player]/wins_counter[player]:.2f}")

def print_config(num_of_games, iteration, exploration, learning_rate, train_model, trained_player_types, dynamic_player_color, add_randomizion, board_size, batch_size, use_gui, export_game):
    print(f"\n############# Configuration #############", flush=True)
    print(f"num_of_games: {num_of_games}", flush=True)
    print(f"iteration: {iteration}", flush=True)
    print(f"exploration: {exploration}", flush=True)
    print(f"learning_rate: {learning_rate}", flush=True)
    print(f"train_model: {train_model}", flush=True)
    print(f"trained_player_types: {[p.name for p in trained_player_types]}", flush=True)
    print(f"dynamic_player_color: {dynamic_player_color}", flush=True)
    print(f"add_randomizion: {add_randomizion}", flush=True)
    print(f"board_size: {board_size}", flush=True)
    print(f"batch_size: {batch_size}", flush=True)
    print(f"use_gui: {use_gui}", flush=True)
    print(f"export_game: {export_game}", flush=True)
    print(f"########################################\n", flush=True)    

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    start_time = datetime.now()
    print(f"Training started at: {start_time}", flush=True)
    
    print("CUDA Available:", torch.cuda.is_available(), flush=True)
    print("CUDA Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)    
    
    num_of_games  = 1000
    iteration     = 2*1000
    exploration   = 1.2
    learning_rate = 0.001
    train_model   = False
    trained_player_types = [PlayerType.PUCTv1, 
                            PlayerType.PUCTv1_2
                            ]
    dynamic_player_color = trained_player_types[0] != trained_player_types[1] and train_model
    add_randomizion = not train_model

    board_size  = 5
    batch_size = 512
    use_gui         = False

    export_game     = False

    print_config(num_of_games, iteration, exploration, learning_rate, train_model, trained_player_types, dynamic_player_color, add_randomizion, board_size, batch_size, use_gui, export_game)

    elo = EloRating(agents=[(p.name if dynamic_player_color else ("White" if i == 0 else "Black")) for i, p in enumerate(trained_player_types)])    

    wins_counter = {}
    moves_counter = {}
    batch_recs = {}
    for agent in elo.ratings.keys():
        wins_counter[agent] = 0
        moves_counter[agent] = 0

    for i in range(num_of_games):
        print(f"\nTime: {datetime.now()}, iteration: {i+1}", flush=True)
        white_idx = i%2 if dynamic_player_color else 0
        black_idx = (i+1)%2 if dynamic_player_color else 1
        white_player_name = trained_player_types[white_idx].name if dynamic_player_color else "White"
        black_player_name = trained_player_types[black_idx].name if dynamic_player_color else "Black"

        winner, steps, game_rec, white_model, black_model = main(game_num=(i + 1),
                                                                    board_size=board_size,
                                                                    batch_size=batch_size,
                                                                    iteration=iteration,
                                                                    exploration=exploration,
                                                                    learning_rate=learning_rate,
                                                                    use_gui=use_gui,
                                                                    train_model=train_model,
                                                                    export_game=export_game,
                                                                    white_player_type = trained_player_types[white_idx],
                                                                    black_player_type = trained_player_types[black_idx],
                                                                    add_randomizion=add_randomizion)
        if train_model:
            batch_recs = batch_recs | game_rec
            if len(batch_recs) >= batch_size or i == (num_of_games - 1):
                game_data_loader, batch_recs = create_data_loader(batch_recs, batch_size)
                if trained_player_types[white_idx].name.startswith("PUCT") and "_" in trained_player_types[white_idx].name:
                    avg_loss = train_game(white_model, game_data_loader, learning_rate)
                if trained_player_types[black_idx].name.startswith("PUCT") and "_" in  trained_player_types[black_idx].name:
                    if trained_player_types[black_idx].name != trained_player_types[white_idx].name:
                        avg_loss = train_game(black_model, game_data_loader, learning_rate)
                    else:
                        black_model.load_weights()
         
        if winner == GameState.WhiteWon:
            rate_change_str = elo.update_ratings(winner=white_player_name, loser=black_player_name)
            wins_counter[white_player_name] += 1
            moves_counter[white_player_name] += steps
        else:
            rate_change_str = elo.update_ratings(black_player_name, loser=white_player_name)
            wins_counter[black_player_name] += 1
            moves_counter[black_player_name] += steps

        elo.print_leaderboard(summary=False, rate_change_str=rate_change_str)

    print_config(num_of_games, iteration, exploration, learning_rate, train_model, trained_player_types, dynamic_player_color, add_randomizion, board_size, batch_size, use_gui, export_game)
    elo.print_leaderboard(summary=True)
    print_game_results(wins_counter, moves_counter)
    
    end_time = datetime.now()
    print(f"\nTraining completed at: {end_time}", flush=True)
    print(f"Total training time: {end_time - start_time}", flush=True)
