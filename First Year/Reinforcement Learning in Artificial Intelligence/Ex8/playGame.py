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
    PUCTv1_1 = 1.1 # exploration 2 + 7k iter
    PUCTv2 = 2
    PUCTv2_1 = 2.1 # exploration 2 + 7k iter
    MCTS = 3


def build_player(player_type: PlayerType, player: Player, board_size: int, batch_size:int, exploration: int, train_model=False):
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
        return PUCTPlayer(game_model, exploration, training=train_model)
    else:
        return MCTSPlayer(player, exploration_weight=exploration)

def main(game_num: int, board_size, batch_size, iteration, exploration, learning_rate, use_gui, train_model, export_game, white_player_type, black_player_type):
    if (white_player_type == PlayerType.USER or black_player_type == PlayerType.USER) and not use_gui:
        exit("Error: You must Gui to play by yourself.")

    records = {}
    game = Breakthrough(board_size=board_size)

    white_player = build_player(white_player_type, Player.White, board_size, batch_size, exploration, train_model)
    black_player = build_player(black_player_type, Player.Black, board_size, batch_size, exploration, train_model)
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
        
        if game.player == Player.White:
            if white_player is None:
                move = my_move(game, screen, clock, use_gui)
            else:
                move, q_value, policy_distribution = white_player.choose_move(game, num_iterations=iteration)
        else:
            if black_player is None:
                move = my_move(game, screen, clock, use_gui)
            else:
                move, q_value, policy_distribution = black_player.choose_move(game, num_iterations=iteration)            
        
        if move is None:
            refresh(game, screen)
            pygame.display.flip()
            clock.tick(30)
            continue
        
        if train_model:
            records["move_" + str(len(records))] = {
                "state": game.encode(),
                "move": game.undecode(move[0], move[1]) if not policy_distribution else policy_distribution,
                "player": game.player,
                "winner": q_value
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

    if train_model:
        game_data_loader = create_data_loader(records, game.state)
        if white_player_type.name.startswith("PUCT") and "_" in white_player_type.name:
            avg_loss = train_game(white_player.model, game_data_loader, learning_rate)
        if black_player_type.name.startswith("PUCT") and "_" in  black_player_type.name:
            if black_player_type.name != white_player_type.name:
                avg_loss = train_game(black_player.model, game_data_loader, learning_rate)
            else:
                black_player.model.load_weights()
        
        if not export_game:
            records.clear()

    if export_game:
        export_game_records(records, game.state.value, game.board_size)
        records.clear()
    
    print(f"Game #{game_num} - White: {white_player_type.name}, Black: {black_player_type.name} - {game.state.name}, winning: {game.state}, steps: {moves_counter}, avg_loss: {avg_loss:.6f}", flush=True)
    return game.state, moves_counter

def create_data_loader(records, winner):
    records_list = [records[key] for key in records.keys()]
    for record in records_list:
        if record["winner"] is None:
            record["winner"] = winner

    dataset = GameDataset(records_list)
    return DataLoader(dataset, batch_size=1, shuffle=False)



def train_game(model:GameNetwork, data_loader, learning_rate):
    avg_loss = game_network.train(model, data_loader, val_loader=None, epochs=1, lr=learning_rate)
    model.save_weights()
    return avg_loss

def evaluate_game(game_num, model, data_loader, player_type, winner, total_loss, total_policy_acc, total_value_acc, moves_counter):
    avg_val_loss, val_policy_acc, val_value_acc = game_network.evaluate(model, data_loader)
    print(f"\nGame #{game_num} - Black Model - {player_type} - {winner} - Loss: {avg_val_loss:.6f}, Policy Accuracy: {val_policy_acc:.2%}, Value Accuracy: {val_value_acc:.2%}, Moves: {moves_counter}", flush=True)    
    return total_loss + avg_val_loss, total_policy_acc + val_policy_acc, total_value_acc + val_value_acc

def print_game_results(winners, total_moves):
    results = {}
    for winner in winners:
        if winner[0] not in results:
            results[winner[0]] = {"wins": 0, "avg":0}
        results[winner[0]]["avg"] = (results[winner[0]]["wins"] * results[winner[0]]["avg"] + winner[1]) / (results[winner[0]]["wins"] + 1)
        results[winner[0]]["wins"] += 1
    print("\n############### GAME RESULTS ###############")
    for player in results:
        print("#### Player:", player, "wins:", results[player]["wins"], "avg:",results[player]["avg"])

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    start_time = datetime.now()
    print(f"Training started at: {start_time}", flush=True)
    
    num_of_games = 2
    iteration   = 1*1000
    exploration = 1.2
    learning_rate = 0.001
    trained_player_types = [PlayerType.PUCTv1, 
                            PlayerType.PUCTv1_1
                            ]
    
    board_size  = 5
    batch_size = 512
    use_gui         = False
    train_model     = True
    export_game     = False

    print(f"\n############# Configuration #############", flush=True)
    print(f"num_of_games: {num_of_games}", flush=True)
    print(f"iteration: {iteration}", flush=True)
    print(f"exploration: {exploration}", flush=True)
    print(f"learning_rate: {learning_rate}", flush=True)
    print(f"trained_player_types: {[p.name for p in trained_player_types]}", flush=True)
    print(f"board_size: {board_size}", flush=True)
    print(f"batch_size: {batch_size}", flush=True)
    print(f"use_gui: {use_gui}", flush=True)
    print(f"train_model: {train_model}", flush=True)
    print(f"export_game: {export_game}", flush=True)
    print(f"########################################\n", flush=True)

    total_moves = 0
    winners = []
    elo = EloRating(agents=[p.name for p in trained_player_types])    

    for i in range(num_of_games):
        print(f"Time: {datetime.now()}, iteration: {i+1}", flush=True)
        
        winner, moves_counter = main(game_num=(i + 1),
                                     board_size=board_size,
                                     batch_size=batch_size,
                                     iteration=iteration,
                                     exploration=exploration,
                                     learning_rate=learning_rate,
                                     use_gui=use_gui,
                                     train_model=train_model,
                                     export_game=export_game,
                                     white_player_type = trained_player_types[i%2],
                                     black_player_type = trained_player_types[(i+1)%2])
        
        if winner == GameState.WhiteWon:
            elo.update_ratings(winner=trained_player_types[(i%2)].name, loser=trained_player_types[(i+1)%2].name)
        else:
            elo.update_ratings(winner=trained_player_types[((i+1)%2)].name, loser=trained_player_types[(i)%2].name)
        elo.print_leaderboard(summary=False)
        total_moves += moves_counter
        winners.append((winner.name, moves_counter))  # Change to False to run without GUI
    
    elo.print_leaderboard(summary=True)
    print_game_results(winners, total_moves)
    
    end_time = datetime.now()
    print(f"\nTraining completed at: {end_time}", flush=True)
    print(f"Total training time: {end_time - start_time}", flush=True)
