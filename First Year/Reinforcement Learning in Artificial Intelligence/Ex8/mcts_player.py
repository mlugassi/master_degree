import random

from mcts_node import MCTSNode
from breakthrough import *
from breakthrough_types import *

class MCTSPlayer:
    def __init__(self, player: Player, exploration_weight=0.2):
        self.player = player
        self.exploration_weight = exploration_weight

    def choose_move(self, game, num_iterations=1000):
        root = MCTSNode(player=self.player, game_state=game.clone())
        blocking_move = self.get_blocking_move(game.clone())
        wining_move = self.get_wining_move(game.clone())
        if wining_move:
            return wining_move
        elif blocking_move:
            return blocking_move
        
        for _ in range(num_iterations):
            node = root
            game_state: Breakthrough = game.clone()

            # Selection: Traverse the tree to find the best node to expand
            while not node.untried_moves and node.children:
                node = node.best_child(self.exploration_weight)
                game_state.make_move(node.move[0], node.move[1])

            # Expansion: Expand a new child node if possible
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                node = node.expand(move, game_state)

            # Simulation: Simulate a random playout
            result = self.simulate(game_state)

            # Backpropagation: Update the tree with the simulation result
            node.backpropagate(result)

        # Choose the move with the highest visit count
        # print("####################################################")
        # for item in root.children.items():
        #     print(item[0])
        #     print("\t", item[1].win_count, item[1].visit_count, "=", item[1].win_count/item[1].visit_count)
        best_move = max(root.children.items(), key=lambda item: (item[1].win_count/item[1].visit_count))[0]
        return best_move
    
    def get_blocking_move(self, game_state: Breakthrough):
        y = 1 if game_state.player == Player.Black else (game_state.board_size - 2)
        other_player_wining_position = []
        for x, p in enumerate(game_state.board[y]):
            if p == game_state.get_other_player():
                other_player_wining_position.append((x,y))
        
        for move in game_state.legal_moves():
            for position in other_player_wining_position:
                if move[1] == position:
                    return move
        return None
    
    def get_wining_move(self, game_state: Breakthrough):
        game = game_state.clone()
        for move in game.legal_moves():
            game.make_move(move[0], move[1])
            if game.state == self.player:
                return move
            game.unmake_move(prev_board=game_state.board)
        return None
    
    def simulate(self, game_state: Breakthrough):
        """Simulate a random game to completion and return the result."""
        while game_state.state == GameState.OnGoing:
            blocking_move = self.get_blocking_move(game_state.clone())
            wining_move = self.get_wining_move(game_state.clone())
            if wining_move:
                move = wining_move
            elif blocking_move:
                move = blocking_move
            else:
                move = random.choice(game_state.legal_moves())
            game_state.make_move(move[0], move[1])
        return game_state.state