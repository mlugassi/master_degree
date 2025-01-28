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
        best_move = max(root.children.items(), key=lambda item: (item[1].win_count/item[1].visit_count))[0]
        return best_move

    def simulate(self, game_state: Breakthrough):
        """Simulate a random game to completion and return the result."""
        while game_state.state == GameState.OnGoing:
            move = random.choice(game_state.legal_moves())
            game_state.make_move(move[0], move[1])
        return game_state.state