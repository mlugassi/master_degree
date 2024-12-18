import random

from mcts_node import MCTSNode
from connect_four_types import *

class MCTSPlayer:
    def __init__(self, exploration_weight=1.41):
        self.exploration_weight = exploration_weight

    def choose_move(self, game, num_iterations=1000):
        root = MCTSNode(game_state=game.clone())

        for _ in range(num_iterations):
            node = root
            game_state = game.clone()

            # Selection: Traverse the tree to find the best node to expand
            while not node.untried_moves and node.children:
                node = node.best_child(self.exploration_weight)
                game_state.make(node.move)

            # Expansion: Expand a new child node if possible
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                node = node.expand(move, game_state)

            # Simulation: Simulate a random playout
            result = self.simulate(game_state)

            # Backpropagation: Update the tree with the simulation result
            node.backpropagate(result)

        # Choose the move with the highest visit count
        best_move = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        return best_move

    def simulate(self, game_state):
        """Simulate a random game to completion and return the result."""
        while game_state.status == GameStatus.ONGOING:
            move = random.choice(game_state.legal_moves())
            game_state.make(move)

        if game_state.status == Player.RED:
            return 1  # Win for RED
        elif game_state.status == Player.YELLOW:
            return 0  # Win for YELLOW
        else:
            return 0.5  # Draw