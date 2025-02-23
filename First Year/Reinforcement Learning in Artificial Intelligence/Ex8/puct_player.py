import math
import numpy as np
import torch
from breakthrough import Breakthrough
from game_network import GameNetwork, GameDataset
from puct_node import PUCTNode

class PUCTPlayer:
    def __init__(self, model, c_puct, training):
        self.model = model
        self.c_puct = c_puct
        self.training = training

    def choose_move(self, game, num_iterations=1000):
        root = PUCTNode(parent=None, prior_prob=1) #TODO Rafuz is wondering if it is needed :)

        for _ in range(num_iterations):
            node = root
            game_state: Breakthrough = game.clone()

            # Selection: Traverse the tree to find the best node to expand
            while not node.is_leaf():
                node = node.rand_child() if self.training else node.best_child(self.c_puct)
                move = game_state.decode(node.move_idx)
                game_state.make_move(move[0], move[1])

            value, action_probs = self.evaluate_state(game_state)
            node.expand(action_probs)

            # Backpropagation: Update the tree with the simulation result
            node.backpropagate(value)

        # Choose action with the highest visit count
        action_visits = {action: child.visit_count for action, child in root.children.items()}
        return game_state.decode(max(action_visits, key=action_visits.get))

    def evaluate_state(self, game_state: Breakthrough):
        """Evaluates the state using the neural network."""
        with torch.no_grad():
            value, policy = self.model(torch.tensor(game_state.encode() , dtype=torch.float))
        # Convert policy output to a dictionary of action probabilities
        return value.item(), self.get_action_probs(game_state, {i: prob for i, prob in enumerate(policy.squeeze().numpy())})

    def get_action_probs(self, game_state:Breakthrough, policy):
        action_probs = dict()
        for move in game_state.legal_moves():
            move_idx = game_state.undecode(move[0], move[1])
            action_probs[move_idx] = policy[move_idx]
        # Normalize the fixed_action_probs values to sum to 1
        total = sum(action_probs.values())
        if total > 0:
            action_probs = {k: v / total for k, v in action_probs.items()}
        return action_probs        
