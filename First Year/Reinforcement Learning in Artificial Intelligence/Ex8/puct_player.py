import math
import numpy as np
import torch
from breakthrough import Breakthrough
from game_network import GameNetwork, GameDataset
from puct_node import PUCTNode
from breakthrough_types import *
import random
import sys
import time
import json

class PUCTPlayer:
    def __init__(self, model, training):
        self.model = model
        self.training = training
        self.seconds = time.time()

    def choose_move(self, game: Breakthrough, num_iterations, c_puct):
        root = PUCTNode(parent=None, prior_prob=1) #TODO Rafuz is wondering if it is needed :)

        if self.training:
            blocking_move = self.get_blocking_move(game.clone())
            wining_move = self.get_wining_move(game.clone())
            # y_policy = [0 for _ in range((game.board_size ** 2) * 3)]
            y_policy = {i: 0 for i in range((game.board_size ** 2) * 3)}
            if wining_move:
                y_policy[game.undecode(wining_move[0], wining_move[1])] = 1
                return wining_move, 1, y_policy
            elif blocking_move:
                y_policy[game.undecode(blocking_move[0], blocking_move[1])] = 1
                return blocking_move, self.evaluate_state(game)[0], y_policy

        for _ in range(num_iterations):
            node = root
            game_state: Breakthrough = game.clone()

            # Selection: Traverse the tree to find the best node to expand
            while not node.is_leaf() and game_state.state == GameState.OnGoing:
                # print("state:", game_state.state)
                # print("board:", game_state.board)
                # print("player:", game_state.player)
                node = node.best_child(c_puct)
                move = game_state.decode(node.move_idx)
                game_state.make_move(move[0], move[1])
            
            if game_state.state == GameState.OnGoing:
                value, action_probs = self.evaluate_state(game_state)
                node.expand(action_probs)
            else:
                value = 0

            node.backpropagate(value)
        
        # Choose action with the highest visit count
        action_visits = {action: child.visit_count for action, child in root.children.items()}
        # out = open(f"puct_tree_{self.seconds}.json", "w")
        # save_puct_tree_as_json(root, out)
        total_visits = sum(action_visits.values())
        policy_distribution = {move: count / total_visits for move, count in action_visits.items()}
        if self.training:
            chosen_action = random.choices(list(action_visits.keys()), weights=list(action_visits.values()), k=1)[0]
        else:
            chosen_action = max(action_visits, key=action_visits.get)
        return game.decode(chosen_action), root.q_value, policy_distribution

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
        # total = torch.sum(torch.tensor(list(action_probs.values()) , dtype=torch.float))
        # if total > 0:
        #     action_probs = {k: v / total for k, v in action_probs.items()}
        # return action_probs        

        probs_tensor = torch.tensor(list(action_probs.values()), dtype=torch.float32)
        normalized_probs = torch.softmax(probs_tensor, dim=0)
        normalized_action_probs = dict(zip(action_probs.keys(), normalized_probs.tolist()))

        return normalized_action_probs
    
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
        y = 1 if game_state.player == Player.White else (game_state.board_size - 2)
        for move in game.legal_moves():
            if move[0][1] == y:
                return move
        return None
    
def puct_tree_to_dict(node:PUCTNode):
    """
    Recursively converts a PUCT tree node into a dictionary for JSON serialization.
    
    :param node: The root node of the tree.
    :return: A dictionary representing the tree structure.
    """
    return {
        "visits": node.visit_count,
        "value": round(node.q_value, 3),
        "prior": round(node.prior_prob, 3),
        "children": {
            action: puct_tree_to_dict(child)
            for action, child in node.children.items()
        } if node.children else None  # Only include children if they exist
    }

def save_puct_tree_as_json(root:PUCTNode, jsonname="puct_tree.json"):
    """
    Saves the PUCT tree to a JSON file for viewing in VS Code.

    :param root: The root node of the tree.
    :param filename: The name of the JSON file to save.
    """
    tree_dict = puct_tree_to_dict(root)
    json.dump(tree_dict, jsonname, indent=4)
    print(f"Tree saved to {jsonname}")