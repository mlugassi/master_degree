import math
import numpy as np
import torch
import random

class PUCTNode:
    def __init__(self, parent, prior_prob, move_idx=None):
        self.parent = parent
        self.children = {}  # Maps actions to child nodes
        self.prior_prob = prior_prob
        self.visit_count = 0
        # self.total_value = 0.0
        self.q_value = 0.0
        self.move_idx = move_idx

    def is_leaf(self):
        return len(self.children) == 0

    def expand(self, action_probs):
        """Expands the node by creating child nodes based on action probabilities."""
        for action, prob in action_probs.items():
            if action not in self.children:
                self.children[action] = PUCTNode(parent=self, prior_prob=prob, move_idx=action)

    def best_child(self, c_puct):
        def puct_score(self, c_puct):
            """Computes the PUCT score for all child nodes."""
            scores = {}
            for action, child in self.children.items():
                u_value = c_puct * child.prior_prob * (math.sqrt(self.visit_count) / (1 + child.visit_count))
                scores[action] = child.q_value + u_value
            return scores        
        scores = puct_score(c_puct)
        return self.children[max(scores, key=scores.get)]

    def rand_child(self):
        action_prob = {action: child.prior_prob for action, child in self.children.items()}
        actions, probs = zip(*action_prob.items())
        # print("action:", actions, "probs:", probs)
        selected_move = random.choices(actions, weights=probs, k=1)[0]
        return self.children[selected_move]        

    def backpropagate(self, value):
        """Updates the node's statistics based on the value of a simulation."""
        self.visit_count += 1
        self.q_value += (1 / self.visit_count) * (value - self.q_value)
        if self.parent:
            self.parent.backpropagate(1 - value)