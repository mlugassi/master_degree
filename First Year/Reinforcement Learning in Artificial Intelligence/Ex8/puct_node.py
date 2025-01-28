import math
import numpy as np
import torch

class PUCTNode:
    def __init__(self, parent, prior_prob):
        self.parent = parent
        self.children = {}  # Maps actions to child nodes
        self.prior_prob = prior_prob
        self.visit_count = 0
        self.total_value = 0.0
        self.q_value = 0.0

    def is_leaf(self):
        return len(self.children) == 0

    def expand(self, action_probs):
        """Expands the node by creating child nodes based on action probabilities."""
        for action, prob in action_probs.items():
            if action not in self.children:
                self.children[action] = PUCTNode(parent=self, prior_prob=prob)

    def update(self, value):
        """Updates the node's statistics based on the value of a simulation."""
        self.visit_count += 1
        self.total_value += value
        self.q_value = self.total_value / self.visit_count

    def puct_score(self, c_puct):
        """Computes the PUCT score for all child nodes."""
        scores = {}
        total_visits = sum(child.visit_count for child in self.children.values())
        for action, child in self.children.items():
            u_value = c_puct * child.prior_prob * math.sqrt(total_visits) / (1 + child.visit_count)
            scores[action] = child.q_value + u_value
        return scores