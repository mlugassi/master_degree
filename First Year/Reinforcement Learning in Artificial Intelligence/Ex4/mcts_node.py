class MCTSNode:
    def __init__(self, parent=None, move=None, game_state=None):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.win_count = 0
        self.move = move
        self.untried_moves = game_state.legal_moves() if game_state else []

    def is_fully_expanded(self):
        """Returns True if all legal moves have been expanded."""
        return len(self.untried_moves) == 0

    def best_child(self, exploration_weight=1.41):
        """
        Select the best child using the UCT formula.
        UCT = win_rate + exploration_weight * sqrt(log(parent_visit_count) / child_visit_count)
        """
        def uct_value(child):
            exploitation = child.win_count / (child.visit_count + 1e-6)  # Win rate
            exploration = exploration_weight * (2 * (self.visit_count)**0.5 / (child.visit_count + 1e-6))
            return exploitation + exploration

        return max(self.children.values(), key=uct_value)

    def expand(self, move, game_state):
        """Expand a new child for the given move and return the child node."""
        game_state.make(move)  # Apply the move to the game state
        child_node = MCTSNode(parent=self, move=move, game_state=game_state)
        self.children[move] = child_node
        self.untried_moves.remove(move)
        return child_node

    def backpropagate(self, result):
        """Backpropagate the result of a simulation."""
        self.visit_count += 1
        self.win_count += result
        if self.parent:
            self.parent.backpropagate(result)