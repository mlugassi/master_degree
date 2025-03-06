class EloRating:
    def __init__(self, agents: list, default_rate=1500, k=32):
        self.k = k
        self.ratings = dict()
        for agent in agents:
            self.add_agent(agent, default_rate)

    def add_agent(self, agent_name, default_rate):
        if agent_name not in self.ratings:
            self.ratings[agent_name] = default_rate

    def expected_score(self, player, opponent):
        return 1 / (1 + 10 ** ((self.ratings[opponent] - self.ratings[player]) / 400))

    def update_ratings(self, winner, loser):
        expected_winner = self.expected_score(winner, loser)
        expected_loser = self.expected_score(loser, winner)

        self.ratings[winner] += self.k * (1 - expected_winner)
        self.ratings[loser] += self.k * (0 - expected_loser)

    def print_leaderboard(self, summary=False):
        if not summary:
            rate_str = "ELO:"
            for agent, rating in sorted(self.ratings.items(), key=lambda x: x[1], reverse=True):
                rate_str += f" {agent}: {rating:.2f}"
            print(rate_str)
        else:
            print("\n############### ELO SUMMARY ###############")
            for agent, rating in sorted(self.ratings.items(), key=lambda x: x[1], reverse=True):
                print(f"#### {agent:<10} {rating:.2f}")
# # Example usage
# elo = EloRating()
# elo.add_agent("Agent_A")
# elo.add_agent("Agent_B")

# # Simulate matches
# elo.update_ratings("Agent_A", "Agent_B")
# elo.update_ratings("Agent_B", "Agent_A")
# elo.update_ratings("Agent_A", "Agent_B")

# elo.leaderboard()