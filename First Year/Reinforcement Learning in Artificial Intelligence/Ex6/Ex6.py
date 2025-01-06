import random
import numpy as np

# Constants
MAX_SUM = 21
ACTIONS = ["Hit", "Stay"]  # Actions: 0 -> Hit, 1 -> Stay

# Initialize the Q-table
Q_table = np.zeros((MAX_SUM + 1, len(ACTIONS)))
returns = {(s, a): [] for s in range(MAX_SUM + 1) for a in range(len(ACTIONS))}  # For storing returns

# Hyperparameters
num_training_episodes = 5000
discount_factor = 1.0  # No discounting
epsilon = 0.1  # Epsilon-greedy exploration
learning_rate = 0.1


def generate_episode(policy):
    """Generate an episode following the given policy."""
    episode = []
    S = 0  # Initial sum

    while True:
        if S > MAX_SUM:
            # Bust: End the episode with reward 0
            episode.append((S, None, 0))
            break

        # Choose an action based on the policy
        action = policy(S)

        if action == "Stay":
            # Stay: Reward is the current sum
            episode.append((S, action, S))
            break
        elif action == "Hit":
            # Hit: Draw a random card (1-10)
            episode.append((S, action, 0))
            S += random.randint(1, 10)

    return episode


def epsilon_greedy_policy(S):
    """Epsilon-greedy policy based on the Q-table."""
    if S > MAX_SUM:
        return None
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    else:
        return ACTIONS[np.argmax(Q_table[S])]


def update_policy(episode):
    """Update Q-table using Monte Carlo first-visit updates."""
    G = 0  # Cumulative reward
    visited = set()

    for S, action, reward in reversed(episode):
        if action is None:
            continue
        action_idx = ACTIONS.index(action)
        state_action = (S, action_idx)

        if state_action not in visited:
            visited.add(state_action)
            G = reward + discount_factor * G
            returns[state_action].append(G)
            Q_table[S, action_idx] = np.mean(returns[state_action])


# Training
for _ in range(num_training_episodes):
    episode = generate_episode(epsilon_greedy_policy)
    update_policy(episode)

# Derived policy after training
def learned_policy(S):
    if S > MAX_SUM:
        return None
    return ACTIONS[np.argmax(Q_table[S])]


# Testing the learned policy
total_reward = 0
num_test_episodes = 1000

for _ in range(num_test_episodes):
    S = 0
    while True:
        action = learned_policy(S)
        if action == "Stay":
            total_reward += S
            break
        elif action == "Hit":
            S += random.randint(1, 10)
        if S > MAX_SUM:
            break

average_reward = total_reward / num_test_episodes
print(f"Average reward over {num_test_episodes} episodes: {average_reward:.2f}")
