import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense


MAX_SUM = 21

class Action:
    HIT = 0
    STAY = 1

    @staticmethod
    def get_actions():
        return [Action.HIT, Action.STAY]

def question_a(num_of_train_episodes, num_of_test_episodes, epsilon, gamma):
    # Intial Q_table
    Q_table = []
    for i in range(MAX_SUM+1):
        Q_table.append({"q": random.randint(0,MAX_SUM), "n":0})

    # train
    for i in range(num_of_train_episodes):
        episode, reward = generate_episode(Q_table, epsilon)
        update_Q_table(Q_table, episode, reward, gamma)
    
    # test
    result = 0
    for _ in range(num_of_test_episodes):
        result += test(Q_table)
    print("The test result is:", result/num_of_test_episodes)
    return result/num_of_test_episodes

def question_b(num_of_train_episodes, num_of_test_episodes, epsilon, gamma):
    # Intial Q_table
    vector_reward = np.array(range(MAX_SUM + 1))
    random.shuffle(vector_reward)

    x_train = np.zeros((1,MAX_SUM*2 + 2), dtype=int)
    y_train = np.zeros((1,1), dtype=int)

    model = Sequential()
    model.add(Input(shape=(x_train.shape[1],)))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    
    for i in range(num_of_train_episodes):
        print("Running episode:", i)
        episode, reward = model_episode(model, vector_reward)
        x_train[0, :MAX_SUM + 1] = vector_reward
        x_train[0, MAX_SUM + 1:] = episode
        y_train[0, 0] = reward      
        model.fit(x_train, y_train, epochs=1)
        
            
    # test
    result = 0
    for _ in range(num_of_test_episodes):
        result += test_model(model, vector_reward)
    print("The test result is:", result/num_of_test_episodes)
    return result/num_of_test_episodes

def generate_episode(Q_table, epsilon):
    state = 0
    hit_states = []
    while state <= MAX_SUM:
        if random.random() < epsilon:
            action = random.choice(Action.get_actions())
        else: 
            if state > Q_table[state]["q"]:
                action = Action.STAY
            else:
                action = Action.HIT
        if action == Action.STAY:
            return hit_states, state
        
        hit_states.append(state)
        state += random.randint(1, 10)

    return hit_states, 0

def state_to_OneHotVec(state):
    state_vec = np.zeros(MAX_SUM + 1, dtype=int)
    state_vec[state] = 1
    return state_vec

def model_episode(model, vector_reward):
    state = 0
    hit_state = state
    while state <= MAX_SUM:
        hit_state = state
        if vector_reward[hit_state] > model.predict(np.concatenate((vector_reward, state_to_OneHotVec(hit_state)), dtype=int).reshape(1, MAX_SUM * 2 + 2)):
            return state_to_OneHotVec(hit_state), vector_reward[state]
        state += random.randint(1, 10)
    return state_to_OneHotVec(hit_state), 0

def update_Q_table(Q_table, episode, reward, gamma):
    sum_reward = 0
    reward_to_update = -1
    if len(episode) > 3:
        x =1
    for i, state in enumerate(reversed(episode)):
        Q_table[state]["n"] += 1
        reward_to_update = reward if reward_to_update == -1 else 0 + gamma * reward_to_update # R_n = r_n + gamma * R_n+1

        Q_table[state]["q"] += (reward_to_update - Q_table[state]["q"])/Q_table[state]["n"]

def test(Q_table):
    state = 0
    while state <= MAX_SUM:
        if state > Q_table[state]["q"]:
            return state
       
        state += random.randint(1, 10)
    return 0

def test_model(model, vector_reward):
    state = 0
    while state <= MAX_SUM:
        if vector_reward[state] > model.predict(np.concatenate((vector_reward, state_to_OneHotVec(state)), dtype=int).reshape(1, MAX_SUM * 2 + 2)):
            return vector_reward[state]
       
        state += random.randint(1, 10)
    return 0

if __name__ == "__main__":
    # Init parameters
    num_of_train_episodes = 200
    num_of_test_episodes = 100
    iteration = 1
    epsilon = 0.1
    gamma = 0.9
    res = 0
    for _ in range(iteration):
        # res += question_a(num_of_train_episodes, num_of_test_episodes, epsilon, gamma)
        res += question_b(num_of_train_episodes, num_of_test_episodes, epsilon, gamma)
    
    print("Average:", round(res/iteration,2))
# # Initialize the Q-table
# returns = {(s, a): [] for s in range(MAX_SUM + 1) for a in range(len(ACTIONS))}  # For storing returns

# # Hyperparameters
# num_training_episodes = 5000
# discount_factor = 1.0  # No discounting
# epsilon = 0.1  # Epsilon-greedy exploration
# learning_rate = 0.1


# def generate_episode(policy):
#     """Generate an episode following the given policy."""
#     episode = []
#     S = 0  # Initial sum

#     while True:
#         if S > MAX_SUM:
#             # Bust: End the episode with reward 0
#             episode.append((S, None, 0))
#             break

#         # Choose an action based on the policy
#         action = policy(S)

#         if action == "Stay":
#             # Stay: Reward is the current sum
#             episode.append((S, action, S))
#             break
#         elif action == "Hit":
#             # Hit: Draw a random card (1-10)
#             episode.append((S, action, 0))
#             S += random.randint(1, 10)

#     return episode


# def epsilon_greedy_policy(S):
#     """Epsilon-greedy policy based on the Q-table."""
#     if S > MAX_SUM:
#         return None
#     if random.random() < epsilon:
#         return random.choice(ACTIONS)
#     else:
#         return ACTIONS[np.argmax(Q_table[S])]


# def update_policy(episode):
#     """Update Q-table using Monte Carlo first-visit updates."""
#     G = 0  # Cumulative reward
#     visited = set()

#     for S, action, reward in reversed(episode):
#         if action is None:
#             continue
#         action_idx = ACTIONS.index(action)
#         state_action = (S, action_idx)

#         if state_action not in visited:
#             visited.add(state_action)
#             G = reward + discount_factor * G
#             returns[state_action].append(G)
#             Q_table[S, action_idx] = np.mean(returns[state_action])


# # Training
# for _ in range(num_training_episodes):
#     episode = generate_episode(epsilon_greedy_policy)
#     update_policy(episode)

# # Derived policy after training
# def learned_policy(S):
#     if S > MAX_SUM:
#         return None
#     return ACTIONS[np.argmax(Q_table[S])]


# # Testing the learned policy
# total_reward = 0
# num_test_episodes = 1000

# for _ in range(num_test_episodes):
#     S = 0
#     while True:
#         action = learned_policy(S)
#         if action == "Stay":
#             total_reward += S
#             break
#         elif action == "Hit":
#             S += random.randint(1, 10)
#         if S > MAX_SUM:
#             break

# average_reward = total_reward / num_test_episodes
# print(f"Average reward over {num_test_episodes} episodes: {average_reward:.2f}")
