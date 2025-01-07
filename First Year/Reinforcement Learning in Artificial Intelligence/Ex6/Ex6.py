import random
import numpy as np
import sys
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
        # Q_table.append({"q": random.randint(0,MAX_SUM), "n":0})
        Q_table.append({"q": random.randint(0,0), "n":0})

    # train
    for i in range(num_of_train_episodes):
        episode, reward = generate_episode(Q_table, epsilon)
        update_Q_table(Q_table, episode, reward, gamma)
    print(Q_table)
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
    out = open("ex6.log", "a")
    print("\n######################################################", file=out)
    print("Vector_reword:", vector_reward,file=out)

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
        episode, reward = model_episode(model, vector_reward, epsilon=0)
        x_train[0, :MAX_SUM + 1] = vector_reward
        x_train[0, MAX_SUM + 1:] = episode
        y_train[0, 0] = reward      
        model.fit(x_train, y_train, epochs=1)
        
            
    # test
    predict_vector = []
    for i in range(MAX_SUM + 1):
        predict_vector.append(round(float(model.predict(np.concatenate((vector_reward, state_to_OneHotVec(i)), dtype=int).reshape(1, MAX_SUM * 2 + 2))[0][0]),2))       
    print(f"pred_vector = {predict_vector}", file=out)

    result = 0
    for _ in range(num_of_test_episodes):
        result += test_model(model, vector_reward, out)
    print("The test result is:", result/num_of_test_episodes)
    print("The test result is::", result/num_of_test_episodes, file=out)

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

def model_episode(model, vector_reward, epsilon = 0):
    state = 0
    hit_state = state
    while state <= MAX_SUM:
        hit_state = state
        if random.random() < epsilon:
            action = random.choice(Action.get_actions())
        else: 
            if vector_reward[hit_state] > model.predict(np.concatenate((vector_reward, state_to_OneHotVec(hit_state)), dtype=int).reshape(1, MAX_SUM * 2 + 2)):
                action = Action.STAY
            else:
                action = Action.HIT
        if action == Action.STAY:
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

def test_model(model, vector_reward, out=sys.stdout):
    state = 0
    print(f"** START **", file=out)
    print(f"vector_reward = {vector_reward}", file=out)
    while state <= MAX_SUM:
        pred = model.predict(np.concatenate((vector_reward, state_to_OneHotVec(state)), dtype=int).reshape(1, MAX_SUM * 2 + 2))
        print(f"Curr state: {state} ({vector_reward[state]}), pred = {pred}, action: {'HIT' if pred >= vector_reward[state] else 'STAY'}", file=out)
        if vector_reward[state] > pred:
            print(f"** END RES = {vector_reward[state]} ** ", file=out)
            return vector_reward[state]
       
        state += random.randint(1, 10)
    print(f"** END RES = 0 ** ", file=out)
    return 0

if __name__ == "__main__":
    # Init parameters
    num_of_train_episodes = 2000
    num_of_test_episodes = 3
    iteration = 1
    epsilon = 0.1
    gamma = 0.9
    res = 0
    for _ in range(iteration):
        res += question_a(num_of_train_episodes, num_of_test_episodes, epsilon, gamma)
        #res += question_b(num_of_train_episodes, num_of_test_episodes, epsilon, gamma)
    
    print("Average:", round(res/iteration,2))