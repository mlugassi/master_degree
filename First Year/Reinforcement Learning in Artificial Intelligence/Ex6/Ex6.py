import random
import numpy as np
import sys
import os
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Input, Dense # type: ignore
from tensorflow.keras.initializers import RandomNormal, RandomUniform # type: ignore

# from clearml import Task 
# task = Task.init(project_name="Reinforcement_Learning", task_name="ex6") 

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
    # Intial rewards
    # vector_reward = np.array(range(MAX_SUM + 1))
    # random.shuffle(vector_reward)
    vector_reward = np.random.randint(1, MAX_SUM + 1, size=(MAX_SUM + 1,))
    vector_reward[0] = 0

    x_train = np.zeros((1,MAX_SUM*2 + 2), dtype=int)
    y_train = np.zeros((1,1), dtype=int)
    layers = [33,22,11]
    
    out = open("ex6_" + "_".join([str(l) for l in layers]) + ".log", "a")
    print("\n######################################################", file=out)
    print("Vector_reword:", vector_reward,file=out)

    model_name = "Jacky_model_" + "_".join([str(l) for l in layers]) + ".keras"
    if os.path.isfile(model_name):
        print("-I- Loading exists model:", file=out)
        model = load_model(model_name)
    else:
        model = Sequential()
        model.add(Input(shape=(x_train.shape[1],)))
        # model.add(Dense(32, activation='relu'))
        # model.add(Dense(16, activation='relu'))
        # model.add(Dense(22, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.05)))
        for layer in layers:
            model.add(Dense(layer, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    
    for i in range(num_of_train_episodes):
        print("Running episode:", i)
        episode, reward = model_episode(model, vector_reward, epsilon, out=out)
        for state in reversed(episode):
            x_train[0, :MAX_SUM + 1] = state_to_OneHotVec(state) # 0 - 21
            x_train[0, MAX_SUM + 1:] = vector_reward # 22 - 43
            y_train[0, 0] = reward
            model.fit(x_train, y_train)
            reward *= gamma


    model.save(model_name)
    # test
    predict_vector = []
    for i in range(MAX_SUM + 1):
        vector = np.concatenate((state_to_OneHotVec(i), vector_reward), dtype=int).reshape(1, MAX_SUM * 2 + 2)
        predict_vector.append(round(float(model.predict(vector, verbose = 0)[0][0]),2))
    print(f"pred_vector = {predict_vector}", file=out)

    result = 0
    for _ in range(num_of_test_episodes):
        result += test_model(model, vector_reward, out)
    print("The test result is:", result/num_of_test_episodes)
    print("The test result is:", result/num_of_test_episodes, file=out)

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

def model_episode(model, vector_reward, epsilon = 0, out = sys.stdout):
    state = 0
    episode = []
    while state <= MAX_SUM:
        if random.random() < epsilon:
            action = random.choice(Action.get_actions())
        else: 
            vector = np.concatenate((state_to_OneHotVec(state), vector_reward), dtype=int)
            predict = model.predict(vector.reshape(1, MAX_SUM * 2 + 2), verbose = 0)
            # print("vector:,", ",".join(map(str, vector)) , "predict:,", predict, file=out)
            if (state > 0 and vector_reward[state] > predict) or state + 1 >= MAX_SUM:
                action = Action.STAY
            else:
                action = Action.HIT
        
        if action == Action.STAY:
            return episode, vector_reward[state]
        
        episode.append(state)
        state += random.randint(1, 10)
    # return episode, 0
    return [episode[-1]], 0

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
        pred = model.predict(
            np.concatenate((state_to_OneHotVec(state), vector_reward),
                           dtype=int).reshape(1, MAX_SUM * 2 + 2), verbose = 0)
        if (state > 0 and vector_reward[state] > pred) or state + 1 >= MAX_SUM:
            print(f"Curr state: {state} ({vector_reward[state]}), pred = {pred}, action: 'STAY'", file=out)
            print(f"** END RES = {vector_reward[state]} ** ", file=out)
            return vector_reward[state]
        
        print(f"Curr state: {state} ({vector_reward[state]}), pred = {pred}, action: 'HIT'", file=out)
        state += random.randint(1, 10)
    print(f"** END RES = 0 ** ", file=out)
    return 0

if __name__ == "__main__":
    # Init parameters
    num_of_train_episodes = 1
    num_of_test_episodes = 1
    iteration = 10000
    epsilon = 0
    gamma = 0.9
    res = 0
    for i in range(iteration):
        epsilon = epsilon * 0.99
        print(f"Running iteration {i+1} from {iteration}.")
        # res += question_a(num_of_train_episodes, num_of_test_episodes, epsilon, gamma)
        res += question_b(num_of_train_episodes, num_of_test_episodes, epsilon, gamma)
        print("Current average:", round(res/(i + 1),2))
    print("Total average:", round(res/iteration,2))

    
