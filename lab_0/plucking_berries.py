import numpy as np

w = np.iinfo(np.int16).min

maze = np.array([
    [0, 1, w, 10, 10, 10, 10],
    [0, 1, w, 10, 0, 0, 10],
    [0, 1, w, 10, 0, 0, 10],
    [0, 1, 1, 1, 0, 0, 10],
    [0, w, w, w, w, w, 10],
    [0, 0, 0, 0, 0, 11, 10],
])

# Define state space
states = []
for i in range(maze.shape[0]):
    for j in range(maze.shape[1]):
        states.append((i, j))
print("States:", states)

# Define action space
T = 20
actions = {}
u = {}  # Initialize as 0
for i in range(maze.shape[0]):
    for j in range(maze.shape[1]):
        key = (i, j)
        value = [(i, j)]
        if (i, j - 1) in states:
            value.append((i, j - 1))
        if (i, j + 1) in states:
            value.append((i, j + 1))
        if (i + 1, j) in states:
            value.append((i + 1, j))
        if (i - 1, j) in states:
            value.append((i - 1, j))
        actions[key] = value
        T_values = np.ones(T)*(w-420)
        u[key] = (T_values, [None]*T)

print("Actions:", actions)

goal_state = (5, 5)
u[goal_state][0][T - 1] = 11
u[goal_state][1][T - 1] = goal_state


# Define reward space
rewards = {}
# To encode the probability of staying in that state, modify the reward as follows
# state_reward + ( num_of_times_might_be_stuck * probability_of_stuck * state_reward)
for state in states:
    rewards[state] = maze[state[0], state[1]]

# u[start_state] = (0, start_state, T)
state_list = []

for possible_state in actions[goal_state]:
    if possible_state != goal_state:
        state_list.append((possible_state, T-2))

while len(state_list) != 0:

    cur_state, t = state_list.pop(0)
    if cur_state == goal_state:
        continue
    print("Current state:", cur_state)
    possible_states = actions[cur_state]

    max_reward = w-420
    best_action = None

    for possible_state in possible_states:
        if u[possible_state][0][t+1] != w-420:
            reward = rewards[possible_state] + u[possible_state][0][t+1]
            if reward > max_reward:
                max_reward = reward
                best_action = possible_state

    if max_reward > u[cur_state][0][t]:
        u[cur_state][0][t] = max_reward
        u[cur_state][1][t] = best_action
        if t > 0:
            for possible_state in possible_states:
                state_list.append((possible_state, t-1))

print("U:", u)

cur_state = (0, 0)
path = [cur_state]
t = np.argmax(u[cur_state][0])
cur_state = u[cur_state][1][t]
path.append(cur_state)
t +=1
while cur_state != (5, 5):
    cur_state = u[cur_state][1][t]
    t += 1
    path.append(cur_state)
print('Path: ', path)

maze[np.array(path)[:, 0], np.array(path)[:, 1]] = 9
print(maze)
