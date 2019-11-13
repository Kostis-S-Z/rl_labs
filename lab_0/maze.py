import numpy as np

maze = np.array([
    [1, 1, 0, 1, 1, 1, 1],
    [1, 1, 0, 1, 1, 1, 1],
    [1, 1, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
])

# Define state space
states = []
for i in range(maze.shape[0]):
    for j in range(maze.shape[1]):
        if maze[i][j] == 1:
            states.append((i, j))
print("States:", states)

# Define action space
T = 11
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
        T_values = np.ones(T)*(-420)
        u[key] = (T_values, [None]*T)

print("Actions:", actions)

start_state = (5, 5)
u[start_state][0][T-1] = 0
u[start_state][1][T-1] = start_state


# Define reward space
rewards = {}
# To encode the probability of staying in that state, modify the reward as follows
# state_reward + ( num_of_times_might_be_stuck * probability_of_stuck * state_reward)
for state in states:
    if state == start_state:
        rewards[state] = 0.0
    # elif state == (5, 0):
    #     rewards[state] = -7*0.5 - 1*0.5
    # elif state == (3, 6):
    #     rewards[state] = -2*0.5 - 1*0.5
    else:
        rewards[state] = -1.0

#u[start_state] = (0, start_state, T)
state_list = []

for possible_state in actions[start_state]:
    if possible_state != start_state:
        state_list.append((possible_state, T-2))

while len(state_list) != 0:

    cur_state, t = state_list.pop(0)
    if cur_state == start_state:
        continue
    print("Current state:", cur_state)
    possible_states = actions[cur_state]

    max_reward = -420
    best_action = None

    for possible_state in possible_states:
        if u[possible_state][0][t+1] != -420:
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
