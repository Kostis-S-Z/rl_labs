import numpy as np

# Define maze grid
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
        u[key] = (-420, None, None)

print("Actions:", actions)

# Define reward space
rewards = {}
for state in states:
    if state == (5, 5):
        rewards[state] = 0.0
    elif state == (5, 0):
        rewards[state] = -7*0.5 - 1*0.5
    elif state == (3, 6):
        rewards[state] = -2*0.5 -1 *0.5
    else:
        rewards[state] = -1.0

#start_state = (5, 5)
#state_list = [start_state]
T = 20

u[(5, 5)] = (0, (5,5), T)
state_list = []
for possible_state in actions[(5, 5)]:
    state_list.append(possible_state)

while len(state_list) != 0:

    cur_state = state_list.pop(0)
    print("Current state:", cur_state)
    possible_states = actions[cur_state]

    max_reward = -420
    best_action = None

    for possible_state in possible_states:
        if u[possible_state][0] != -420:
            reward = rewards[possible_state] + u[possible_state][0]
            if reward > max_reward:
                max_reward = reward
                best_action = possible_state
                t = u[possible_state][2]

    if max_reward > u[cur_state][0]:
        u[cur_state] = (max_reward, best_action, t-1)
        if t-1 > 0:
            for possible_state in possible_states:
                state_list.append(possible_state)

print("U:", u)

cur_state = (0, 0)
path = [cur_state]
while cur_state != (5, 5):
    cur_state = u[cur_state][1]
    path.append(cur_state)
print('Path: ', path)

maze[np.array(path)[:, 0], np.array(path)[:, 1]] = 9
print(maze)
