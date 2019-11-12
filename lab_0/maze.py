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

# Define action space
actions = {}
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

# Define reward space
rewards = {}
for state in states:
    if state == (5, 5):
        rewards[state] = 0
    else:
        rewards[state] = -1
