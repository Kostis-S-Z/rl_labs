import numpy as np

"""
Problem 1: The Maze and the Random Minotaur

Reach the goal in the maze without the minotaur catching you!
"""

maze = np.array([
    [1, 1, 0, 1, 1, 1, 1, 1],
    [1, 1, 0, 1, 1, 0, 1, 1],
    [1, 1, 0, 1, 1, 0, 0, 0],
    [1, 1, 0, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 0, 1, 1, 1]
])

min_val = np.iinfo(np.int16).min

T = 20
m_punish = -1000
start_state = (0, 0)
goal_state = (6, 5)
minotaur_state = (6, 5)


def build_state_space(with_obstacles=True):
    """
    Define state space
    """
    states = []
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if with_obstacles:  # If with obstacles, states with zero will not be added
                if maze[i][j] == 1:
                    states.append((i, j))
            else:
                states.append((i, j))

    return states


def build_action_space(states, can_stay=True):
    """
    Define action space
    """
    actions = {}
    u = {}
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            key = (i, j)
            if can_stay:
                value = [(i, j)]
            else:
                value = []
            if (i, j - 1) in states:
                value.append((i, j - 1))
            if (i, j + 1) in states:
                value.append((i, j + 1))
            if (i + 1, j) in states:
                value.append((i + 1, j))
            if (i - 1, j) in states:
                value.append((i - 1, j))
            actions[key] = value

            T_values = np.ones(T) * min_val
            u[key] = (T_values, [None] * T)

    return actions, u


def build_prob_matrix(states, actions, init_state):
    prob_matrix = np.zeros(shape=(T, maze.shape[0], maze.shape[1]))

    prob_matrix[0][init_state] = 1.0

    for t in range(1, T):
        for state in states:
            prob_of_this_state = prob_matrix[t - 1][state]
            if prob_of_this_state != 0:
                possible_states = np.array(actions[state])

                prob_of_each_new_state = prob_of_this_state * (1 / possible_states.shape[0])

                prob_matrix[t][possible_states[:, 0], possible_states[:, 1]] += prob_of_each_new_state

    return prob_matrix


def find_path():
    state_list = [(goal_state, T - 1)]

    while len(state_list) != 0:

        cur_state, t = state_list.pop(0)

        possible_states = p_actions[cur_state]

        max_reward = min_val
        best_action = None

        if cur_state == goal_state:
            max_reward = 0
            best_action = goal_state
        else:
            for possible_state in possible_states:
                if u[possible_state][0][t + 1] != min_val:
                    m_prob_state_t = m_prob[t + 1][possible_state]  # Prob of minotaur being in this state at time t+1
                    punish = m_prob_state_t * m_punish  # "Reward" if you get caught
                    reward = (1 - m_prob_state_t) * -1  # Reward if you don't get caught
                    reward = reward + punish + u[possible_state][0][t + 1]  # Final reward of state plus future
                    if reward > max_reward:
                        max_reward = reward
                        best_action = possible_state

        if max_reward > u[cur_state][0][t]:
            u[cur_state][0][t] = max_reward
            u[cur_state][1][t] = best_action
            if t > 0:
                for possible_state in possible_states:
                    state_list.append((possible_state, t - 1))

    cur_state = start_state
    path = [cur_state]

    for t in range(1, T):
        cur_state = u[cur_state][1][t]
        path.append(cur_state)

    return path


test_t = range(1, 20)
for t in test_t:
    T = t
    print("T = ", T)

    player_state_space = build_state_space(with_obstacles=True)
    minotaur_state_space = build_state_space(with_obstacles=False)  # The minotaur can walk through obstacles

    p_actions, u = build_action_space(player_state_space, can_stay=True)
    m_actions, _ = build_action_space(minotaur_state_space, can_stay=True)  # The minotaur is always on the move

    m_prob = build_prob_matrix(minotaur_state_space, m_actions, minotaur_state)

    path = find_path()

    if path[-1] == goal_state:
        maze_with_solution = maze.copy()
        maze_with_solution[np.array(path)[:, 0], np.array(path)[:, 1]] = 9  # Add path to maze

        print('Path: ', path)
        print('Solution: \n', maze_with_solution)
