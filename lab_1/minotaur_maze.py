import numpy as np
import matplotlib.pyplot as plt

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

np.random.seed(1)

global T
p_reward = 0  # Reward for every move
m_punish = -10000
player_state = (0, 0)
goal_state = (6, 5)
minotaur_state = (6, 5)
starting_state = (player_state, minotaur_state)


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

            for m_i in range(maze.shape[0]):
                for m_j in range(maze.shape[1]):
                    p_m_state = ((i, j), (m_i, m_j))  # Player - Minotaur state

                    T_values = np.ones(T) * min_val
                    u[p_m_state] = (T_values, [None] * T)

    return actions, u


def build_prob_matrix(states, actions, init_state, timesteps):
    """
    Create a matrix with the probabilities of the minotaur at each state when he begins from a initial state
    """
    prob_matrix = np.zeros(shape=(timesteps, maze.shape[0], maze.shape[1]))

    prob_matrix[0][init_state] = 1.0

    for t in range(1, timesteps):
        for state in states:
            prob_of_this_state = prob_matrix[t - 1][state]
            if prob_of_this_state != 0:
                possible_states = np.array(actions[state])

                prob_of_each_new_state = prob_of_this_state * (1 / possible_states.shape[0])

                prob_matrix[t][possible_states[:, 0], possible_states[:, 1]] += prob_of_each_new_state

    return prob_matrix


def a_random_walk(actions, init_state):
    """
    Generate a random walk of the minotaur in the maze
    """
    path_matrix = np.zeros(shape=(T, maze.shape[0], maze.shape[1]))

    m_path = [init_state]
    path_matrix[0][init_state] = 1.0

    state = init_state
    for t in range(1, T):
        possible_states = np.array(actions[state])

        pick_state = np.random.choice(possible_states.shape[0])  # Pick a random state
        next_state = possible_states[pick_state]

        path_matrix[t][next_state[0], next_state[1]] = 1.0
        state = tuple(next_state)
        m_path.append(state)

    return path_matrix, m_path


def a_random_walk_hybrid(actions, init_state):
    """
    Generate a random walk of the minotaur in the maze
    """
    path_matrix = np.zeros(shape=(T, maze.shape[0], maze.shape[1]))

    path_matrix[0][init_state] = 1.0

    state = init_state
    for t in range(1, T):
        possible_states = np.array(actions[state])

        n_possible_states = possible_states.shape[0]

        path_matrix[t][possible_states[:, 0], possible_states[:, 1]] = 1.0 / n_possible_states

        pick_state = np.random.choice(possible_states.shape[0])  # Pick a random state
        next_state = possible_states[pick_state]
        state = tuple(next_state)

    return path_matrix


def dp_max_prob(u, minotaur_prob):
    # Key is a tuple of:
    #   1) tuple of player state & minotaur state
    #   2) timestep

    state_list = [(goal_state, T-1)]

    while len(state_list) != 0:

        player_cur_state, t = state_list.pop(0)

        possible_player_states = p_actions[player_cur_state]

        possible_minotaur_states_list = []
        for i in range(maze.shape[0]):
            for j in range(maze.shape[1]):
                if minotaur_prob[t, i, j] > 0:
                    possible_minotaur_states_list.append((i, j))

        continue_path = False
        for minotaur_cur_state in possible_minotaur_states_list:
            # if minotaur_cur_state == player_cur_state:
            #     print('hi')
            #     continue
            cur_state = (player_cur_state, minotaur_cur_state)
            if player_cur_state == goal_state:
                max_reward = 0
                best_action = goal_state
            else:
                possible_minotaur_states = m_actions[minotaur_cur_state]

                max_reward = min_val
                best_action = None
                # Get the probability matrix for the next timestep given the current state of the minotaur
                next_m_prob = build_prob_matrix(minotaur_state_space, m_actions, minotaur_cur_state, 2)[1]
                next_m_prob = np.minimum(minotaur_prob[t+1] * 1e10, 1)*next_m_prob
                next_m_prob = next_m_prob/np.sum(next_m_prob)

                for possible_player_state in possible_player_states:
                    u_t = 0
                    for possible_minotaur_state in possible_minotaur_states:
                        if minotaur_prob[t+1][possible_minotaur_state] > 0:
                            # Prob of minotaur being in this state at time t+1
                            m_prob_state_t = next_m_prob[possible_minotaur_state]

                            possible_state = (possible_player_state, possible_minotaur_state)
                            u_possible = u[possible_state][0][t + 1]

                            u_t += m_prob_state_t * u_possible

                    # Probability of player and minotaur being in the same state
                    p_prob_state_t = next_m_prob[possible_player_state]
                    punish = p_prob_state_t * m_punish  # "Reward" if you get caught
                    reward = (1 - p_prob_state_t) * p_reward  # Reward if you don't get caught
                    reward = reward + punish + u_t  # Final reward of state plus future

                    if reward > max_reward:
                        max_reward = reward
                        best_action = possible_player_state

            if max_reward > u[cur_state][0][t]:
                u[cur_state][0][t] = max_reward
                u[cur_state][1][t] = best_action
                continue_path = True
        if t > 0 and continue_path:
            # print(t)
            for possible_player_state in possible_player_states:
                state_list.append((possible_player_state, t - 1))
    return u


def simulate_dp(u):

    survival_count = 0
    die_count = 0
    for n in range(n_samples):
        path = [starting_state]

        _, random_m_path = a_random_walk(m_actions, minotaur_state)

        # print(f"Sample: {n}: ")
        next_state = starting_state
        for n_t in range(1, T):
            n_best_action = u[next_state][1][n_t - 1]
            next_state = (n_best_action, random_m_path[n_t])
            # print(f"T: {n_t} is {next_state}")
            path.append(next_state)

            if next_state[0] == goal_state and next_state[1] != goal_state:
                survival_count += 1
                break
            elif next_state[0] == next_state[1]:
                die_count += 1
                break

    return survival_count / n_samples


def value_iteration(v):
    """
    V = reward
    """
    p_mean = 30

    state_list = [starting_state]


n_samples = 10000
sim_max_prob_function = []
max_exp_prob_function = []
shortest_possible_path = 16
for test_t in range(1, 50):
    # If you have less time than the shortest possible path, the probability of reaching it is always 0
    print(f"T: {test_t}")
    if test_t < shortest_possible_path:
        sim_max_prob_function.append(0)
        max_exp_prob_function.append(0)
    else:
        T = test_t

        player_state_space = build_state_space(with_obstacles=True)
        minotaur_state_space = build_state_space(with_obstacles=False)  # The minotaur can walk through obstacles

        p_actions, u_init = build_action_space(player_state_space, can_stay=True)
        m_actions, _ = build_action_space(minotaur_state_space, can_stay=True)  # The minotaur is always on the move

        m_prob = build_prob_matrix(minotaur_state_space, m_actions, minotaur_state, T)

        u_final = dp_max_prob(u_init, m_prob)
        expected_max_prob = 1 - u_final[starting_state][0][0] / m_punish
        sim_max_prob = simulate_dp(u_final)

        print("Maximum expected probability: ", expected_max_prob)
        print("Simulated maximum probability: ", sim_max_prob)

        sim_max_prob_function.append(sim_max_prob)
        max_exp_prob_function.append(expected_max_prob)

x_axis = range(len(sim_max_prob_function))
y_axis_sim = sim_max_prob_function
y_axis_exp = max_exp_prob_function
plt.plot(x_axis, y_axis_sim)
plt.show()

# print(path)

# for n_t in range(T):
#     temp_maze = maze.copy()
#     temp_maze[path[n_t][0]] = 10
#     temp_maze[path[n_t][1]] = 20
#     plt.imshow(temp_maze)
#     plt.title(f'Sample: {n}, Timestep: {n_t}')
#     plt.show()
#     if path[n_t][0] == goal_state:
#         break
