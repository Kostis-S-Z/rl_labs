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
max_val = np.iinfo(np.int16).max

np.random.seed(1)

n_samples = 10000
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


def build_action_space(states, timesteps=None, can_stay=True):
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

                    if timesteps is None:
                        u[p_m_state] = [0, None]  # Initialize V with zeros
                    else:
                        t_values = np.ones(timesteps) * min_val
                        u[p_m_state] = (t_values, [None] * timesteps)

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


def a_random_walk(actions, init_state, timesteps):
    """
    Generate a random walk of the minotaur in the maze
    """
    path_matrix = np.zeros(shape=(timesteps, maze.shape[0], maze.shape[1]))

    m_path = [init_state]
    path_matrix[0][init_state] = 1.0

    state = init_state
    for t in range(1, timesteps):
        possible_states = np.array(actions[state])

        pick_state = np.random.choice(possible_states.shape[0])  # Pick a random state
        next_state = possible_states[pick_state]

        path_matrix[t][next_state[0], next_state[1]] = 1.0
        state = tuple(next_state)
        m_path.append(state)

    return path_matrix, m_path


def a_random_walk_hybrid(actions, init_state, timesteps):
    """
    Generate a random walk of the minotaur in the maze
    """
    path_matrix = np.zeros(shape=(timesteps, maze.shape[0], maze.shape[1]))

    path_matrix[0][init_state] = 1.0

    state = init_state
    for t in range(1, timesteps):
        possible_states = np.array(actions[state])

        n_possible_states = possible_states.shape[0]

        path_matrix[t][possible_states[:, 0], possible_states[:, 1]] = 1.0 / n_possible_states

        pick_state = np.random.choice(possible_states.shape[0])  # Pick a random state
        next_state = possible_states[pick_state]
        state = tuple(next_state)

    return path_matrix


def dp_max_prob(mss, p_a, m_a, minotaur_prob, u, timesteps):
    minotaur_state_space = mss
    p_actions = p_a
    m_actions = m_a

    # Key is a tuple of: 1) tuple of player state & minotaur state and 2) timestep
    state_list = [(goal_state, timesteps-1)]

    p_reward = 0  # Reward for not dying and for not reaching the goal, basically reward of moving
    m_punish = -10000  # "Reward" for dying
    goal_reward = 0  # Reward for reaching goal

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

            cur_state = (player_cur_state, minotaur_cur_state)
            if player_cur_state == goal_state:
                max_reward = goal_reward
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

    expected_max_prob = 1 - u[starting_state][0][0] / m_punish
    return u, expected_max_prob


def better_dp_max_prob(pss, mss, p_a, m_a, u, timesteps):
    player_state_space = pss
    minotaur_state_space = mss
    p_actions = p_a
    m_actions = m_a

    p_reward = 0  # Reward for not dying and for not reaching the goal, basically reward of moving
    m_punish = 0  # "Reward" for dying
    goal_reward = 1  # Reward for reaching goal

    for t in range(timesteps-1, -1, -1):
        for player_cur_state in player_state_space:
            for minotaur_cur_state in minotaur_state_space:

                cur_state = (player_cur_state, minotaur_cur_state)
                # If you die or reach terminal state, there is no need to calculate future rewards
                if player_cur_state == minotaur_cur_state or (t == timesteps-1 and player_cur_state != goal_state):
                    max_reward = m_punish
                    best_action = player_cur_state
                elif player_cur_state == goal_state:
                    max_reward = goal_reward
                    best_action = goal_state
                else:
                    possible_minotaur_states = m_actions[minotaur_cur_state]
                    possible_player_states = p_actions[player_cur_state]

                    max_reward = min_val
                    best_action = None
                    # Get the probability matrix for the next timestep given the current state of the minotaur
                    next_m_prob = build_prob_matrix(minotaur_state_space, m_actions, minotaur_cur_state, 2)[1]

                    for possible_player_state in possible_player_states:
                        u_t = 0
                        for possible_minotaur_state in possible_minotaur_states:
                            # Prob of minotaur being in this state at time t+1
                            m_prob_state_t = next_m_prob[possible_minotaur_state]

                            possible_state = (possible_player_state, possible_minotaur_state)
                            u_possible = u[possible_state][0][t + 1]
                            u_t += m_prob_state_t * u_possible

                        reward = p_reward + u_t  # Final reward of state plus future

                        if reward > max_reward:
                            max_reward = reward
                            best_action = possible_player_state

                # Update u values
                u[cur_state][0][t] = max_reward
                u[cur_state][1][t] = best_action

    expected_max_prob = u[starting_state][0][0]
    return u, expected_max_prob


def simulate_dp_runs(m_actions, u, time):

    survival_count = 0
    die_count = 0
    for n in range(n_samples):
        path = [starting_state]

        _, random_m_path = a_random_walk(m_actions, minotaur_state, time)

        # print(f"Sample: {n}: ")
        next_state = starting_state
        for n_t in range(1, time):
            n_best_action = u[next_state][1][n_t - 1]
            next_state = (n_best_action, random_m_path[n_t])
            # print(f"T: {n_t} is {next_state}")
            path.append(next_state)

            if next_state[0] == next_state[1]:
                die_count += 1
                break
            elif next_state[0] == goal_state:
                survival_count += 1
                break

    return survival_count / n_samples


def run_dp():

    use_better = False
    test_time = 40

    pss = build_state_space(with_obstacles=True)
    mss = build_state_space(with_obstacles=False)  # The minotaur can walk through obstacles

    sim_max_prob_function = []
    max_exp_prob_function = []

    shortest_possible_path = 16

    for test_t in range(1, test_time):
        # If you have less time than the shortest possible path, the probability of reaching it is always 0
        print(f"T: {test_t}")

        if test_t < shortest_possible_path:
            sim_max_prob_function.append(0)
            max_exp_prob_function.append(0)
        else:
            p_a, u_init = build_action_space(pss, timesteps=test_t, can_stay=True)
            m_a, _ = build_action_space(mss, timesteps=test_t, can_stay=True)

            if use_better:
                u_final, expected_max_prob = better_dp_max_prob(pss, mss, p_a, m_a, u_init, test_t)
            else:
                # Build minotaur probability matrix
                m_prob = build_prob_matrix(mss, m_a, minotaur_state, test_t)
                u_final, expected_max_prob = dp_max_prob(mss, p_a, m_a, m_prob, u_init, test_t)

            sim_max_prob = simulate_dp_runs(m_a, u_final, test_t)

            print("Maximum expected probability: ", expected_max_prob)
            print("Simulated maximum probability: ", sim_max_prob)

            sim_max_prob_function.append(sim_max_prob)
            max_exp_prob_function.append(expected_max_prob)

    x_axis = range(len(sim_max_prob_function))
    y_axis_sim = sim_max_prob_function
    y_axis_exp = max_exp_prob_function
    plt.plot(x_axis, y_axis_sim, label="Simulated")
    plt.plot(x_axis, y_axis_exp, label="Analytical")
    plt.show()


def value_iteration(pss, mss, p_a, m_a, v, lifespan):
    player_state_space = pss
    minotaur_state_space = mss
    p_actions = p_a
    m_actions = m_a

    gamma = 1 - (1 / lifespan)

    delta = max_val
    convergence_condition = 10e-3

    while delta > convergence_condition:

        delta = 0
        for player_cur_state in player_state_space:
            for minotaur_cur_state in minotaur_state_space:

                cur_state = (player_cur_state, minotaur_cur_state)

                # If you die or reach terminal state, there is no need to calculate future rewards
                if player_cur_state == minotaur_cur_state:
                    max_reward = 0
                    best_action = player_cur_state
                elif player_cur_state == goal_state:
                    max_reward = 1
                    best_action = goal_state
                else:
                    possible_minotaur_states = m_actions[minotaur_cur_state]
                    possible_player_states = p_actions[player_cur_state]

                    max_reward = min_val
                    best_action = None

                    # Get the probability matrix for the next timestep given the current state of the minotaur
                    next_m_prob = build_prob_matrix(minotaur_state_space, m_actions, minotaur_cur_state, 2)[1]

                    # Calculate future rewards
                    for possible_player_state in possible_player_states:
                        future_sum = 0
                        for possible_minotaur_state in possible_minotaur_states:
                            # Prob of minotaur being in this state at time t+1
                            m_prob_state_t = next_m_prob[possible_minotaur_state]

                            possible_state = (possible_player_state, possible_minotaur_state)
                            v_possible = v[possible_state][0]
                            future_sum += m_prob_state_t * v_possible

                        r_s_a = 0  # reward of not winning and not dying
                        reward = r_s_a + (gamma * future_sum)

                        if reward > max_reward:
                            max_reward = reward
                            best_action = possible_player_state

                prev_v = v[cur_state][0]
                v[cur_state][0] = max_reward
                v[cur_state][1] = best_action

                delta += np.square(prev_v - max_reward)

        delta = np.sqrt(delta)

    return v


def run_vi():

    pss = build_state_space(with_obstacles=True)
    mss = build_state_space(with_obstacles=False)  # The minotaur can walk through obstacles

    p_a, v_init = build_action_space(pss, can_stay=True)  # Do not build V with a time dimension
    m_a, _ = build_action_space(mss, can_stay=True)

    lifespan = 30  # if lifespan is small, gamma should be small which means future doesn't matter that much

    v = value_iteration(pss, mss, p_a, m_a, v_init, lifespan)

    survival_count = 0
    minotaur_deaths = 0
    time_deaths = 0

    results = {}

    for n in range(n_samples):
        print(f"Sample: {n} / {n_samples}")

        path = [starting_state]

        time = np.random.geometric(1 / lifespan) + 1

        _, random_m_path = a_random_walk(m_a, minotaur_state, time)

        next_state = starting_state

        for n_t in range(1, time):

            n_best_action = v[next_state][1]
            next_state = (n_best_action, random_m_path[n_t])

            path.append(next_state)

            if next_state[0] == next_state[1]:
                minotaur_deaths += 1
                outcome = 0
                break
            elif next_state[0] == goal_state:
                survival_count += 1
                outcome = 1
                break
        else:
            time_deaths += 1
            outcome = 2

        if time in results:
            results[time].append(outcome)
        else:
            results[time] = [outcome]

    print(f"Win ratio: {survival_count / n_samples}")
    print(f"Survival count: {survival_count}")
    print(f"Minotaur deaths: {minotaur_deaths}")
    print(f"Time-out deaths: {time_deaths}")

    res_per_t = {}

    for key, value in results.items():
        # Count each type of outcome
        mino_deaths_of_t = value.count(0)
        survival_of_t = value.count(1)
        time_deaths_of_t = value.count(2)

        # All samples for a specific T
        n_samples_of_t = mino_deaths_of_t + survival_of_t + time_deaths_of_t

        # Percentage of each outcome for specific T
        res_per_t[key] = {"m": mino_deaths_of_t / n_samples_of_t,
                          "s": survival_of_t / n_samples_of_t,
                          "t": time_deaths_of_t / n_samples_of_t}

        if 10 < key < 50:
            plt.scatter(key, res_per_t[key]["m"], c='red')
            plt.scatter(key, res_per_t[key]["s"], c='blue')
            plt.scatter(key, res_per_t[key]["t"], c='green')
            # plt.axvline(x=key, ymax=res_per_t[key]["m"], c='red')
            # plt.axvline(x=key, ymax=res_per_t[key]["s"], c='blue')
            # plt.axvline(x=key, ymax=res_per_t[key]["t"], c='green')

    plt.show()


def plot_m_evolution():
    time = 20
    can_stay = True
    title = "can_stay" if can_stay else "no_stay"

    mss = build_state_space(with_obstacles=False)  # The minotaur can walk through obstacles
    m_a, _ = build_action_space(mss, timesteps=time, can_stay=can_stay)
    m_prob = build_prob_matrix(mss, m_a, minotaur_state, time)

    for i in range(time):
        plt.imshow(m_prob[i])
        plt.axis('off')
        plt.colorbar()
        plt.savefig(f"figures/{title}_{i}.png")
        plt.show()


# run_dp()
# run_vi()
