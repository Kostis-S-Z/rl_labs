import numpy as np
import matplotlib.pyplot as plt

"""
Problem 3: Bank Robbing

Heist a bank while escaping from the police!
"""

min_val = np.iinfo(np.int16).min
max_val = np.iinfo(np.int16).max
n_actions = 5  # stay, up, down, left, right

init_robber = (0, 0)
init_police = (3, 3)
bank = (1, 1)

reward_map = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])
reward_map[bank] = 1

# Parameter values
iterations = 5000000  # 10.000.000 = 10000000
d_factor = 0.8


def build_state_space():
    """
    Define state space
    """
    states = []
    for i in range(reward_map.shape[0]):
        for j in range(reward_map.shape[1]):
            states.append((i, j))

    return states


def build_action_space(states, can_stay=True):
    """
    Define action space
    """
    actions = {}
    for i in range(reward_map.shape[0]):
        for j in range(reward_map.shape[1]):
            key = (i, j)

            if can_stay:
                value = [((i, j), "stay")]
            else:
                value = []

            if (i, j - 1) in states:
                value.append(((i, j - 1), "left"))
            if (i, j + 1) in states:
                value.append(((i, j + 1), "right"))
            if (i + 1, j) in states:
                value.append(((i + 1, j), "down"))
            if (i - 1, j) in states:
                value.append(((i - 1, j), "up"))
            actions[key] = value

    return actions


def init_q(states, actions):
    q = {}
    for robber_state in states:
        for police_state in states:
            possible_actions = actions[robber_state]
            q[robber_state, police_state] = {action[1]: 0 for action in possible_actions}
    return q


def reward_function(r_state, p_state):
    if r_state == p_state:  # If you are caught by the police
        return -10
    elif r_state == bank:  # If you reach the bank
        return 1
    else:  # Otherwise
        return 0


def choose_random_action(action_space, state):
    i_action = np.random.randint(len(action_space[state]))
    return action_space[state][i_action]


def q_learning(states, action_space_robber, action_space_police):

    robber_state = init_robber
    police_state = init_police
    init_state = (robber_state, police_state)

    q = init_q(states, action_space_robber)
    lr = init_q(states, action_space_robber)

    # q_init_evolution = {"stay": [0], "up": [0], "down": [0], "left": [0], "right": [0]}
    q_init_evolution = {"stay": [0], "down": [0], "right": [0]}

    s_t = init_state
    for i in range(iterations):

        # Choose random action out of legal actions
        next_robber_state, robber_action = choose_random_action(action_space_robber, robber_state)
        next_police_state, police_action = choose_random_action(action_space_police, police_state)

        s_t_1 = (next_robber_state, next_police_state)

        # Calculate reward
        r = reward_function(next_robber_state, next_police_state)

        # Find appropriate learning rate (alpha) based on current state and action
        lr[s_t][robber_action] += 1
        alpha = 1 / (lr[s_t][robber_action] ** (2/3))

        # Update Q values
        q[s_t][robber_action] += alpha * (r + d_factor * np.max(list(q[s_t_1].values())) - q[s_t][robber_action])

        # Move to next state
        robber_state = next_robber_state
        police_state = next_police_state
        s_t = (robber_state, police_state)

        q_init_evolution["stay"].append(q[init_state]["stay"])
        q_init_evolution["down"].append(q[init_state]["down"])
        q_init_evolution["right"].append(q[init_state]["right"])
        if i % 1000 == 0:
            print(f"Iteration: {i} Q{s_t}: {q[s_t]}")

    plot_q(q_init_evolution)


def plot_q(q_evolution, title="q_evolution"):
    x_axis = range(len(q_evolution["stay"]))
    y_axis_stay = q_evolution["stay"]
    y_axis_down = q_evolution["down"]
    y_axis_right = q_evolution["right"]
    plt.plot(x_axis, y_axis_stay, label="stay")
    plt.plot(x_axis, y_axis_down, label="down")
    plt.plot(x_axis, y_axis_right, label="right")
    plt.legend()
    plt.show()
    plt.savefig(title + ".png")


def e_greedy(epsilon, robber_state, possible_action_values, action_space):
    if np.random.rand() > epsilon:  # With probability random > 0.1, choose optimal action
        i_action = np.argmax(possible_action_values)
    else:
        i_action = np.random.randint(len(possible_action_values))  # Otherwise, choose randomly

    return action_space[robber_state][i_action]


def sarsa(states, action_space_robber, action_space_police, epsilon=0.1):

    robber_state = init_robber
    police_state = init_police
    init_state = (robber_state, police_state)

    q = init_q(states, action_space_robber)
    lr = init_q(states, action_space_robber)

    # q_init_evolution = {"stay": [0], "up": [0], "down": [0], "left": [0], "right": [0]}
    q_init_evolution = {"stay": [0], "down": [0], "right": [0]}

    s_t = init_state
    for i in range(iterations):

        # Choose random action for police
        next_police_state, police_action = choose_random_action(action_space_police, police_state)
        # Choose action for robber based on epsilon greedy
        possible_action_values = list(q[s_t].values())
        next_robber_state, robber_action = e_greedy(epsilon, robber_state, possible_action_values, action_space_robber)

        s_t_1 = (next_robber_state, next_police_state)

        # Calculate reward
        r = reward_function(next_robber_state, next_police_state)

        # Find appropriate learning rate (alpha) based on current state and action
        lr[s_t][robber_action] += 1
        alpha = 1 / (lr[s_t][robber_action] ** (2/3))

        # Select new action
        possible_next_action_values = list(q[s_t_1].values())
        _, next_robber_action = e_greedy(epsilon, next_robber_state, possible_next_action_values, action_space_robber)

        # Update Q values
        q[s_t][robber_action] += alpha * (r + d_factor * q[s_t_1][next_robber_action] - q[s_t][robber_action])

        # Move to next state
        robber_state = next_robber_state
        police_state = next_police_state
        s_t = (robber_state, police_state)

        q_init_evolution["stay"].append(q[init_state]["stay"])
        q_init_evolution["down"].append(q[init_state]["down"])
        q_init_evolution["right"].append(q[init_state]["right"])
        if i % 1000 == 0:
            print(f"Iteration: {i} Q{s_t}: {q[s_t]}")

    plot_q(q_init_evolution, title="q_evolution_" + str(epsilon))


s_space = build_state_space()
a_space = build_action_space(s_space)
p_a_space = build_action_space(s_space, can_stay=False)


# We assume we choose only from LEGAL actions
# q_learning(s_space, a_space, p_a_space)
sarsa(s_space, a_space, p_a_space)

e_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
for e in e_values:
    sarsa(s_space, a_space, p_a_space, epsilon=e)
