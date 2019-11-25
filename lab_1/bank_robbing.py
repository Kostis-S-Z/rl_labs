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
iterations = 10000000
epsilon = 0.1
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


def build_action_space(states):
    """
    Define action space
    """
    actions = {}
    for i in range(reward_map.shape[0]):
        for j in range(reward_map.shape[1]):
            key = (i, j)
            value = [((i, j), "stay")]

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
    action = np.random.choice(len(action_space[state]))
    return action_space[state][action]


def q_learning(states, action_space):

    robber_state = init_robber
    police_state = init_police
    s_t = (robber_state, police_state)

    q = init_q(states, action_space)
    lr = init_q(states, action_space)

    for i in range(iterations):

        # Choose random action out of legal actions
        next_robber_state, robber_action = choose_random_action(action_space, robber_state)
        next_police_state, police_action = choose_random_action(action_space, police_state)

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

        if i % 1000 == 0:
            print(f"Iteration: {i} Q{s_t}: {q[s_t]}")


def sarsa(states):
    action_count = np.zeros(shape=n_actions)
    q = init_q(states)

    action = e_greedy(q)
    action_count[action] += 1


def e_greedy(q):
    if np.random.rand() > epsilon:  # With probability random > 0.1, choose optimal action
        return np.argmax(q)
    else:
        return np.random.randint(n_actions)  # Otherwise, choose randomly


s_space = build_state_space()
a_space = build_action_space(s_space)

# We assume we choose only from LEGAL actions
q_learning(s_space, a_space)
