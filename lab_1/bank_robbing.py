import numpy as np
import matplotlib.pyplot as plt

"""
Problem 3: Bank Robbing

Heist a bank while escaping from the police!
"""

min_val = np.iinfo(np.int16).min
max_val = np.iinfo(np.int16).max
n_actions = len(["stay", "up", "down", "right", "left"])

robber = (0, 0)
police = (3, 3)
bank = (1, 1)

reward_map = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])
reward_map[bank] = 1

# Parameter values
iterations = 10000
epsilon = 0.1
discount_factor = 0.8


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

    return actions


def reward_function(r_state, p_state):
    if r_state == p_state:  # If you are caught by the police
        return -10
    elif r_state == bank:  # If you reach the bank
        return 1
    else:  # Otherwise
        return 0


def q_learning():
    action_count = np.zeros(shape=n_actions)
    q_values = np.zeros(shape=n_actions)

    for i in range(iterations):

        action = e_greedy(q_values)
        action_count[action] += 1


def sarsa():
    pass


def e_greedy(q):
    if np.random.rand() > epsilon:  # With probability random > 0.1, choose optimal action
        return np.argmax(q)
    else:
        return np.random.randint(n_actions)  # Otherwise, choose randomly


s_space = build_state_space()
a_space = build_action_space(s_space)

