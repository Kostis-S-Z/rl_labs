import numpy as np
import matplotlib.pyplot as plt

"""
Problem 2: Robbing Banks (Bonus)

Rob banks and escape the police!
"""

min_val = np.iinfo(np.int16).min
max_val = np.iinfo(np.int16).max

bank_1 = (0, 0)
bank_2 = (0, 5)
bank_3 = (2, 0)
bank_4 = (2, 5)
banks = [bank_1, bank_2, bank_3, bank_4]
city = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
])
for bank in banks:
    city[bank] = 1

init_robber = bank_1
init_police = (1, 2)

n_samples = 10000


def build_state_space():
    """
    Define state space
    """
    states = []
    for i in range(city.shape[0]):
        for j in range(city.shape[1]):
            states.append((i, j))

    return states


def build_action_space(states, can_stay=True):
    """
    Define action space
    """
    actions = {}
    u = {}
    for i in range(city.shape[0]):
        for j in range(city.shape[1]):
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

            for m_i in range(city.shape[0]):
                for m_j in range(city.shape[1]):
                    p_m_state = ((i, j), (m_i, m_j))  # Player - Minotaur state

                    u[p_m_state] = [0, None]  # Initialize V with zeros

    return actions, u


def build_prob_matrix(states, actions, init_state, timesteps):
    """
    Create a matrix with the probabilities of the minotaur at each state when he begins from a initial state
    """
    prob_matrix = np.zeros(shape=(timesteps, city.shape[0], city.shape[1]))

    prob_matrix[0][init_state] = 1.0

    for t in range(1, timesteps):
        for state in states:
            prob_of_this_state = prob_matrix[t - 1][state]
            if prob_of_this_state != 0:
                possible_states = np.array(actions[state])

                prob_of_each_new_state = prob_of_this_state * (1 / possible_states.shape[0])

                prob_matrix[t][possible_states[:, 0], possible_states[:, 1]] += prob_of_each_new_state

    return prob_matrix


def reward_function(r_state, p_state):
    if r_state == p_state:  # If you are caught by the police
        return -50
    elif r_state in banks:  # If you reach a bank
        return 10
    else:  # Otherwise
        return 0

