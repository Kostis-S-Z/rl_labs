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
    v = {}
    for i in range(city.shape[0]):
        for j in range(city.shape[1]):
            key = (i, j)
            if can_stay:
                value = [(i, j)]
            else:
                value = []
            if (i, j - 1) in states:  # left
                value.append((i, j - 1))
            if (i, j + 1) in states:  # right
                value.append((i, j + 1))
            if (i + 1, j) in states:  # down
                value.append((i + 1, j))
            if (i - 1, j) in states:  # up
                value.append((i - 1, j))
            actions[key] = value

            for p_i in range(city.shape[0]):
                for p_j in range(city.shape[1]):
                    r_p_state = ((i, j), (p_i, p_j))  # Robber - Police state

                    v[r_p_state] = [0, None]  # Initialize V with zeros

    return actions, v


def next_police_move(police_state, robber_state, p_actions):
    """
    Create a matrix with the probabilities of the minotaur at each state when he begins from a initial state
    """
    prob_matrix = np.zeros(shape=(city.shape[0], city.shape[1]))

    # 0: left, 1: right, 2: down, 3: up
    possible_states = p_actions[police_state]
    possible_states_indices = np.array(possible_states)

    prob_matrix[possible_states_indices[:, 0], possible_states_indices[:, 1]] = 1

    if robber_state[1] > police_state[1]:  # Robber is on the right
        left_state = (police_state[0], police_state[1] - 1)
        if left_state in possible_states:
            prob_matrix[left_state] = 0
    elif robber_state[1] < police_state[1]:  # Robber is on the left
        right_state = (police_state[0], police_state[1] + 1)
        if right_state in possible_states:
            prob_matrix[right_state] = 0

    if robber_state[0] > police_state[0]:  # Robber is below
        above_state = (police_state[0] - 1, police_state[1])
        if above_state in possible_states:
            prob_matrix[above_state] = 0
    elif robber_state[0] < police_state[0]:  # Robber is above
        below_state = (police_state[0] + 1, police_state[1])
        if below_state in possible_states:
            prob_matrix[below_state] = 0

    prob_matrix = prob_matrix / np.sum(prob_matrix)

    return prob_matrix


def reward_function(r_state, p_state):
    if r_state == p_state:  # If you are caught by the police
        return -50
    elif r_state in banks:  # If you reach a bank
        return 10
    else:  # Otherwise
        return 0


def value_iteration(r_states, p_states, r_actions, p_actions, v):

    iterations = 0
    delta = max_val
    convergence_condition = 10e-6

    v_init_evolution = []

    while delta > convergence_condition:

        delta = 0
        for robber_state in r_states:
            for police_state in p_states:

                cur_state = (robber_state, police_state)

                if robber_state == police_state:
                    possible_robber_states = [init_robber]
                    possible_police_states = [init_police]
                else:
                    possible_robber_states = r_actions[robber_state]
                    possible_police_states = p_actions[police_state]

                max_reward = min_val
                next_state = None

                # Calculate reward for this state
                cur_reward = reward_function(robber_state, police_state)

                # Probability matrix of the next moves of the police
                next_police_prob = next_police_move(police_state, robber_state, p_actions)

                # Calculate future rewards
                for possible_robber_state in possible_robber_states:
                    future_sum = 0
                    for possible_police_state in possible_police_states:
                        police_next_prob = next_police_prob[possible_police_state]

                        possible_state = (possible_robber_state, possible_police_state)
                        v_possible = v[possible_state][0]

                        future_sum += police_next_prob * v_possible

                    reward = cur_reward + (gamma * future_sum)

                    if reward > max_reward:
                        max_reward = reward
                        next_state = possible_robber_state

                prev_v = v[cur_state][0]
                v[cur_state][0] = max_reward
                v[cur_state][1] = next_state

                delta += np.square(prev_v - max_reward)

        v_init_evolution.append(v[(init_robber, init_police)][0])
        iterations += 1
        delta = np.sqrt(delta)

        print(f"I:{iterations} | Delta: {delta}")

    return v, v_init_evolution, iterations


state_space = build_state_space()
robber_actions, v_val = build_action_space(state_space)
police_actions, _ = build_action_space(state_space, can_stay=False)

gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for gamma in gammas:
    v, _, i_s = value_iteration(state_space, state_space, robber_actions, police_actions, v_val)
    plt.scatter(gamma, v[(init_robber, init_police)][0])
    print(f"VI converged with lambda = {gamma} after {i_s} iterations.")

plt.show()
plt.savefig("v_gamma.png")


