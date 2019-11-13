import numpy as np

maze = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
])


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

    return actions


def build_prob_matrix(start_state, T, states, actions):

    prob_matrix = np.zeros(shape=(T, maze.shape[0], maze.shape[1]))

    prob_matrix[0][start_state] = 1.0

    for t in range(1, T):
        for state in states:
            prob_of_this_state = prob_matrix[t-1][state]
            if prob_of_this_state != 0:

                possible_states = np.array(actions[state])
                print(f"From state {state} you can go to {possible_states}")

                prob_of_each_new_state = prob_of_this_state * (1 / possible_states.shape[0])

                prob_matrix[t][possible_states[:, 0], possible_states[:, 1]] += prob_of_each_new_state

        # print("Probality matrix at T: ", t)
        # print("Sum of probabilities: ", prob_matrix[t].sum())
        # print(prob_matrix[t])
        # exit()
    return prob_matrix


player_state_space = build_state_space(with_obstacles=True)
minotaur_state_space = build_state_space(with_obstacles=False)

player_actions = build_action_space(player_state_space, can_stay=True)
minotaur_actions = build_action_space(minotaur_state_space, can_stay=False)

T = 20
start_state = (5, 5)
print(minotaur_state_space)
print(minotaur_actions)
mat = build_prob_matrix(start_state, T, minotaur_state_space, minotaur_actions)
