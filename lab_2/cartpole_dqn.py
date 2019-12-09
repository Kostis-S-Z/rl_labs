import sys
import gym
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

from lab_2.utils import *

EPISODES = 1000  # Maximum number of episodes
test_state_no = 10000

# Define default parameters of agent
def_params = {
    'discount_factor': 0.95,
    'learning_rate': 0.005,
    'epsilon': 0.02,
    'batch_size': 32,
    'memory_size': 1000,
    'train_start': 1000,
    'target_update_frequency': 1,
}

# Define a simple Neural Network Architecture
net = {
    'input_layer': 256,
    'layer_1': 256,
    'layer_2': 256
}


class DQNAgent:
    """
    DQN Agent for the Cartpole

    Q function approximation with NN, experience replay, and target network
    """

    def __init__(self, net_arch):

        self.check_solve = False  # If True, stop if you satisfy solution condition
        self.render = False  # If you want to see Cartpole learning, then change to True

        # Get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # Modify here

        # Set hyper parameters for the DQN. Do not adjust those labeled as Fixed.
        self.discount_factor = def_params['discount_factor']
        self.learning_rate = def_params['learning_rate']
        self.epsilon = def_params['epsilon']
        self.batch_size = def_params['batch_size']
        self.memory_size = def_params['memory_size']
        self.train_start = def_params['train_start']
        self.target_update_frequency = def_params['target_update_frequency']

        self.net = net_arch

        # Number of test states for Q value plots
        self.test_state_no = test_state_no

        # Create memory buffer using deque
        self.memory = deque(maxlen=self.memory_size)

        # Create main network and target network (using build_model defined below)
        self.model = self.build_model()
        self.target_model = self.build_model()

        # Initialize target network
        self.update_target_model()

    def build_model(self):
        """
        Approximate Q function using Neural Network
        Input: State
        Output: Q-values
        based on: https://keras.io/getting-started/sequential-model-guide/
        """
        model = Sequential()

        model.add(Dense(self.net['input_layer'], input_dim=self.state_size,
                        activation='relu', kernel_initializer='he_uniform'))

        # Add layers
        for i, units in self.net.items():
            model.add(Dense(units, activation='relu', kernel_initializer='he_uniform'))

        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))

        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def update_target_model(self):
        """
        After some time interval update the target model to be same with model
        """
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        """
        Get action from model using epsilon-greedy policy

        With probability random > epsilon, choose optimal action
        Otherwise, choose randomly from the action space
        """

        if np.random.rand() > self.epsilon:
            action = np.argmax(self.model.predict(state))
        else:
            action = random.randrange(self.action_size)

        return action

    def append_sample(self, state, action, reward, next_state, done):
        """
        Experience Replay: Save sample <s,a,r,s'> to the replay memory
        """
        self.memory.append((state, action, reward, next_state, done))  # Add sample to the end of the list

    def train_model(self):
        """
        Sample <s,a,r,s'> from replay memory
        """

        if len(self.memory) < self.train_start:  # Do not train if not enough memory
            return

        batch_size = min(self.batch_size, len(self.memory))  # Train on at most as many samples as you have in memory
        mini_batch = random.sample(self.memory, batch_size)  # Uniformly sample the memory buffer

        # Preallocate network and target network input matrices.
        update_input = np.zeros((batch_size, self.state_size))  # batch_size by state_size two-dimensional array
        update_target = np.zeros((batch_size, self.state_size))  # Same as above, but used for the target network

        action, reward, done = [], [], []  # Empty arrays that will grow dynamically

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]  # Allocate s(i) to the network input array from iteration i of batch
            action.append(mini_batch[i][1])  # Store a(i)
            reward.append(mini_batch[i][2])  # Store r(i)
            update_target[i] = mini_batch[i][3]  # Allocate s'(i) for the target network array from iteration i of batch
            done.append(mini_batch[i][4])  # Store done(i)

        # Generate target values for training the inner loop network using the network model
        target = self.model.predict(update_input)

        # Generate the target values for training the outer loop target network
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            target[i][action[i]] = reward[i]
            if not done[i]:
                target[i][action[i]] += self.discount_factor * np.max(target_val[i])

        # Train the inner loop network
        self.model.fit(update_input, target, batch_size=self.batch_size, epochs=1, verbose=0)

        return


def train(net_arch, model_name=""):
    """
    Train DQN Agent
    """
    agent = DQNAgent(net_arch)

    max_q = np.zeros((EPISODES, agent.test_state_no))
    max_q_mean = np.zeros((EPISODES, 1))

    scores, episodes = [], []  # Create dynamically growing score and episode counters
    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()  # Initialize/reset the environment
        state = np.reshape(state, [1, state_size])  # Reshape state ie. [x_1,x_2] to [[x_1,x_2]]

        # Compute Q values for plotting
        # Helps us monitor the model performance by evaluating in random states and see what actions it would take
        tmp = agent.model.predict(test_states)
        max_q[e][:] = np.max(tmp, axis=1)
        max_q_mean[e] = np.mean(max_q[e][:])

        while not done:
            if agent.render:
                env.render()  # Show cartpole animation

            # Get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])  # Reshape next_state similarly to state

            # Save sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, done)

            # Training step
            agent.train_model()
            score += reward  # Store episodic reward
            state = next_state  # Propagate state

            if done:
                # At the end of very episode, update the target network
                if e % agent.target_update_frequency == 0:
                    agent.update_target_model()
                # Plot the play time for every episode
                scores.append(score)
                episodes.append(e)

                print("episode:", e, " score:", score, " q_value:", max_q_mean[e], " memory length:", len(agent.memory))

                # if the mean of scores of last 100 episodes is bigger than 195
                # stop training
                if agent.check_solve:
                    if np.mean(scores[-min(100, len(scores)):]) >= 195:
                        print("solved after", e - 100, "episodes")
                        plot_data(episodes, scores, max_q_mean[:e + 1], model_name)
                        sys.exit()
    plot_data(episodes, scores, max_q_mean, model_name)


def sample_test_states():
    """
    Collect test states for plotting Q values using uniform random policy
    """
    samples_test_states = np.zeros((test_state_no, state_size))

    done = True
    state = env.reset()
    for i in range(test_state_no):
        if done:
            done = False
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            samples_test_states[i] = state
        else:
            action = random.randrange(action_size)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            samples_test_states[i] = state
            state = next_state

    return samples_test_states


def net_exp():
    net_1 = {'input_layer': 64, 'layer_1': 128, 'layer_2': 256}
    net_2 = {'input_layer': 64, 'layer_1': 128, 'layer_2': 256}
    net_3 = {'input_layer': 64, 'layer_1': 128, 'layer_2': 256}
    net_4 = {'input_layer': 64, 'layer_1': 128, 'layer_2': 256}

    nets = [net_1, net_2, net_3, net_4]

    for net_i in nets:
        train(net_i)


def hyper_exp():
    d_factors = [0.5, 0.8, 0.9]
    lrs = [0.01, 0.05, 0.001]
    mems = [1000, 5000]

    for d_factor in d_factors:
        def_params['discount_factor'] = d_factor
        for lr in lrs:
            def_params['learning_rate'] = lr
            for mem in mems:
                def_params['memory_size'] = mem

                train(net)


def target_update_exp():
    tar_updates = [1, 10, 100]

    for tar_update in tar_updates:
        def_params['target_update_frequency'] = tar_update

        train(net)


if __name__ == "__main__":
    # For CartPole-v0, maximum episode length is 200
    env = gym.make('CartPole-v0')  # Generate Cartpole-v0 environment object from the gym library

    # Get state and action sizes from the environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Collect test states
    test_states = sample_test_states()

    train(net)

    # net_exp()
    # hyper_exp()
    # target_update_exp()

