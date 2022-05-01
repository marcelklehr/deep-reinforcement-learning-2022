import os
import platform
from turtle import end_fill
import numpy as np

import random


class Environment:
    def __init__(self):
        self.block = 'X'
        self.up = 'u'
        self.right = 'r'
        self.down = 'd'
        self.left = 'l'
        self.agent_position = np.array([0, 0])
        self.wind_strength = 0.2
        self.reset()

    def reset(self):
        self.agent_position = np.array([0, 0])

        self.reward_matrix = np.array([
            [0, 0, 0, 0, 20],
            [0, 0, 0, 0, -2],
            [0, 0, 0, 0, -4],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]])

        # self.map_matrix = np.array([
        #     ['S', 'X', '0', '0', '0'],
        #     ['S', 'E', '0', 'E', 'S'],
        #     ['S', 'N', 'X', '0', 'W'],
        #     ['W', 'X', '0', '0', 'X'],
        #     ['0', '0', '0', '0', 'X']])
        # self.map_matrix = np.array([
        #     ['S', 'X', '0', '0', '0'],
        #     ['S', 'E', '0', 'E', 'S'],
        #     ['S', 'N', 'X', '0', 'W'],
        #     ['W', 'X', '0', '0', 'X'],
        #     ['0', '0', '0', '0', 'X']])

        self.map_matrix = np.array([
            ['0', '0', 'X', '0', '0'],
            ['0', '0', 'X', '0', '0'],
            ['0', '0', '0', '0', '0'],
            ['0', '0', '0', '0', '0'],
            ['0', '0', '0', '0', '0']])

    def step(self, action):
        """
            Applies the state transition dynamics and reward dynamics based on the state of the environment
            and the action argument. Returns (1) The new state, (2) the reward of this step,
            (3) a boolean indicating whether this state is terminal.
                Args:
                    action(np.array):
                Returns:
                    new_state (np.array): new agent position
                    reward (float):
                    terminal (bool): indicates whether the state is terminal
        """
        action_vector = ACTIONS_TO_VECTORS[action]
        map_tile = self.map_matrix[self.agent_position[0],
                                   self.agent_position[1]]
        reward = 0

        # check if wind is present
        if map_tile in ['N', 'S', 'W', 'E'] and self.wind_strength > random.uniform(0, 1):
            next_step = WIND_TO_VECTORS[map_tile]
        else:
            next_step = action_vector
        next_pos = next_step + self.agent_position

        # check for validity of the step (if in gridworld and no wall)
        if self.valid_step(next_pos):
            # GO!🚨
            reward = self.reward_matrix[next_pos[0], next_pos[1]]
            self.agent_position = next_pos
        # print('ACTION:')
        # print(next_step)
        return self.agent_position, reward, reward == 20

    def valid_step(self, next_pos):
        """
        Check if next step the agents want to take is valid. Valid being in the gridworld field and not running into a wall.
            Args:
                next_pos (np.array): future position of the agent
            returns:
                bool if action is valid
        """
        if next_pos[0] < 0 or next_pos[1] < 0 or next_pos[0] >= self.map_matrix.shape[0] or next_pos[1] >= self.map_matrix.shape[1]:
            return False
        if self.map_matrix[next_pos[0], next_pos[1]] == 'X':
            return False
        return True

    def visualize(self):
        """
        Visualizes the environment including the agent
        (prints characters to stdout)
        """
        # clear previous visualisation
        # if platform.system() == ('Linux' or 'Darwin'):
        # os.system('clear')
        # elif platform.system() == 'Windows':
        # os.system('cls')

        for i in range(0, 5):
            for j in range(0, 5):
                if self.map_matrix[i, j] == self.block:
                    print('🧱',  end="")
                elif i == self.agent_position[0] and j == self.agent_position[1]:
                    print('🦘',  end="")
                elif self.reward_matrix[i, j] < 0:
                    print('🏮',  end="")
                elif self.reward_matrix[i, j] > 0:
                    print('🥇',  end="")
                else:
                    print('⬜',  end="")
            print('\n',  end="")


class Agent:
    def __init__(self, epsilon, alpha, gamma):
        self.state = np.array([0, 0])
        #self.q_table = np.ones((5, 5, 4))
        self.q_table = np.random.random_sample((5, 5, 4))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def reset(self):
        self.state = np.array([0, 0])

    def visualize_q_table(self):
        for i in range(0, 5):
            for j in range(0, 5):
                action = ACTIONS[np.argmax(self.q_table[i, j, :])]
                print(ACTIONS_TO_EMOJI[action], end='')
            print("\n", end='')

    def choose_action(self, state):
        """
        Agent randomly chooses an action or chooses the best action, based on epsilon
        Args:
            state (np.array): current state
        Returns:
            action (char): action
        """
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(ACTIONS)
            # print('EXPLORE')
        else:
            # print('EXPLOIT')
            action = ACTIONS[np.argmax(self.q_table[state[0], state[1], :])]
            # thomson sampling
            # probabilities = (2 ** self.q_table[state[0], state[1],
            #                                  :]) / np.sum(2**self.q_table[state[0], state[1], :])
            #action = np.random.choice(ACTIONS, p=probabilities)

        return action

    def q_value(self, state, action):
        """
        Get Q value out of table
        Args:
            state (np.array)
            action (char)
        Returns
            q value (int)
        """
        index_of_action = np.where(ACTIONS == action)[0][0]

        return self.q_table[state[0], state[1], index_of_action]

    def q_update(self, backup):
        """
        Update Q value
        Args:
            backup (list): entries with [original state, action and reward]
        """
        td_error = 0
        first_step = backup[0]
        last_step = backup[len(backup)-1]
        # calculate q value for initial state
        q_old = self.q_value(first_step[0], first_step[1])
        for i, step in enumerate(backup):
            if i == len(backup)-1 and not last_step[3]:  # not terminal
                continue
            # step-wise reward discounting
            td_error += self.gamma**i * step[2]
            if np.isnan(td_error):
                raise Exception("NAN!")
        # calculate q value for last state, gamma-discounted
        if not last_step[3]:  # not terminal
            td_error += self.gamma**len(backup) * \
                self.q_value(last_step[0], last_step[1])
        # subtract original q-value
        td_error -= q_old
        index_of_action = np.where(ACTIONS == last_step[1])[0][0]
        # set new q-value for initial state-action-pair with learning rate alpha
        self.q_table[last_step[0][0], last_step[0][1],
                     index_of_action] += self.alpha * td_error
        new_value = self.q_table[last_step[0][0], last_step[0][1],
                                 index_of_action]
        if np.isnan(new_value):
            print(self.q_table)
            raise Exception("NAN!")
        if np.isinf(new_value):
            print(self.q_table)
            raise Exception("Inf!")

    def n_sarsa(self, environment, n_steps):
        backup = []
        terminal = False

        # one episode
        n = 0
        while n < 100:
            original_state = self.state
            # choose action
            action = self.choose_action(original_state)
            # take step in environment
            state, reward, terminal = environment.step(action)
            # environment.visualize()
            # update state
            self.state = state
            backup.append([original_state, action, reward, terminal])
            n += 1
            #print('STEP %i' % n)
            if len(backup) == n_steps:
                # update q value for first state in backup
                self.q_update(backup)
                backup.pop(0)  # remove first entry in backup
            if terminal == True:
                break
        for i, _ in enumerate(backup):
            if i < len(backup) - 1:
                self.q_update(backup[1:])
        print("%i steps" % n)


ACTIONS = np.array(['w', 'a', 's', 'd'])

ACTIONS_TO_EMOJI = {
    'w': '⬆',
    'a': '⬅',
    's': '⬇',
    'd': '➡',
}

WIND_TO_VECTORS = {
    'N': np.array([-1, 0]),
    'W': np.array([0, -1]),
    'S': np.array([1, 0]),
    'E': np.array([0, 1]),
}
ACTIONS_TO_VECTORS = {
    'w': np.array([-1, 0]),
    'a': np.array([0, -1]),
    's': np.array([1, 0]),
    'd': np.array([0, 1]),
}
