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
        self.wind_probability = 0.2
        self.reset()

    def reset(self):
        self.agent_position = np.array([0, 0])

        self.reward_matrix = np.array([
            [0, 0, 0, 0, 20],
            [0, 0, 0, 0, -2],
            [0, 0, 0, 0, -4],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]])

        self.map_matrix = np.array([
            ['S', 'X', '0', '0', '0'],
            ['S', 'E', '0', 'E', 'S'],
            ['S', 'N', 'X', '0', 'W'],
            ['W', 'X', '0', '0', 'X'],
            ['0', '0', '0', '0', 'X']])

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
        if map_tile in ['N', 'S', 'W', 'E'] and self.wind_probability > random.uniform(0, 1):
            next_step = WIND_TO_VECTORS[map_tile]
        else:
            next_step = action_vector
        next_pos = next_step + self.agent_position

        # check for validity of the step (if in gridworld and no wall)
        if self.valid_step(next_pos):
            # GO!🚨
            reward = self.reward_matrix[next_pos[0], next_pos[1]]
            self.agent_position = next_pos

        return self.agent_position, reward, reward == 20

    def valid_step(self, next_pos):
        """
        Check if next step the agents want to take is valid. Valid being in the gridworld field and not running into a wall.
            Args:
                next_pos (np.array): future position of the agent
            returns:
                bool if action is valid
        """
        if next_pos[0] < 0 or next_pos[1] < 0 or next_pos[0] > self.map_matrix.shape[0] or next_pos[1] > self.map_matrix.shape[1] or self.map_matrix[next_pos[0], next_pos[1]] == 'X':
            return False
        else:
            return True

    def visualize(self):
        """
        Visualizes the environment including the agent
        (prints characters to stdout)
        """
        # clear previous visualisation
        if platform.system() == ('Linux' or 'Darwin'):
            os.system('clear')
        elif platform.system() == 'Windows':
            os.system('cls')

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
    def __init__(self):
        self.action = np.array(['w', 'a', 's', 'd'])
        self.state = np.array([0, 0])


ACTIONS = np.array(['w', 'a', 's', 'd'])

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
