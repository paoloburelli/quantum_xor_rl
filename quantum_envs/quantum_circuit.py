import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import List, Optional


class QuantumCircuit(gym.Env):

    def __init__(self, cardinality, init_steps_limit=None, episode_steps_limit=None):
        super().__init__()
        self.render_mode = None
        self.identity = np.identity(cardinality, dtype=np.uint8)
        self.cardinality = cardinality

        self.action_space = spaces.Discrete(self.cardinality * (self.cardinality - 1))
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(1, cardinality, cardinality), dtype=np.uint8)

        self.state = np.array([np.identity(self.cardinality, dtype=np.uint8)], dtype=np.uint8)

        self.valid_actions = np.concatenate((np.stack(np.triu_indices(self.cardinality, k=1), axis=-1),
                                             np.stack(np.tril_indices(self.cardinality, k=-1), axis=-1)))

        self.max_init_permutations = init_steps_limit if init_steps_limit is not None else pow(self.cardinality, 2)
        self.init_operations = []
        self.correct_lines = self._get_correct_lines()
        self.operations_count = 0
        self.max_episode_steps = episode_steps_limit if episode_steps_limit is not None else pow(self.cardinality, 2)
        self.reset()

    def action_masks(self) -> List[bool]:
        return [not self.correct_lines[v[0]] for v in self.valid_actions]

    def apply_operator(self, indices):
        self.state[0, indices[0]] = np.logical_xor(self.state[0, indices[0]], self.state[0, indices[1]])
        self.correct_lines = self._get_correct_lines()

    def step(self, action):

        self.operations_count += 1

        correct_lines_count_t0 = self.correct_lines.sum()
        hamming_distance_t0 = np.logical_xor(self.state[0], self.identity).sum()

        self.apply_operator(self.valid_actions[action])

        hamming_distance_t1 = np.logical_xor(self.state[0], self.identity).sum()
        correct_lines_count_t1 = self.correct_lines.sum()

        if hamming_distance_t1 == 0:
            reward = 0.7
        elif correct_lines_count_t1 != correct_lines_count_t0:
            reward = 0.2 * (correct_lines_count_t1 - correct_lines_count_t0) / self.cardinality
        elif hamming_distance_t1 != hamming_distance_t0:
            reward = 0.1 * (hamming_distance_t0 - hamming_distance_t1) / pow(self.cardinality, 2)
        else:
            reward = -0.001 / pow(self.cardinality, 2)

        return self.state, reward, hamming_distance_t1 == 0, self.operations_count >= self.max_episode_steps, {}

    def _get_correct_lines(self):
        return np.logical_xor(self.state[0], self.identity).sum(axis=1) == 0

    def reset(self, seed=None, options=None):
        self.operations_count = 0
        rng = np.random.default_rng(seed)

        self.state[0] = np.identity(self.cardinality, dtype=np.uint8)

        self.init_operations = [rng.choice(self.cardinality, size=2, replace=False) for i in
                                range(rng.choice(self.max_init_permutations) + 1)]

        for p in self.init_operations:
            self.apply_operator(p)

        return self.state, {}


if __name__ == '__main__':
    env = QuantumCircuit(4, 16)
    print(env.state)
    terminated = False
    count = 0
    while not terminated:
        action = env.action_space.sample()
        mask = env.action_masks()
        print(action, env.valid_actions[action])
        state, reward, terminated, truncated, info = env.step(action)
        print(state, reward, terminated, truncated, info)
        count += 1
    print(count)
