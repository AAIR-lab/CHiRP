import numpy as np
import random

class Qtable:
    def __init__(self):
        self._qtable = {}

    def get_qvalue(self, state, action):
        return self._qtable[state][action]

    def update_qvalue(self, state, action, qvalue):
        self._qtable[state][action] = qvalue

    def get_qvalues(self, state):
        return self._qtable[state]

    def update_qvalues(self, state, qvalues):
        self._qtable[state] = qvalues

    def get_max_qvalue(self, state):
        return np.max(list(self._qtable[state].values()))

    def get_best_action(self, state):
        return self.argmax_rand_tie_breaker_dictqtable(self._qtable[state])

    def get_np_init_qvalues(self, action_size, q_init):
        return np.ones((1, action_size))[0] * q_init
    
    def get_init_qvalues(self, action_size, q_init):
        np_qvalues = self.get_np_init_qvalues(action_size, q_init)
        return {i: qvalue for i, qvalue in enumerate(np_qvalues)}

    def initialize_qvalues(self, state, action_size, q_init):
        if state not in self._qtable:
            self._qtable[state] = self.get_init_qvalues(action_size, q_init)

    def initialize_qtable(self, qtable_full):
        self._qtable = qtable_full

    def delete_state(self, state):
        del self._qtable[state]

    def argmax_rand_tie_breaker_dictqtable(self, data):
        max_value = np.max(list(data.values()))
        max_indices = []
        for i in data.keys():
            if np.allclose(data[i], max_value):
                max_indices.append(i)
        res = random.choice(max_indices)
        return res
    
    def get_sum_qvalues(self, states, action_size, q_init):
        pulled = np.zeros((1, action_size))[0]
        for state in states:
            if state not in self._qtable:
                qvalues = self.get_np_init_qvalues(action_size, q_init)
            else:
                qvalues = self.get_qvalues(state)
            pulled += np.array([qvalues[i] for i in range(len(qvalues))])
        return pulled
