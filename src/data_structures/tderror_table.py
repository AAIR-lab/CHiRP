import numpy as np
import random

class TDerrorTable:
    def __init__(self, action_size):
        self._table = {}
        self._action_size = action_size

    def get_tderror(self, state, action):
        return self._table[state][action]

    def initialize_state_action(self, state, action):
        self._table[state][action] = []

    def initialize_state(self, state):
        self._table[state] = {}
        for action_index in range(self._action_size):
            self.initialize_state_action(state, action_index)

    def append_tderror(self, state, action, tderror):
        self._table[state][action].append(tderror)
        
