import numpy as np
import random 
import pickle

import hyper_param
from src.data_structures.qvalue_table import Qtable
from src.data_structures.tderror_table import TDerrorTable

############### All are Qlearning agents ###########################

class Agent:
    def __init__(self, env, decay, epsilon = hyper_param.epsilon_con, eps_min = hyper_param.epsilon_min, gamma = hyper_param.gamma, alpha = hyper_param.alpha_con, q_init = 100):
        self._epsilon = epsilon
        self._epsilon_min = eps_min
        self._decay = decay
        self._gamma = gamma
        self._alpha = alpha
        self._q_init = q_init
        self._env = env
        self._action_size = self._env._action_size
        self._qtable = Qtable()
        #2d array, rows are states and cols are q-value action pairs

    def batch_train(self, batch):
        for b in batch:
            self.train(b)

    def get_init_qvalues(self):
        return self._qtable.get_init_qvalues(self._action_size, self._q_init)

    def initialize_qvalues(self, state):
        return self._qtable.initialize_qvalues(state, self._action_size, self._q_init)

    def train(self, sample):
        state, next_state, action_index, reward = sample.state, sample.next_state, sample.action, sample.reward
        state = self._env.state_to_index(state)
        next_state = self._env.state_to_index(next_state)
        self._qtable.initialize_qvalues(state, self._action_size, self._q_init)
        self._qtable.initialize_qvalues(next_state, self._action_size, self._q_init)
        old_state_qvalue = self._qtable.get_qvalue(state, action_index)
        best_next_state_qvalue = self._qtable.get_max_qvalue(next_state)
        td_target = reward + self._gamma * best_next_state_qvalue
        td_error = td_target - old_state_qvalue
        new_state_qvalue = old_state_qvalue + self._alpha * td_error
        self._qtable.update_qvalue(state, action_index, new_state_qvalue)

    def policy (self, state, sample=True):
        state = self._env.state_to_index(state)
        self.initialize_qvalues(state, self._action_size, self._q_init)
        if random.uniform (0,1) < self._epsilon and sample:
            action_index = random.randint (0, self._action_size-1)
        else:
            action_index = self._qtable.get_best_action(state)
        return action_index

    def policy_greedy (self, state):
        state = self._env.state_to_index(state)
        self.initialize_qvalues(state, self._action_size, self._q_init)
        action_index = self._qtable.get_best_action(state)
        return action_index

    def decay (self):
        if self._epsilon > self._epsilon_min: self._epsilon =  self._epsilon * self._decay

    def get_max_q (self, state):
        state = self._env.state_to_index(state)
        self.initialize_qvalues(state, self._action_size, self._q_init)
        max_value = max(self._qtable[state][:])
        return max_value
 
    def pull_qvalues_single (self, states):
        return self._qtable.get_sum_qvalues(states, self._action_size, self._q_init)


class AbstractAgent:
    def __init__(self, env, decay, epsilon = hyper_param.epsilon, eps_min = hyper_param.epsilon_min, gamma = hyper_param.gamma, alpha = hyper_param.alpha, q_init = 200):
        self._epsilon = epsilon
        self._epsilon_min = eps_min
        self._gamma = gamma
        self._alpha = alpha
        self._env = env
        self._action_size = self._env._action_size
        self._q_init = q_init
        self._decay = decay
        self._abstract = None
        self.abs_to_con = {}
        self._qtable = Qtable()
        self._qtable_concrete_approx = Qtable()

    def get_init_qvalues (self):  
        return self._qtable.get_init_qvalues(self._action_size, self._q_init)

    def batch_train (self, batch):
        for b in batch:
            self.train(b)

    def reuse_qvalues(self, pickle_path, state_mapping):
        with open(pickle_path, 'rb') as f:         
            temp = pickle.load(f)
            for state, qvalues in temp.items():
                abstract_state = state_mapping[state]
                transfer_qvalues = [v for v in qvalues[0]]
                self._qtable.update_qvalues(abstract_state, np.array([transfer_qvalues]))

    def train(self, sample):
        state, action_index, next_state, reward = sample.state, sample.action, sample.next_state, sample.reward
        old_state_qvalue = self._qtable.get_qvalue(state, action_index)
        best_next_state_qvalue = self._qtable.get_max_qvalue(next_state)
        td_target = reward + self._gamma * best_next_state_qvalue
        td_error = td_target - old_state_qvalue
        new_state_qvalue = old_state_qvalue + self._alpha * td_error
        self._qtable.update_qvalue(state, action_index, new_state_qvalue)

    def estimate_concrete_qvalues(self, sample):
        state, action_index, next_abs_state, reward = sample.state, sample.action, sample.next_state, sample.reward
        self._qtable_concrete_approx.initialize_qvalues(state, self._action_size, self._q_init)
        best_next_abs_state_qvalue = self._qtable.get_max_qvalue(next_abs_state)
        new_state_qvalue = reward + self._gamma * best_next_abs_state_qvalue
        self._qtable_concrete_approx.update_qvalue(state, action_index, new_state_qvalue)

    def pull_qvalues_single (self, states):
        return self._qtable_concrete_approx.get_sum_qvalues(states, self._action_size, self._q_init)

    def policy (self, state_abs, sample=True, include_options=True, use_epsilon=None):
        if use_epsilon is None:
            use_epsilon = self._epsilon
        if random.uniform (0,1) < use_epsilon and sample:
            action_ids = list(range(0, self._action_size))
            action_index = random.choice(action_ids)
        else:
            action_index = self._qtable.get_best_action(state_abs)
        return action_index

    def update_qtable (self, state_abs):
        if state_abs not in self._qtable._qtable:
            if self._abstract._bootstrap == "from_estimated_concrete":
                pulled = self._abstract.bootstrap(state_abs, self.abs_to_con[state_abs])
            else:
                pulled = self._abstract.bootstrap(state_abs)
            pulled = {i:qvalue for i, qvalue in enumerate(pulled)}
            self._qtable.update_qvalues(state_abs, pulled)

    def initialize_tderror_state(self, state_abs):
        if state_abs not in self._tderror_table._table:
            self._tderror_table.initialize_state(state_abs)

    def initialize_tderror (self):
        self.abs_to_con = {} #TODO: use this?
        self._tderror_table = TDerrorTable(self._action_size)
        for state_abs in self._qtable._qtable:
            if state_abs in self._abstract._tree._leaves:
                self._tderror_table.initialize_state(state_abs)
    
    def log_values (self, sample):
        state, next_state, action_index, reward = sample.state, sample.next_state, sample.action, sample.reward
        old_state_qvalue = self._qtable.get_qvalue(state, action_index)
        best_next_state_qvalue = self._qtable.get_max_qvalue(next_state)
        td_target = reward + self._gamma * best_next_state_qvalue
        td_error = td_target - old_state_qvalue
        self._tderror_table.append_tderror(state, action_index, td_error)
       
    def decay (self):
        if self._epsilon > self._epsilon_min: self._epsilon =  self._epsilon * self._decay


class IntraOptionAbstractAgent:
    def __init__(self, env, decay, epsilon = hyper_param.epsilon, eps_min = hyper_param.epsilon_min, gamma = hyper_param.gamma, alpha = hyper_param.alpha, q_init = 100):
        self._epsilon = epsilon
        self._epsilon_min = eps_min
        self._decay = decay
        self._alpha = alpha
        self._gamma = gamma
        self._env = env
        self._action_size = self._env._action_size
        self._q_init = q_init
        self._abstract = None
        self._qtable = Qtable()
        self._qtable_concrete_approx = Qtable()

    def get_init_qvalues (self):  
        return self._qtable.get_init_qvalues(self._action_size, self._q_init)
    
    def initialize_qtable(self, option, qtable_full):
        for abs_state in option.initiation_set:
            if type(abs_state) != tuple:
                abs_state = option.get_tuple_state(abs_state)
            self._qtable.update_qvalue(abs_state, qtable_full._qtable[abs_state])
        for abs_state in option.termination_set:
            if type(abs_state) != tuple:
                abs_state = option.get_tuple_state(abs_state)
            self._qtable.update_qvalue(abs_state, qtable_full._qtable[abs_state])
    
    def update_qtable(self, state_abs):
        state_abs = tuple(state_abs)
        self._qtable.initialize_qvalues(state_abs, self._action_size, self._q_init)

    def batch_train (self, batch):
        for b in batch:
            self.train(b[0], b[1], b[2], b[3])

    def train(self, sample):
        state_abs, new_state_abs, action_index, reward = sample.state, sample.next_state, sample.action, sample.reward
        state_abs, new_state_abs = tuple(state_abs), tuple(new_state_abs)
        old_state_qvalue = self._qtable.get_qvalue(state_abs, action_index)
        best_next_state_qvalue = self._qtable.get_max_qvalue(new_state_abs)
        td_target = reward + self._gamma * best_next_state_qvalue
        td_error = td_target - old_state_qvalue
        new_state_qvalue = old_state_qvalue + self._alpha * td_error
        self._qtable.update_qvalue(state_abs, action_index, new_state_qvalue)

    def estimate_concrete_qvalues(self, sample):
        state, action_index, next_abs_state, reward = sample.state, sample.action, sample.next_state, sample.reward
        self._qtable_concrete_approx.initialize_qvalues(state, self._action_size, self._q_init)
        best_next_abs_state_qvalue = self._qtable.get_max_qvalue(next_abs_state)
        new_state_qvalue = reward + self._gamma * best_next_abs_state_qvalue
        self._qtable_concrete_approx.update_qvalue(state, action_index, new_state_qvalue)

    def pull_qvalues_single (self, states):
        return self._qtable_concrete_approx.get_sum_qvalues(states, self._action_size, self._q_init)

    def policy (self, state_abs, sample=True):
        state_abs = tuple(state_abs)
        if sample and random.uniform (0,1) < self._epsilon:
            action_ids = list(range(0, self._action_size))
            action_index = random.choice(action_ids)
        else:
            action_index = self._qtable.get_best_action(state_abs)
        return action_index

    def initialize_tderror_state(self, state_abs):
        if state_abs not in self._tderror_table._table:
            self._tderror_table.initialize_state(state_abs)

    def initialize_tderror(self):
        self.abs_to_con = {}
        self._tderror_table = TDerrorTable(self._action_size)
        for state_abs in self._qtable._qtable:
            if state_abs in self._abstract._tree._leaves:
                self._tderror_table.initialize_state(state_abs)
        
    def log_values (self, sample):
        state, next_state, action_index, reward = sample.state, sample.next_state, sample.action, sample.reward
        old_state_qvalue = self._qtable.get_qvalue(state, action_index)
        best_next_state_qvalue = self._qtable.get_max_qvalue(next_state)
        td_target = reward + self._gamma * best_next_state_qvalue
        td_error = td_target - old_state_qvalue
        self._tderror_table.append_tderror(state, action_index, td_error)

    def decay (self):
        if self._epsilon > self._epsilon_min: self._epsilon =  self._epsilon * self._decay
