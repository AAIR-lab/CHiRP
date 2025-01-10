import random 
import numpy as np
import hyper_param

from src.data_structures.qvalue_table import Qtable

class Option:

    def __init__(self, initiation_set = None, termination_set = None, is_goal_option=False, optionid=None, cost=hyper_param.step_max, action_size=None): 
        self.initiation_set = initiation_set
        self.termination_set = termination_set # this is a set but contains just one state
        self._qtable = Qtable()
        self._is_goal_option = is_goal_option
        self._is_bridge_option = False
        self.optionid = optionid
        self.init_states = []
        self.term_states = []
        self.cat = None
        self.cost = cost
        self.abstract_cost = cost
        self.trajectories = []
        self.abstract_trajectories = []
        self.action_ids = None
        self.policy_states = set()
        
    def __hash__(self) -> int:
        return hash(frozenset(self.termination_set))
    
    def __str__(self) -> str:
        return str(self.termination_set)
    
    def __repr__(self) -> str:
        return str(self.termination_set)
    
    def __lt__(self, other):
        return True
    
    def __gt__(self, other):
        return True

    def get_tuple_state(self, state):
        state = state.split(" ->")[0]
        state = state.replace("(","")
        state = state.replace(")","")
        state = state.replace("'","")
        return tuple(state.split(", "))
    
    def get_goal_abs_state(self):
        goal_abs_states = []
        for state_abs in self.termination_set:
            if type(state_abs) != tuple:
                state_abs = self.get_tuple_state(state_abs)
            if type(state_abs[0]) == str:
                goal_abs_states.append(state_abs)
        return goal_abs_states[0]

    def is_satisfied(self, state, init_term_set):
        state = tuple(state)
        if type(init_term_set) != tuple:
            init_term_set = self.get_tuple_state(init_term_set)
        return tuple(state) == init_term_set

    def is_initiation_set(self, abs_state): # only abstract states in initiation sets
        # accepts state in the form of tuple e.g. ('1,4', '5,6')
        is_satisfied = False
        for init_node in self.initiation_set:
            if self.is_satisfied(abs_state, init_node):
                is_satisfied = True
                break
        return is_satisfied

    def is_termination_set(self, state):
        # accepts state in the form of tuple e.g. ('1,4', '5,6') or a concrete state tuple
        if self._is_goal_option:
            return tuple(state) in self.termination_set
        else:
            is_satisfied = False
            for term_node in self.termination_set:
                if self.is_satisfied(state, term_node):
                    is_satisfied = True
                    break
            return is_satisfied
    
    def argmax_rand_tie_breaker(self, data):
        max_value = np.max(data)
        max_indices = []
        for i in range(len(data)):
            if np.allclose(data[i], max_value):
                max_indices.append(i)
        index = random.randint(0,len(max_indices)-1)
        res = max_indices[index]
        return res

    def argmax_rand_tie_breaker_dictqtable(self, data):
        max_value = np.max(list(data.values()))
        max_indices = []
        for i in data.keys():
            if np.allclose(data[i], max_value):
                max_indices.append(i)
        index = random.randint(0,len(max_indices)-1)
        res = max_indices[index]
        return res
    
    def update_initiation_term(self, initiation_set):
        self.initiation_set = initiation_set
    
    def policy(self, state, stochastic=False, use_epsilon=0.2):
        abs_state = self.state(state)
        abs_state = tuple(abs_state)
        if abs_state not in self._qtable._qtable: # or (not self._is_goal_option and not self.is_initiation_set(abs_state)):
            if self.action_ids is None:
                action_size = len(list(self._qtable._qtable.values())[0])
                self.action_ids = list(range(0, action_size))
            return random.choice(self.action_ids)
        else:
            if stochastic and random.uniform(0,1) < use_epsilon:
                action_size = len(self._qtable.get_qvalues(abs_state))
                action_ids = list(range(0, action_size))
                return random.choice(action_ids)
            else:
                return self._qtable.get_best_action(abs_state)
    
    def initialize_qtable(self, qtable_full, only_init_set=False):
        if only_init_set:
            for abs_state in self.initiation_set:
                if type(abs_state) != tuple:
                    abs_state = self.get_tuple_state(abs_state)
                self._qtable._qtable[abs_state] = qtable_full._qtable[abs_state]
        else:
            for abs_state in qtable_full._qtable:
                self._qtable._qtable[abs_state] = qtable_full._qtable[abs_state]

        for abs_state in self.termination_set:
            if type(abs_state) != tuple:
                abs_state = self.get_tuple_state(abs_state)
            if abs_state in self._qtable._qtable:
                self._qtable._qtable[abs_state] = qtable_full._qtable[abs_state]

    def state(self, state_con):
        state_abstract = self.state_recursive(state_con, self.cat._root) 
        assert state_abstract in self.cat._leaves
        return state_abstract 

    def state_recursive(self, state_con, start_node):
        found = False
        result = None
        abstract_state = self.con_state_to_abs(state_con, start_node._split)    
        flag = False
        for n in start_node._child:
            if abstract_state == n._state:
                flag = True
                temp_node = n
                if len(n._child) == 0:
                    found = True 
                    result = abstract_state
        if start_node._child == []: return self.cat._root_abs_state
        if found: 
            return result
        else:
            if not flag:
                print (state_con)
            return self.state_recursive(state_con, temp_node)
        
    def con_state_to_abs(self, state_con, split):
        state = []
        for i in range(len(state_con)):
            for j in range (len(split[i]) -1):
                if state_con[i] >= split[i][j] and state_con[i] < split[i][j+1]:
                    state.append(str(split[i][j]) + ',' + str(split[i][j+1]) )
                    break
                elif state_con[i] >= split[i][j] and state_con[i] == split[i][j+1] and j+1==len(split[i])-1:
                    state.append(str(split[i][j]) + ',' + str(split[i][j+1]) )
                    break
        state = tuple(state)
        if len(state) == len(state_con): return state
        else: return None  

    def add_term_states(self, optimal_trajectory, optimal_abstract_trajectory):
        self.term_states = []
        for i, step in enumerate(optimal_abstract_trajectory):
            if step[2] in self.termination_set:
                self.term_states.append(optimal_trajectory[i].next_state) # s'
                break