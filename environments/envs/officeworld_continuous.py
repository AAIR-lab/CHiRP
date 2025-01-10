import numpy as np
import random 
import os
import copy
import gym

class OfficeWorldContinuousEnvironment(gym.Env):
    def __init__(self, map_name, include_goal_variable= False):
        super().__init__()
        self._visited = []
        self._action_space = ['up','down','left','right']
        self._action_size = len(self._action_space)
        self.action_space = gym.spaces.Discrete(self._action_size)
        
        self._action_probs = {0:[2,3], 1:[2,3], 2:[0,1], 3:[0,1]}
        self._stoch_prob = 0.8

        self.load_map(map_name)
        self._visit_map = np.zeros_like(self._maze)
        self.include_goal_variable = include_goal_variable

        self._dimension = self._maze.shape
        self._coffee_locs = self._init_coffee_locs.copy()
        self._mail_locs = self._init_mail_locs.copy()
        self._has_coffee = 0
        self._has_mail = 0

        self._obj_locs = [list(loc) for loc in self._coffee_locs] + [list(loc) for loc in self._mail_locs]
        self._state_ranges = self.get_state_ranges()
        self._n_state_variables = len(self._state_ranges)
        self._vars_split_allowed = [1 for i in range(self._n_state_variables)]
        self._vars_split_allowed_initially = [1 for i in range(len(self._original_vars))] + [0 for i in range(self._n_state_variables - len(self._original_vars))]
        self._train_option = None
        self.observation_space = gym.spaces.Box(low=self._state_ranges[:,0], high = self._state_ranges[:,1], dtype = np.int32)
        self.is_int_variable = [False, False, True, True]

    def initialize_problem(self, args, step_max):
        start = args["init_vars"][:2]
        if len(args["init_vars"]) > 2:    
            self._problem_has_coffee = args["init_vars"][2]
            self._problem_has_mail = args["init_vars"][3]
        else:
            self._problem_has_coffee = 0
            self._problem_has_mail = 0
        self._office_loc = tuple(args["goal_vars"])
        self._goal = list(self._office_loc)
        if not self.include_goal_variable:
            self._init_state = copy.deepcopy(start) + [self._has_coffee, self._has_mail] 
            self._goal_state = copy.deepcopy(self._goal) + [1,1]
        else:
            self._init_state = copy.deepcopy(start) + [self._has_coffee, self._has_mail] + copy.deepcopy(self._goal)
            self._goal_state = copy.deepcopy(self._goal) + [1,1] + copy.deepcopy(self._goal)
        # self._goal_range = [(item,item) for item in self._goal_state] 
        self._goal_range = self.get_goal_range()
        self.reset()
        self._relevant_states = self.get_relevant_states()
        self._step_max = step_max

    def get_goal_range(self):
        goal_range = []
        allowed_dx_dy = 0.9
        for i in range(len(self._goal_state)):
            item = self._goal_state[i]
            if i==0 or i==1:
                item1 = max(0, item - allowed_dx_dy)
                item2 = item + allowed_dx_dy
            else:
                item1 = copy.deepcopy(item)
                item2 = copy.deepcopy(item)
            goal_range.append((item1, item2))
        return goal_range

    def reset(self):
        self.steps = 0
        self._has_coffee = copy.deepcopy(self._problem_has_coffee)
        self._has_mail = copy.deepcopy(self._problem_has_mail)
        self._state = copy.deepcopy(self._init_state)
        # assert self._state[:2] not in self.walls, "invalid agent location"
        loc = self.get_closest_int_location(self._state[:2])
        assert self._maze[loc[0],loc[1]] != 1, f"invalid agent location {loc}"
        return self._state

    def get_state_ranges(self):
        ranges = [ [0,self._dimension[0]], # robot y
                   [0,self._dimension[1]], # robot x
                 ]
        ranges += [ [0,2], # has_coffee
                    [0,2], # has_mail
                 ]
        self._original_vars = [i for i in range(0,len(ranges))]
        self._goal_vars = []
        if self.include_goal_variable:
            # Destination location for passengers
            ranges += [[0, self._dimension[0]], [0, self._dimension[1]]]
            self._goal_vars = [len(ranges)-2, len(ranges)-1]
        return np.array(ranges)
    
    def step (self, action_index_input, abstract = None):
        [a,b] = self._state[:2]
        reward = 0 # the episode's reward (-100 for pitfall, 0 for reaching the goal, and -1 otherwise)
        done = False # termination flag is true if the agent falls in a pitfall or reaches to the goal
        success = False
        goal_reward = 500
        pseudo_reward = 500
        dx, dy = 0.6, 0.6
        
        # flag_pitfall = False
        action_index = self.action_stochastic (action_index_input)
        action = self.index_to_action (action_index)
        self.steps+=1
        if action == 'up':
            next_loc = [a-dy,b]
        elif action == 'down':
            next_loc = [a+dy,b]
        elif action == 'left':
            next_loc = [a, b-dx] 
        elif action == 'right':
            next_loc = [a, b+dx] 

        next_int_loc = self.get_closest_int_location(next_loc)

        # if next_loc in self.walls:
        if not self.in_bound(next_loc) or self._maze[next_int_loc[0], next_int_loc[1]] == 1:
            next_loc = tuple([a,b])
        if self.in_bound(next_loc):
            self._state[:2] = next_loc
        else:
            next_loc = self._state[:2]

        if self._has_coffee and self._has_mail and next_int_loc == self.get_closest_int_location(self._office_loc):
            if self._train_option is None: 
                reward = goal_reward
                done = True
                success = True
            self._state = self.get_updated_state(next_loc, self._has_coffee, self._has_mail)     
        else:
            if not self._has_coffee and tuple(next_int_loc) in self._coffee_locs:
                self._has_coffee = 1
            elif self._has_coffee and not self._has_mail and tuple(next_int_loc) in self._mail_locs:
                self._has_mail = 1
            self._state = self.get_updated_state(next_loc, self._has_coffee, self._has_mail)
            
        # print(self._state, self._goal_state)
        if self._train_option is not None:
            done = False
            success = False
            if self._train_option._is_goal_option:
                if self.is_goal_state(self._state):
                    reward += pseudo_reward
                    done = True
                    success = True
            else:
                if self._train_option.is_termination_set(abstract.state(self._state)):
                    reward += pseudo_reward
                    done = True
                    success = True
        
        if self.steps >= self._step_max:
            done = True 

        
        return self._state, reward, done, {'success':success, 'steps':self.steps}
        
    def get_updated_state(self, loc, has_coffee, has_mail):
        state = [loc[0], loc[1], has_coffee, has_mail]
        if self.include_goal_variable:
            state += self._goal
        return state

    def action_stochastic (self, action_index):
        if random.uniform (0,1) > self._stoch_prob:
            if random.uniform (0,1) > 0.5 : 
                action_index_stoch = self._action_probs[action_index][0]
            else: 
                action_index_stoch = self._action_probs[action_index][1]
        else: 
            action_index_stoch = action_index
        return action_index_stoch
    
    # checks if a location is withing the env bound
    def in_bound (self, loc):
        flag = False
        if loc[0] < self._dimension[0] and loc[0] >= 0:
            if loc[1] < self._dimension[1] and loc[1] >= 0:
                flag = True
        return flag

    # action_index into action
    def index_to_action (self, action_index):
        return self._action_space [action_index]
    
    
    # state to state_index
    def state_to_index (self, state):
        # return state[0]*self._dimension[0] + state[1]
        return tuple(state)
    
    def reset_visited (self):
        self._visit_map = np.zeros_like(self._maze)

    def update_visited (self, state):
        flag = True
        for i in self._visited:
            if state == i: flag = False
        if flag: self._visited.append(state)

    def get_relevant_states(self): # for heatmap
        relevant_states = list()
        relevant_states.append(self._init_state)
        relevant_states.append(self._goal_state)
        for loc in self._obj_locs + [self._init_state[:2]]:
            state1 = list(loc) + [0, 0]
            state2 = list(loc) + [0, 1]
            state3 = list(loc) + [1, 0]
            state4 = list(loc) + [1, 1]
            if self.include_goal_variable:
                state1 += self._goal
                state2 += self._goal
                state3 += self._goal
                state4 += self._goal
            relevant_states.append(state1)
            relevant_states.append(state2)
            relevant_states.append(state3)
            relevant_states.append(state4)
        return relevant_states

         
    def load_map(self, map_name):

        self._init_coffee_locs = []
        self._init_mail_locs = []
        self._rooms = {}
        self._maze = self.get_map_rooms(map_name)

        # height: 9 width: 12
        #    0 1 2 3 4 5 6 7 8 9 10 11
        # --------------------------
        #  0|     *     *     *      |
        #   |                        |
        #  1|  a                 b   |
        #   |                        |
        #  2|     *f    *     *      |
        #   |--  ----  ----  ----  --|
        #  3|     *     *     *      |
        #   |                        |
        #  4|     *  g  *  e  *      |
        #   |                        |
        #  5|     *     *     *      |
        #   |--  ----------------  --|
        #  6|     *     *    f*      |
        #   |                        |
        #  7|  d A               c   |
        #   |                        |
        #  8|     *     *     *      |
        #  --------------------------

    def get_map_rooms(self, file_name):
        basepath = os.getcwd()
        image_path = basepath+"/environments/maps/" + file_name + ".map"
        f = open(image_path)
        string = ""
        for line in f:
            string += line
        env_map = self.string_to_bool_map(string)
        return env_map

    def string_to_bool_map(self, str_map):
        bool_map = []
        for i, row in enumerate(str_map.split('\n')[:]):
            bool_map_row = []
            for j,r in enumerate(row):
                if r==' ':
                    bool_map_row.append(0)
                elif r=='x':
                    bool_map_row.append(1)
                elif r == 'a':
                    self._rooms['a'] = (i,j)
                    bool_map_row.append(0)
                elif r == 'b':
                    self._rooms['b'] = (i,j)
                    bool_map_row.append(0)
                elif r == 'c':
                    self._rooms['c'] = (i,j)
                    bool_map_row.append(0)
                elif r == 'd':
                    self._rooms['d'] = (i,j)
                    bool_map_row.append(0)
                elif r == 'o':
                    # self._office_loc = (i,j) # now changing it and passing as hyperparameter
                    bool_map_row.append(0) 
                elif r == 'm':
                    self._init_mail_locs.append((i,j))
                    bool_map_row.append(0)
                elif r == 'f':
                    self._init_coffee_locs.append((i,j))         
                    bool_map_row.append(0)   
                # bool_map_row.append(int(not (r==' ')))
            bool_map.append(bool_map_row)
        return np.array(bool_map)
    
    def get_closest_int_location(self, location):
        rounded_x = int(round(location[0],0))
        rounded_y = int(round(location[1],0))
        rounded_x = min(rounded_x, self._dimension[0]-1)
        rounded_y = min(rounded_y, self._dimension[1]-1)
        return [rounded_x, rounded_y]
    
    def seed(self, seed):
        # self.seed = seed
        pass
    
class OfficeWorldContinuous(OfficeWorldContinuousEnvironment):
    
    def __init__(self, map_name, include_goal_variable=False):
        super().__init__(map_name, include_goal_variable)
        self._n_state_variables = len(self._state_ranges)
        self._vars_split_allowed = [1 for i in range(self._n_state_variables)]
        self._vars_split_allowed_initially = [1 for i in range(len(self._original_vars))] + [0 for i in range(self._n_state_variables - len(self._original_vars))]
        
    # def step(self, action_index_input, abstract=None):
    #     state, reward, done, info = super().step(action_index_input, abstract)
    
    def get_goal_abstract_states(self, abstract_states):
        goal_abs_states = set()
        for abs_state in abstract_states:
            temp_abs_state = [[np.float32(item2) for item2 in item.split(",")] for item in abs_state]
            goal = True
            for i in range(len(temp_abs_state)):
                low1, high1 = temp_abs_state[i][0], temp_abs_state[i][1]
                low2, high2 = self._goal_range[i][0], self._goal_range[i][1]
                if (low1 < low2 and high1 <= low2) or (low2 < low1 and high2 <= low1):
                    goal = False
                    break
            if goal:
                goal_abs_states.add(abs_state)
        return goal_abs_states
    
    def is_goal_state(self, state):
        for i in range(len(state)):
            val = state[i]
            if not (val >= self._goal_range[i][0] and val <= self._goal_range[i][1]*1.001):
                return False
        return True