import numpy as np
import random 
import copy
import gym
import src.misc.map_maker as map_maker

class MazeWorldContinuousEnvironment(gym.Env):
    def __init__(self, env_name, include_goal_variable=False):
        super().__init__()
        self._visited = []
        self._action_space = ['up','down','left','right']
        self._maze = map_maker.get_map(env_name)
        self._dimension = self._maze.shape
        self._action_size = len(self._action_space)
        self.action_space = gym.spaces.Discrete(self._action_size)
        self._action_probs = {0:[2,3], 1:[2,3], 2:[0,1], 3:[0,1]}
        self._stoch_prob = 0.8
        self._visit_map = np.zeros_like(self._maze)
        # self._step_max = step_max
        self.include_goal_variable = include_goal_variable
        self._state_ranges = self.get_state_ranges()
        self.observation_space = gym.spaces.Box(low=self._state_ranges[:,0], high = self._state_ranges[:,1], dtype = np.int32)

        self._n_state_variables = len(self._state_ranges)
        self._vars_split_allowed = [1 for i in range(self._n_state_variables)]
        self._obj_locs = []
        # self._relevant_states = self.get_relevant_states()
        self._train_option = None
        self._vars_split_allowed_initially = [1 for i in range(len(self._original_vars))] + [0 for i in range(self._n_state_variables - len(self._original_vars))]

        self.allowed_dx_dy = 0.6
        self.is_int_variable = [False, False]

    def initialize_problem(self, args, step_max):
        start, goal = args["init_vars"][:2], args["goal_vars"]
        assert not self._maze[goal[0], goal[1]], "invalid goal location"

        if not self.include_goal_variable:
            self._init_state = copy.deepcopy(start)
            self._goal_state = copy.deepcopy(goal)
        else:
            self._init_state = copy.deepcopy(start) + copy.deepcopy(goal)
            self._goal_state = copy.deepcopy(goal) + copy.deepcopy(goal)
        # self._goal_range = [(item,item) for item in self._goal_state] 
        self._goal_range = self.get_goal_range()

        # self._state = copy.deepcopy(self._init_state)
        self.reset()
        self._relevant_states = self.get_relevant_states()
        self._step_max = step_max

    def get_goal_range(self):
        goal_range = []
        allowed_dx_dy = self.allowed_dx_dy
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
    
    # def is_goal_satisfied(self, state):
    #     flag = False
    #     if state[0] < self._goal_range[0][0] and state[0] >= self._goal_range[0][1]:
    #         if state[1] < self._goal_range[1][0] and state[1] >= self._goal_range[1][1]:
    #             flag = True
    #     return flag

    def is_goal_state(self, state):
        for i in range(len(state)):
            val = state[i]
            if not (val >= self._goal_range[i][0] and val <= self._goal_range[i][1]*1.001):
                return False
        return True

    def get_state_ranges(self):
        ranges = [ [0,self._dimension[0]], # robot y
                   [0,self._dimension[1]], # robot x
                 ]
        self._original_vars = [i for i in range(0,len(ranges))]
        self._goal_vars = []
        if self.include_goal_variable:
            ranges += [ [0,self._dimension[0]], # goal y
                        [0,self._dimension[1]], # goal x
                    ]
            self._goal_vars = [len(ranges)-2, len(ranges)-1]
            
        return np.array(ranges)

    def get_relevant_states(self): 
        relevant_states = set()
        relevant_states.add(tuple(self._init_state))
        relevant_states.add(tuple(self._goal_state))
        return relevant_states

    def reset(self):
        self.steps = 0
        self._state = copy.deepcopy(self._init_state)
        self._goal_range = self.get_goal_range()
        loc = self.get_closest_int_location(self._state[:2])
        assert self._maze[loc[0],loc[1]] != 1, "invalid agent location"
        return self._state

    def step (self, action_index_input, abstract=None):
        wall_cost = -2
        step_cost = -1
        goal_reward = 500
        self.steps += 1
        [a,b] = self._state[:2]
        reward = 0 # the episode's reward (-100 for pitfall, 0 for reaching the goal, and -1 otherwise)
        done = False # termination flag is true if the agent falls in a pitfall or reaches to the goal
        success = False
        dx, dy = 0.6, 0.6
        
        action_index = self.action_stochastic (action_index_input)
        action = self.index_to_action (action_index)
        if action == 'up':
            next_loc = [a-dy,b]
        elif action == 'down':
            next_loc = [a+dy,b]
        elif action == 'left':
            next_loc = [a, b-dx] 
        elif action == 'right':
            next_loc = [a, b+dx] 
        next_int_loc = self.get_closest_int_location(next_loc)
        if self.is_goal_state(next_loc):
            if self._train_option is None:
                reward += goal_reward
                done = True
                success = True
        elif not self.in_bound(next_loc) or self._maze[next_int_loc[0],next_int_loc[1]] == 1:
            next_loc = self._state[:2]
            reward += wall_cost
        else:
            reward += step_cost
        self._state = [next_loc[0],next_loc[1]]
        if self.include_goal_variable:
            self._state += self._goal_state[:2]
        
        if self.steps >= self._step_max:
            done = True 

        return self._state, reward, done, {"success": success, "steps": self.steps}

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


    def action_stochastic (self, action_index):
        if random.uniform (0,1) > self._stoch_prob:
            if random.uniform (0,1) > 0.5 : 
                action_index_stoch = self._action_probs[action_index][0]
            else: action_index_stoch = self._action_probs[action_index][1]
        else: action_index_stoch = action_index
        return action_index_stoch

    # checks if a location is within the env bound
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

    def seed(self, seed):
        pass
    
    def get_closest_int_location(self, location):
        rounded_x = int(round(location[0],0))
        rounded_y = int(round(location[1],0))
        rounded_x = min(rounded_x, self._dimension[0]-1)
        rounded_y = min(rounded_y, self._dimension[1]-1)
        return [rounded_x, rounded_y]

class MazeWorldContinuous(MazeWorldContinuousEnvironment):

    def __init__(self, env_map, include_goal_variable=False):
        super().__init__(env_map, include_goal_variable=include_goal_variable)
        self._n_state_variables = len(self._state_ranges)
        self._vars_split_allowed = [1 for i in range(self._n_state_variables)]
        self._vars_split_allowed_initially = [1 for i in range(len(self._original_vars))] + [0 for i in range(self._n_state_variables - len(self._original_vars))]
        self._train_option = None

    # def get_state_ranges(self):
    #     state_ranges =  super().get_state_ranges()
    #     state_ranges = list(state_ranges[:-2]) + [[0,2]] + list(state_ranges[-2:])
    #     self._goal_vars = [len(state_ranges)-2, len(state_ranges)-1]
    #     return np.array(state_ranges)
    
    def initialize_problem(self, args, step_max):
        self.start, self.goal = args["init_vars"][:2], args["goal_vars"]
        self.reset()
        self._relevant_states = self.get_relevant_states()
        self._step_max = step_max

    def reset(self):
        start_loc, goal_loc = copy.deepcopy(self.start), copy.deepcopy(self.goal)
        self._init_state = self.get_updated_state(start_loc)
        self._goal_state = self.get_updated_state(goal_loc)
        self._goal_range = self.get_goal_range()
        self._state = copy.deepcopy(self._init_state)
        self.steps = 0
        self._obj_locs = []
        loc = self.get_closest_int_location(self._state[:2])
        assert self._maze[loc[0],loc[1]] != 1, "invalid agent location"
        return self._state

    def get_updated_state(self, agent_loc):
        state = agent_loc
        # state += self.get_relation_vars(agent_loc, self.goal)
        if self.include_goal_variable:
            state += self.goal
        return state
    
    # def get_relation_vars(self, agent_loc, goal_loc):
    #     return [int(agent_loc==goal_loc)]
    
    def step(self, action_index_input, abstract=None):
        state, reward, done, info = super().step(action_index_input, abstract)
        self._state = self.get_updated_state(state[:2])
        goal_reward, pseudo_reward = 500, 500
        success = info["success"]
        if self._train_option is not None:
            if self._train_option._is_goal_option:
                # goal_abs_state = list(self._train_option.initiation_set)[0]
                # if self._train_option.is_termination_set(self._state):
                if self.is_goal_state(self._state):
                    reward += goal_reward 
                    reward += pseudo_reward
                    done = True
                    success = True
                # elif not abstract._tree.is_child_of_state(goal_abs_state, abstract.state(self._state)): # self._train_option.is_initiation_set(abstract.state(self._state)):
                # elif abstract.init_goal_abs_state is not None and not abstract._tree.is_child_of_state(abstract.init_goal_abs_state, abstract.state(self._state)):
                #     reward += -5
                #     done = False
                #     success = False
                else:
                    done = False
                    success = False
            else:
                if self._train_option.is_termination_set(abstract.state(self._state)):
                    reward += pseudo_reward
                    done = True
                    success = True
                else:
                    done = False
                    success = False
        
        return self._state, reward, done, {"success": success, "steps": self.steps}
        