import numpy as np
import random 
import copy
import gym


class CraftWorldObject:
    AGENT = "A"
    WOOD = "a"
    TOOLSHED = "b"
    WORKBENCH = "c"
    GRASS = "d"
    FACTORY = "e"
    IRON = "f"
    BRIDGE = "g"
    AXE = "h"
    WALL = "X"


class MineCraftEnvironment(gym.Env):
    def __init__(self, env_name, task, include_goal_variable=False):
        super().__init__()
        self._visited = []
        self._action_space = ['up','down','left','right']
        self._maze = self.load_map(env_name)
        self._dimension = self._maze.shape
        self._task = task


        self._action_size = len(self._action_space)
        self.action_space = gym.spaces.Discrete(self._action_size)
        self._action_probs = {0:[2,3], 1:[2,3], 2:[0,1], 3:[0,1]}
        self._stoch_prob = 0.8
        self._visit_map = np.zeros_like(self._maze)
        self.action_len = 0.45
        self.proximity_distance = 1
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
        self.is_int_variable = [False, False, False, False, True, True, True, True, True]
    
    def initialize_problem(self, args, step_max):
        start, goal = args["init_vars"][:2], args["goal_vars"]
        self._goal = goal
        assert not self._maze[goal[0], goal[1]], "invalid goal location"

        if self._task=="makeIronAxe" or self._task=="makeStoneAxe":
            self.woodPosition = args["init_vars"][2:4]
            self.workbenchPosition = args["locs"][0:2] 
            self.ironPosition = args["locs"][2:4] 
            self.stonePosition = args["locs"][4:6] 
            self.toolshedPosition = args["locs"][6:8] 
            if len(args["init_vars"]) > 6:
                self.problemHasWood = args["init_vars"][4]
                self.problemUsedWorkbench = args["init_vars"][5]
                self.problemGotIron = args["init_vars"][6]
                self.problemGotStone = args["init_vars"][7]
                self.problemUsedToolshed = args["init_vars"][8]
            else:
                self.problemHasWood = 0
                self.problemUsedWorkbench = 0
                self.problemGotIron = 0
                self.problemGotStone = 0
                self.problemUsedToolshed = 0
        
        self._init_state = copy.deepcopy(start) + self.woodPosition + [self.problemHasWood, self.problemUsedWorkbench, self.problemGotIron, self.problemGotStone, self.problemUsedToolshed]
        if self._task == "makeIronAxe":
            self._goal_state = copy.deepcopy(goal) + self.woodPosition + [1,1,1,0,1]
        elif self._task == "makeStoneAxe":
            self._goal_state = copy.deepcopy(goal) + self.woodPosition + [1,1,0,1,1]

        if self.include_goal_variable:
            self._init_state = self._init_state + copy.deepcopy(goal)
            self._goal_state = self._goal_state + copy.deepcopy(goal)

        self._goal_range = [(0,self._dimension[0])] + [(0,self._dimension[1])] + [(item,item) for item in self._goal_state][2:]
        # self._state = copy.deepcopy(self._init_state)
        self.reset()
        # print(self._state)
        self._relevant_states = self.get_relevant_states()
        self._step_max = step_max


    def get_state_ranges(self):
        ranges = [ [0,self._dimension[0]], # robot y
                   [0,self._dimension[1]], # robot x
                 ]

        ranges += [ [0,self._dimension[0]], # wood y
                [0,self._dimension[1]], # wood x
                ]
        
        if self._task=="makeIronAxe" or self._task=="makeStoneAxe":
            ranges+=[[0,2], [0,2], [0,2], [0,2], [0,2]]
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

    def distance(self, a, b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5

    def reset(self):
        self.steps = 0
        self.hasWood = copy.deepcopy(self.problemHasWood)
        self.usedWorkbench = copy.deepcopy(self.problemUsedWorkbench)
        self.gotIron = copy.deepcopy(self.problemGotIron)
        self.gotStone = copy.deepcopy(self.problemGotStone)
        self.usedToolshed = copy.deepcopy(self.problemUsedToolshed)
        self._state = copy.deepcopy(self._init_state)
        assert not self._maze[int(self._state[0]), int(self._state[1])], "invalid agent location"
        # print(self._state)
        return self._state


    def step(self, action_index_input, abstract=None):
        goal_reward = 1000
        pseudo_reward = 500
        self.steps += 1
        [a,b] = self._state[:2]
        reward = -1 # the episode's reward (-100 for pitfall, 0 for reaching the goal, and -1 otherwise)
        done = False # termination flag is true if the agent falls in a pitfall or reaches to the goal
        success = False
        action_index = self.action_stochastic (action_index_input)
        action = self.index_to_action (action_index)
        if action == 'up':
            next_loc = [a-self.action_len,b]
        elif action == 'down':
            next_loc = [a+self.action_len,b]
        elif action == 'left':
            next_loc = [a, b-self.action_len] 
        elif action == 'right':
            next_loc = [a, b+self.action_len] 

        if (self.in_bound(next_loc) and self._maze[int(next_loc[0]), int(next_loc[1])] != 1):
                    
            if not self.hasWood:
                if self.distance(next_loc,self.woodPosition)<self.proximity_distance:
                    self.hasWood = 1
            elif self.hasWood and not self.usedWorkbench:
                if self.distance(next_loc, self.workbenchPosition)<self.proximity_distance:
                    self.usedWorkbench = 1
            elif self._task=="makeIronAxe":
                if self.hasWood and self.usedWorkbench and not self.gotIron:
                    if self.distance(next_loc, self.ironPosition)<self.proximity_distance:
                        self.gotIron = 1
                elif self.hasWood and self.usedWorkbench and self.gotIron and not self.usedToolshed:
                    if self.distance(next_loc, self.toolshedPosition)<self.proximity_distance:
                        self.usedToolshed = 1    
                        if self._train_option is None:
                            reward = goal_reward
                            done = True
                            success  = True
            elif self._task=="makeStoneAxe":
                # print(self._state)
                if self.hasWood and self.usedWorkbench and not self.gotStone:
                    if self.distance(next_loc, self.stonePosition)<self.proximity_distance:
                        self.gotStone = 1
                elif self.hasWood and self.usedWorkbench and self.gotStone and not self.usedToolshed:
                    if self.distance(next_loc, self.toolshedPosition)<self.proximity_distance:
                        self.usedToolshed = 1    
                        if self._train_option is None: 
                            reward = goal_reward
                            done = True
                            success  = True
        else:
            reward = -5  # A negative reward for hitting wall
            next_loc = self._state[:2]

        self._state = self.get_updated_state(next_loc, self.hasWood, self.usedWorkbench, self.gotIron, self.gotStone, self.usedToolshed)

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

        # print(self._state)
        return self._state, reward, done, {"success": success, "steps": self.steps}


    def get_updated_state(self, next_loc, hasWood, usedWorkBench, gotIron, gotStone, usedToolshed):
        state = next_loc + self.woodPosition + [hasWood] + [usedWorkBench] + [gotIron] + [gotStone] + [usedToolshed]
        if self.include_goal_variable:
            state += self._goal
        # state += self.get_relation_vars(taxi_loc, passenger_locs, self.dropoff_locs)
        # if self.include_goal_variable:
        #     state += self.
        return state

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
        # self.seed = seed
        pass

    def load_map(self, map_name):
        map_array, start = [], []
        with open(f"environments/maps/{map_name}.map", 'r') as file:
            for line in file:
                row = [0 if char == ' ' else 1 for char in line.rstrip('\n')]
                map_array.append(row)

        return np.array(map_array, dtype=np.int32)
    
    def is_goal_state(self, state):
        for i in range(len(state)):
            val = state[i]
            if not (val >= self._goal_range[i][0] and val <= self._goal_range[i][1]):
                return False
        return True

class MineCraft(MineCraftEnvironment):

    def __init__(self, env_name, task, include_goal_variable=False):
        super().__init__(env_name, task, include_goal_variable)

