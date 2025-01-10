import numpy as np
import random 
import copy
import itertools
import gym
import src.misc.map_maker as map_maker

class TaxiWorldContinuousEnvironment(gym.Env):

    def __init__(self, env_name, passenger_n = 1, include_goal_variable = False):
        super().__init__()
        self._maze = map_maker.get_map(env_name)
        self._dimension = self._maze.shape
        self._action_space = ['up','down','left','right','pickup','dropoff']
        self._action_size = len(self._action_space)
        self._action_probs = {0:[2,3], 1:[2,3], 2:[0,1], 3:[0,1], 4:[4,4], 5:[5,5]}
        self.in_taxi_loc_id = -4
        self.in_taxi_loc = [-2*self._dimension[0],-2*self._dimension[0]]
        self.id_to_special_locations = {self.in_taxi_loc_id: self.in_taxi_loc}
        self.id_to_special_locations[1] = [0, 0]
        self.id_to_special_locations[2] = [0, self._dimension[1]-1]
        self.id_to_special_locations[3] = [self._dimension[0]-1, 0]
        self.id_to_special_locations[4] = [self._dimension[0]-1, self._dimension[1]-1]
        self.special_location_to_id = {}
        for id, loc in self.id_to_special_locations.items():
            self.special_location_to_id[tuple(loc)] = id

        self.action_space = gym.spaces.Discrete(self._action_size)
        self.include_goal_variable = False
        self._train_option = None
        self._taxi_capacity = 1
        self._passenger_n = passenger_n # current code only works for 1 passenger currently
        self._state_ranges = self.get_state_ranges()
        self.observation_space = gym.spaces.Box(low=self._state_ranges[:,0], high = self._state_ranges[:,1], dtype = np.int32)
        self._stoch_prob = 0.8
        self._n_state_variables = len(self._state_ranges)
        self._vars_split_allowed = [1 for i in range(self._n_state_variables)]
        self._vars_split_allowed_initially = [1 for i in range(len(self._original_vars))] + [0 for i in range(self._n_state_variables - len(self._original_vars))]
        self.is_int_variable = [False, False, True, True]
        self.allowed_dx_dy = 0.6

    def get_state_ranges(self):
        # Taxi location 
        ranges = [[0, self._dimension[0]], [0, self._dimension[1]]]

        # Passenger locations where (-a,-a) represents in taxi
        for i in range(self._passenger_n): 
            # ranges += [[self.in_taxi_loc[0], self._dimension[0]], [self.in_taxi_loc[1], self._dimension[1]]]
            ranges += [[self.in_taxi_loc_id, len(self.id_to_special_locations)]]

        ranges += [[0,2]]

        self._original_vars = [i for i in range(0,len(ranges))]
        if self.include_goal_variable:
            # Destination location for passengers
            ranges += [[0, self._dimension[0]], [0, self._dimension[1]]]
            self._goal_vars = [len(ranges)-2, len(ranges)-1]
        return np.array(ranges)

    def initialize_problem(self, args, step_max):
        # handles init state for the main problem as well as subproblems, and goal state for the main problem
        self.problem_start_loc = args["init_vars"][:2]
        problem_pickup_loc_ids = args["init_vars"][2:2+self._passenger_n][0]
        self.problem_pickup_locs = self.id_to_special_locations[problem_pickup_loc_ids]
        problem_pickup_locs = self.get_closest_int_location(self.problem_pickup_locs)
        self.problem_in_taxi = 1 if tuple(problem_pickup_locs) == tuple(self.in_taxi_loc) else 0
        problem_goal_loc_ids = args["goal_vars"][0]
        self.problem_goal_locs = self.get_locs(problem_goal_loc_ids)
        self.reset()
        self._step_max = step_max

    def reset(self):
        taxi_loc = copy.deepcopy(self.problem_start_loc)
        self.passenger_locs = copy.deepcopy(self.problem_pickup_locs)
        self.in_taxi = copy.deepcopy(self.problem_in_taxi)
        self.dropoff_locs = copy.deepcopy(self.problem_goal_locs)
        self._init_state = self.get_updated_state(taxi_loc, self.passenger_locs, self.in_taxi)
        self._goal_state = self.get_updated_state(self.dropoff_locs, self.dropoff_locs, 0)
        # self._goal_range = [(item,item) for item in self._goal_state] 
        self._goal_range = self.get_goal_range()
        self._state = copy.deepcopy(self._init_state)
        self.steps = 0
        self.update_relevant_states()
        self._obj_locs = []
        for i in range(1,len(self.passenger_locs),1):
            loc = [self.passenger_locs[i-1],self.passenger_locs[i]]
            self._obj_locs.append(loc)
        # print(self._state, self._goal_range, self.dropoff_locs)
        return self._state
    
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

    def get_loc_ids(self, passenger_loc):
        # loc_ids = []
        # for passenger_loc in passenger_locs:
        #     loc_ids.append(self.special_location_to_id[passenger_loc])
        return self.special_location_to_id[tuple(passenger_loc)]
    
    def get_locs(self, loc_id):
        # loc_ids = []
        # for passenger_loc in passenger_locs:
        #     loc_ids.append(self.special_location_to_id[passenger_loc])
        return self.id_to_special_locations[loc_id]
    
    def get_updated_state(self, taxi_loc, passenger_locs, in_taxi):
        state = taxi_loc 
        state += [self.get_loc_ids(passenger_locs)]
        state += [in_taxi]
        # state += self.get_relation_vars(taxi_loc, passenger_locs, self.dropoff_locs)
        if self.include_goal_variable:
            state += self.dropoff_locs
        return state

    def step (self, action_index_input, abstract=None):
        self.steps += 1
        r_move = -1
        r_wrong_pickup, r_wrong_dropoff = -100, -100
        final_reward = 500
        dx, dy = self.allowed_dx_dy, self.allowed_dx_dy

        loc = self._state[:2]
        # self.update_locs_from_state(self._state)
        [a,b] = copy.deepcopy(loc)
        reward  = 0 # the episode's reward (-100 for pitfall, 0 for reaching the goal, and -1 otherwise)
        done = False # termination flag is true if the agent falls in a pitfall or reaches to the goal
        success = False
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
        elif action == 'pickup':
            next_loc = copy.deepcopy(loc)
        elif action == 'dropoff':
            next_loc = copy.deepcopy(loc)

        if action != 'pickup' and action != 'dropoff': 
            closest_next_loc = self.get_closest_int_location(next_loc)
            if self.in_bound (next_loc) == False or self._maze [closest_next_loc[0]] [closest_next_loc[1]] == 1: # not within bounds or walls
                next_loc = copy.deepcopy(loc)
            reward = r_move
        elif action == 'pickup':
            if self.there_is_passenger_and_not_at_dropoff_loc_id(loc) and self.taxi_is_not_full():
                self.update_passenger_location (loc, action)
            else:
                reward = r_wrong_pickup
        elif action == 'dropoff':
            if self.there_is_passenger_in_taxi_and_at_dropoff_loc_id(loc):
                self.update_passenger_location (loc, action)
                if self._train_option is None and self.all_passengers_at_destination():
                    reward = final_reward
                    done, success = True, True
            else:
                reward = r_wrong_dropoff

        # state = next_loc + self.passenger_locs
        # if self.include_goal_variable:
        #     state += self._goal_vars
        # if self.in_taxi:
        #     self.passenger_locs = self.state[:2]
        self._state = self.get_updated_state(next_loc, self.passenger_locs, self.in_taxi)
        # state += [self.in_taxi]
        if self.steps >= self._step_max:
            done = True 

        return self._state, reward, done, {"success":success, "steps":self.steps}

    def action_stochastic(self, action_index):
        if random.uniform(0,1) > self._stoch_prob:
            if random.uniform (0,1) > 0.5 : 
                action_index_stoch = self._action_probs[action_index][0]
            else: 
                action_index_stoch = self._action_probs[action_index][1]
        else: 
            action_index_stoch = action_index
        return action_index_stoch

    def update_passenger_location (self, current_location, action):
        if action == 'pickup':
            temp = []
            for i in range(1,len(self.passenger_locs),1):
                pass_loc = (self.passenger_locs[i-1],self.passenger_locs[i])
                dropoff_loc = (self.dropoff_locs[i-1],self.dropoff_locs[i])
                pass_loc = self.get_closest_int_location(pass_loc)
                dropoff_loc = self.get_closest_int_location(dropoff_loc)
                current_location = self.get_closest_int_location(current_location)
                if tuple(current_location) == tuple(pass_loc) and tuple(current_location) != tuple(dropoff_loc):
                    temp.extend(self.in_taxi_loc)
                    # temp.extend(current_location)
                    # self.in_taxi = 1
                else:
                    temp.extend(pass_loc)
            self.passenger_locs = temp

        if action == 'dropoff':
            temp = []
            for i in range(1,len(self.passenger_locs),1):
                pass_loc = (self.passenger_locs[i-1],self.passenger_locs[i])
                dropoff_loc = (self.dropoff_locs[i-1],self.dropoff_locs[i])
                pass_loc = self.get_closest_int_location(pass_loc)
                dropoff_loc = self.get_closest_int_location(dropoff_loc)
                current_location = self.get_closest_int_location(current_location)
                if tuple(pass_loc) == tuple(self.in_taxi_loc) and tuple(current_location) == tuple(dropoff_loc): 
                    temp.extend (dropoff_loc)
                    # self.in_taxi = 0
                else: 
                    temp.extend (pass_loc)
            self.passenger_locs = temp
        self.in_taxi = 1 if tuple(self.passenger_locs) == tuple(self.in_taxi_loc) else 0

    def get_closest_int_location(self, location):
        return [int(round(location[0],2)), int(round(location[1],2))]

    def there_is_passenger_and_not_at_dropoff_loc_id(self, current_location):
        for i in range(1,len(self.passenger_locs),1):
            pass_loc = (self.passenger_locs[i-1],self.passenger_locs[i])
            dropoff_loc = (self.dropoff_locs[i-1],self.dropoff_locs[i])
            pass_loc = self.get_closest_int_location(pass_loc)
            dropoff_loc = self.get_closest_int_location(dropoff_loc)
            current_location = self.get_closest_int_location(current_location)
            if tuple(current_location) == tuple(pass_loc) and tuple(current_location) != tuple(dropoff_loc): 
                return True
        return False
    
    def there_is_passenger_in_taxi_and_at_dropoff_loc_id(self, current_location):
        for i in range(1,len(self.passenger_locs),1):
            pass_loc = (self.passenger_locs[i-1],self.passenger_locs[i])
            dropoff_loc = (self.dropoff_locs[i-1],self.dropoff_locs[i])
            pass_loc = self.get_closest_int_location(pass_loc)
            dropoff_loc = self.get_closest_int_location(dropoff_loc)
            current_location = self.get_closest_int_location(current_location)
            if tuple(pass_loc) == tuple(self.in_taxi_loc) and tuple(current_location) == tuple(dropoff_loc): 
                return True
        return False
    
    def taxi_is_not_full(self):
        count = 0
        for i in range(1,len(self.passenger_locs),1):
            pass_loc = (self.passenger_locs[i-1],self.passenger_locs[i])
            pass_loc = self.get_closest_int_location(pass_loc)
            if tuple(pass_loc) == tuple(self.in_taxi_loc):
                count += 1
        if count >= self._taxi_capacity: return False
        else: return True

    def any_passenger_in_taxi(self):
        for pass_loc in self.passenger_locs:
            pass_loc = self.get_closest_int_location(pass_loc)
            if tuple(pass_loc) == tuple(self.in_taxi_loc): 
                return True
        return False

    def all_passengers_at_destination (self):
        for i in range(0,len(self.passenger_locs)-1,1): 
            pass_loc = (self.passenger_locs[i], self.passenger_locs[i+1])
            dropoff_loc = (self.dropoff_locs[i], self.dropoff_locs[i+1])
            pass_loc = self.get_closest_int_location(pass_loc)
            dropoff_loc = self.get_closest_int_location(dropoff_loc)
            if tuple(pass_loc) != tuple(dropoff_loc): 
                return False
        return True

    def in_bound (self, loc):
        flag = False
        if loc[0] < self._dimension[0] and loc[0] >= 0:
            if loc[1] < self._dimension[1] and loc[1] >= 0:
                flag = True
        return flag

    def index_to_action (self, action_index):
        return self._action_space [action_index]
    
    def state_to_index (self, state):
        return tuple(state)

    def update_relevant_states(self): # for heatmap
        self._relevant_states = set()
        # passenger at pickup loc
        _state = self._init_state[:2] + self.problem_pickup_locs 
        _state += [0] 
        # _state += [0, 0, 0]
        if self.include_goal_variable:
            _state += self.dropoff_locs
        self._relevant_states.add(tuple(_state))

        # taxi at pickup loc
        _state = self.problem_pickup_locs + self.problem_pickup_locs 
        _state += [0] 
        # _state += [1, 0, 0]
        if self.include_goal_variable:
            _state += self.dropoff_locs
        self._relevant_states.add(tuple(_state))

        # passenger in taxi, taxi at pickup loc
        _state = self._init_state[2:4] + self.in_taxi_loc
        _state += [1] 
        # _state += [0, 0, 0]
        if self.include_goal_variable:
            _state += self.dropoff_locs
        self._relevant_states.add(tuple(_state))

        # taxi at dropoff loc, passenger in taxi
        _state = self.dropoff_locs + self.in_taxi_loc 
        _state += [1] 
        # _state += [0, 0, 1]
        if self.include_goal_variable:
            _state += self.dropoff_locs
        self._relevant_states.add(tuple(_state))

        # taxi at dropoff loc, passenger at dropoff loc
        _state = self.dropoff_locs + self.dropoff_locs 
        _state += [0] 
        # _state += [0, 1, 1]
        if self.include_goal_variable:
            _state += self.dropoff_locs
        self._relevant_states.add(tuple(_state))

        self._relevant_states.add(tuple(self._goal_state))

    def seed(self, seed):
        # self.seed = seed
        pass

class TaxiWorldContinuous(TaxiWorldContinuousEnvironment):
    def __init__(self, env_name, passenger_n):
        super().__init__(env_name, passenger_n=passenger_n)

        self.include_goal_variable = False
        self._obj_locs = []
        self._relevant_states = set()
        self._train_option = None
        self.problem_start_loc = None
        self.problem_pickup_locs = None
        self.problem_goal_locs = None

    def get_state_ranges(self):
        # Taxi location 
        ranges = [(0, self._dimension[0]), (0, self._dimension[1])]

        # Passenger locations where (-a,-a) represents in taxi
        for i in range(self._passenger_n): 
            # ranges += [(self.in_taxi_loc[0], self._dimension[0]), (self.in_taxi_loc[1], self._dimension[1])]
            ranges += [[self.in_taxi_loc_id, len(self.id_to_special_locations)]]


        # Passenger in taxi
        ranges += [[0, 2]]

        self._original_vars = [i for i in range(0,len(ranges))]

        # add relations between variables
        # for i in range(3):
        #     ranges += [(0, 2)]

        # Destination location for passengers
        if self.include_goal_variable:
            ranges += [(0, self._dimension[0]), (0, self._dimension[1])]
            self._goal_vars = [len(ranges)-2, len(ranges)-1]
        else:
            self._goal_vars = []

        return np.array(ranges)

    def reset(self):
        taxi_loc = copy.deepcopy(self.problem_start_loc)
        self.passenger_locs = copy.deepcopy(self.problem_pickup_locs)
        self.in_taxi = copy.deepcopy(self.problem_in_taxi)
        self.dropoff_locs = copy.deepcopy(self.problem_goal_locs)
        self._init_state = self.get_updated_state(taxi_loc, self.passenger_locs, self.in_taxi)
        self._goal_state = self.get_updated_state(self.dropoff_locs, self.dropoff_locs, 0)
        # self._goal_range = [(item,item) for item in self._goal_state]
        self._goal_range = self.get_goal_range()
        self._state = copy.deepcopy(self._init_state)
        self.steps = 0
        self.update_relevant_states()
        self._obj_locs = []
        for i in range(1,len(self.passenger_locs),1):
            loc = [self.passenger_locs[i-1],self.passenger_locs[i]]
            self._obj_locs.append(loc)
        # print(self._state, self._goal_state, self.dropoff_locs)
        return self._state

    def get_updated_state(self, taxi_loc, passenger_locs, in_taxi):
        state = list(taxi_loc) 
        state += [self.get_loc_ids(passenger_locs)]
        state += [in_taxi]
        # state += self.get_relation_vars(taxi_loc, passenger_locs, self.dropoff_locs)
        if self.include_goal_variable:
            state += self.dropoff_locs
        return state

    # def get_relation_vars(self, taxi_loc, passenger_locs, dropoff_locs):
    #     relation_vars = []
    #     relation_vars += [1] if tuple(taxi_loc) == tuple(passenger_locs) else [0]
    #     relation_vars += [1] if tuple(passenger_locs) == tuple(dropoff_locs) else [0]
    #     relation_vars += [1] if tuple(taxi_loc) == tuple(dropoff_locs) else [0] 
    #     return relation_vars
                
    def step (self, action_index_input, abstract=None):
        pseudo_reward = 500
        state, reward, done, info = super().step(action_index_input)
        success = info['success']
        self._state = self.get_updated_state(state[:2], self.passenger_locs, self.in_taxi)
        # print(self._state, action_index_input, self._goal_state)
        if self._train_option is not None:
            done = False
            success = False
            if self._train_option._is_goal_option:
                if self.is_goal_state(self._state):
                    reward = pseudo_reward
                    done = True
                    success = True
            else:
                if self._train_option.is_termination_set(abstract.state(self._state)):
                    reward = pseudo_reward
                    done = True
                    success = True
        return self._state, reward, done, {"success": success, "steps": self.steps}

    # def get_goal_abstract_states(self, abstract_states):
    #     goal_abs_states = set()
    #     for abs_state in abstract_states:
    #         temp_abs_state = [[np.float32(item2) for item2 in item.split(",")] for item in abs_state]
    #         goal = True
    #         for i in range(len(temp_abs_state)):
    #             low1, high1 = temp_abs_state[i][0], temp_abs_state[i][1]
    #             low2, high2 = self._goal_range[i][0], self._goal_range[i][1]
    #             if (low1 < low2 and high1 <= low2) or (low2 < low1 and high2 <= low1):
    #                 goal = False
    #                 break
    #         if goal:
    #             goal_abs_states.add(abs_state)
    #     return goal_abs_states

    # def check_validity_abstract_state(self, abs_state):
    #     valid = True
    #     # print(abs_state)
    #     for var_i, val_var_i in self.var_dict.items():
    #         boolean_val1 = [int(i)-1 for i in abs_state[var_i].split(",")]
    #         if len(boolean_val1) !=2:
    #             boolean_val1 = boolean_val1[0]
    #             boolean_val2 = True
    #             for key,val in val_var_i.items():
    #                 var1 = [int(i) for i in abs_state[key].split(",")]
    #                 # var2 = [int(i) for i in abs_state[val].split(",")]
    #                 low1, up1 = var1[0], var1[1]
    #                 # low2, up2 = var2[0], var2[1]
    #                 goal_X, goal_y = self._goal_state[-2], self._goal_state[-1]
    #                 # if (low1 < low2 and low2 < up1) or (low2 < low1 and low1 < up2):
    #                 #     boolean_val2 = True
    #                 # else:
    #                 #     boolean_val2 = False
    #                 #     break
    #                 if (low1 <= goal_X < up1) and (low2 <= goal_y < up2):
    #                     boolean_val2 = True
    #                 else:
    #                     boolean_val2 = False
    #                     break
    #             if boolean_val1 != boolean_val2:
    #                 valid = False
    #                 break
    #     return valid

    def is_goal_state(self, state):
        for i in range(len(state)):
            val = state[i]
            if not (val >= self._goal_range[i][0] and val <= self._goal_range[i][1]*1.001):
                return False
        return True

    def update_relevant_states(self): # for heatmap
        self._relevant_states = set()
        # passenger at pickup loc
        _state = self._init_state[:2] + [self.get_loc_ids(self.problem_pickup_locs)]
        _state += [0] 
        # _state += [0, 0, 0]
        # _state += self.dropoff_locs
        self._relevant_states.add(tuple(_state))

        # taxi at pickup loc
        _state = self.problem_pickup_locs + [self.get_loc_ids(self.problem_pickup_locs)]
        _state += [0] 
        # _state += [1, 0, 0]
        # _state += self.dropoff_locs
        self._relevant_states.add(tuple(_state))

        # passenger in taxi, taxi at pickup loc
        _state = self._init_state[2:4] + [self.get_loc_ids(self.in_taxi_loc)]
        _state += [1] 
        # _state += [0, 0, 0]
        # _state += self.dropoff_locs
        self._relevant_states.add(tuple(_state))

        # taxi at dropoff loc, passenger in taxi
        _state = self.dropoff_locs + [self.get_loc_ids(self.in_taxi_loc)]
        _state += [1] 
        # _state += [0, 0, 1]
        # _state += self.dropoff_locs
        self._relevant_states.add(tuple(_state))

        # taxi at dropoff loc, passenger at dropoff loc
        _state = self.dropoff_locs + [self.get_loc_ids(self.dropoff_locs)]
        _state += [0] 
        # _state += [0, 1, 1]
        # _state += self.dropoff_locs
        self._relevant_states.add(tuple(_state))

        self._relevant_states.add(tuple(self._goal_state))

