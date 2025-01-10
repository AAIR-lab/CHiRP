import numpy as np
from copy import copy, deepcopy
import math
import itertools
from sklearn.cluster import KMeans
import networkx as nx
import os
from functools import reduce
from operator import mul

from src.agent.learning import *
from src.data_structures.cat import CAT
from src.misc import utils
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in scalar divide')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Degrees of freedom <= 0')
np.seterr(divide='ignore')

class Abstraction:
    def __init__(self, env, agent_con, agent, k_cap, init_abs_level=1, bootstrap = 'from_init', continuous_state=False, refinement_method="deliberative"):
        self._env = env
        self.is_int_variable = self._env.is_int_variable
        if continuous_state == False:
            self._maze_abstract = np.chararray(self._env._maze.shape, itemsize = 100 ,unicode=True)
            self._maze_abstract[:] = ''
        self._init_abs_level = init_abs_level
        self._n_states = env._n_state_variables
        self._state_ranges = env._state_ranges
        self._vars_split_allowed = env._vars_split_allowed
        self._vars_split_allowed_initially = env._vars_split_allowed_initially
        self._n_abstract_states = 0
        self._n_action_size = env._action_size
        self._agent_concrete = agent_con
        self._agent = agent
        self._local_traj = {}
        self._local_stg = {}
        self._stgs = {}
        self._abstract_colors = dict()
        self._abstract_best_actions = dict()
        self.continuous_state = continuous_state
        self._tree = CAT(self._state_ranges, self._vars_split_allowed_initially, self._env.is_int_variable, continuous_state)
        self.update_n_abstract_state()
        self._gen_pool = []
        self._current_counter_examples = []
        self._k_initial = k_cap
        self._k = deepcopy(k_cap)
        self._unstable_cap = []
        self._bootstrap = bootstrap #'from_concrete' #'from_ancestor' #  #'from_init'      
        self._buffer = {}
        self.refinement_method = refinement_method
        self.init_goal_abs_state = None
      
    def split_to_all_state_values (self, split):
        state_values = []
        for k in range(len(split)):
            sp = split[k]
            temp = []
            for i in range (len(sp)-1):
                temp.append(str (sp[i]) + ',' + str(sp[i + 1]))
            start_end = temp[-1].split(",")
            if len(start_end)>0 and np.float32(start_end[0]) == np.float32(start_end[1]): # need to exclude the invalid values
                temp = temp[:-1]
            state_values.append(temp)
        return state_values

    def reuse_tree(self, reuse_cat_path=None, directory_path=None, regenerate_tree=False, plot_cat=True):
        # reads old CAT from dot into abstract tree data structure, 
        # then builds a CAT for new problem using same partitioning
        if regenerate_tree:
            reuse_tree = utils.reuse_tree(reuse_cat_path, self.is_int_variable, self.continuous_state) # need to regenerate too right now, can't use this standalone as need to test
            self.regenerate_tree(reuse_tree, directory_path, plot_cat)
        else:
            with open(reuse_cat_path, 'rb') as f:
                self._tree = pickle.load(f)

    def initialize_tree(self):
        root = self._tree._root
        if self._init_abs_level == 1:
            new_state_values = self.split_to_all_state_values(root._split)
            new_leaf_nodes = list(itertools.product(*new_state_values))
            for s in new_leaf_nodes:
                new_node = self._tree.add_node (split = [], abs_state = s)
                new_node._parent = self._tree._root
                self._tree._root._child.append(new_node)
                self._tree._leaves[s] = new_node
            for leaf in self._tree._leaves:
                if self._tree._leaves[leaf]._is_root:
                    del self._tree._leaves[leaf] # the no longer is a leaf node
                    break
        elif self._init_abs_level == 0:
            new_leaf_nodes = root
        
    def add_child_nodes_using_vector(self, unstable_state, vector):
        node = self._tree.find_node (unstable_state)
        split = node._parent._split
        
        if 1 in vector:
            new_split, new_state_values = self._tree.update_split(unstable_state, split, vector)
            node._split = new_split # the node now has a different split compared to its parent 
            del self._tree._leaves[unstable_state] # the no longer is a leaf node
            new_leaf_states = list(itertools.product(*new_state_values))
            new_leaf_nodes = []
            for s in new_leaf_states:
                new_node = self._tree.add_node (split = [], abs_state = s)
                new_node._parent = node
                node._child.append(new_node)
                self._tree._leaves[s] = new_node 
                new_leaf_nodes.append(new_node)
            return new_leaf_nodes
        return []
    
    def regenerate_tree(self, reuse_tree, directory_path, plot_cat=True):
        root = self._tree._root
        new_state_values = self.split_to_all_state_values(root._split)
        new_leaf_states = list(itertools.product(*new_state_values))
        new_children = []
        for s in new_leaf_states:
            new_node = self._tree.add_node (split = [], abs_state = s)
            new_node._parent = self._tree._root
            self._tree._root._child.append(new_node)
            self._tree._leaves[s] = new_node
            new_children.append(new_node)
        for leaf in self._tree._leaves:
            if self._tree._leaves[leaf]._is_root:
                del self._tree._leaves[leaf] # it is no longer a leaf node
                break
            
        node_mapping = {}
        stack = []
        node_mapping[reuse_tree._root] = root 
        children = reuse_tree._root._child
        for i in range(len(children)):
            node_mapping[children[i]] = new_children[i]
            stack.append(children[i])
        visited = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.append(node)
            
            children = node._child
            if len(children) > 0:
                child = children[0]
                vector = list()
                for i in range(len(child._state)):
                    if node._state[i] != child._state[i]:
                        vector.append(1)
                    else:
                        vector.append(0)
    
                new_node = node_mapping[node]
                new_children = self.add_child_nodes_using_vector(new_node._state, vector)
                for i in range(len(new_children)):
                    node_mapping[children[i]] = new_children[i]
                    if children[i] not in visited:
                        stack.append(children[i])
                            
        if plot_cat:
            self.plot_cat(directory_path)

        self.state_mapping = {}
        for state, abs_state in node_mapping.items():
            self.state_mapping[state._state] = abs_state._state
    
    def state(self, state_con):
        state_abstract = self._tree.state(state_con) 
        assert state_abstract in self._tree._leaves
        return state_abstract 

    def split_abs_state_wrs (self, abs_state, wrt_variable_index):
        abs_state_1 = list(abs_state)
        abs_state_2 = list(abs_state)
        state_value = abs_state[wrt_variable_index]
        
        interval = state_value.split(",")
        for i in range(2):
            interval[i] = int(interval[i])
        midpoint = int((interval[1] - interval[0])/2) + interval[0] 
        interval1 = str(interval[0]) + "," + str(midpoint)
        interval2 = str(midpoint) + "," + str(interval[1])  
        abs_state_1[wrt_variable_index] = interval1
        abs_state_2[wrt_variable_index] = interval2
        return [(*abs_state_1, ), (*abs_state_2, )]

    def qtable_variation(self, abs_state, wrt_variable_index, unstable_action=None):
        unstable_state_expanded = self.split_abs_state_wrs(abs_state, wrt_variable_index)
        q_values = []

        if unstable_action is None:
            for item in unstable_state_expanded: 
                q_values.append(np.average(self.bootstrap(item)))
        else:
            for item in unstable_state_expanded: 
                qvalue = self.bootstrap(item)[unstable_action]
                q_values.append(qvalue)

        norm = np.linalg.norm(q_values)
        q_values = q_values/norm
        return np.std(q_values)
    
    def choose_vars_1(self, vars, k, unstable_state, unstable_action):
        interval = [np.float32(item) for item in unstable_state[k].split(",")]
        if self._tree.is_refinable(interval, k):
            variation = self.qtable_variation(unstable_state, k, unstable_action)
            vars.append(variation)
        else: 
            vars.append(0) 
        return vars

    def choose_vars_2(self, vars, k, abs_to_con, unstable_state):
        if unstable_state not in abs_to_con:
            interval = [np.float32(item) for item in unstable_state[k].split(",")]
            if self._tree.is_refinable(interval, k): #TODO
                vars.append(1)
            else: 
                vars.append(0) 
        else:
            variance = np.std(np.array(list(abs_to_con[unstable_state])), axis = 0)
            interval = [np.float32(item) for item in unstable_state[k].split(",")]
            if self._tree.is_refinable(interval, k):
                vars.append(variance[int(k/2)])
            else: 
                vars.append(0) 
        return vars

    @ignore_warnings(category=ConvergenceWarning)
    def get_to_split_variables(self, abs_to_con, unstable_state, unstable_action=None):
        vars = []
        for k in range (len(self._vars_split_allowed)):
            if self._vars_split_allowed[k]:
                if not self.continuous_state:
                    vars = self.choose_vars_1(vars, k, unstable_state, unstable_action)
                else:
                    vars = self.choose_vars_2(vars, k, abs_to_con, unstable_state)
            else:
                vars.append(0)
        vector = []
        for i in range (len(vars)):
            vector.append(0)

        if self._tree._root._state == unstable_state:
            n_clusters = 2
            X = np.array([[i] for i in vars])
            X.reshape(-1, 1)
            kmeans = KMeans(n_clusters=n_clusters, n_init="auto").fit(X)
            res = kmeans.predict(X)
                
        for i in range (len(vars)):
            if self._tree._root._state == unstable_state:
                if res[i] >= n_clusters-1 and vars[i] > 0:
                    vector[i] = 1
            else:
                if vars[i] > 0: 
                    vector[i] = 1
        return vector

    def get_to_split_variables_aggressive(self, unstable_state):
        vector = []
        for i in range (self._n_states):
            vector.append(1)
        return vector

    def bootstrap(self, state, concrete_states={}):
        # returns numpy array of qvalues
        if not self.continuous_state:
            if self._bootstrap == 'from_concrete':
                q_table_concrete_states = []
                for concrete_state in self._agent_concrete._qtable._qtable.keys():
                    if utils.fallsWithin_global(concrete_state, state, self._tree._root):
                        q_table_concrete_states.append(concrete_state)
                num_possible_concrete_states = self.possible_concrete_states(state)
                pulled = self._agent_concrete.pull_qvalues_single(q_table_concrete_states)
                pulled = ( (np.ones((1, self._env._action_size))[0] * self._agent_concrete._q_init * (num_possible_concrete_states-len(q_table_concrete_states))) + pulled)/ num_possible_concrete_states
        else:
            if self._bootstrap == "from_estimated_concrete":
                if len(concrete_states) == 0:
                    pulled = self._agent._qtable.get_np_init_qvalues(self._env._action_size, self._agent._q_init)
                else:
                    pulled = self._agent.pull_qvalues_single(concrete_states)
                    pulled = pulled / len(concrete_states)
        if self._bootstrap == 'from_ancestor':
            if state not in self._agent._qtable._qtable:
                pulled = self._agent._qtable.get_np_init_qvalues(self._env._action_size, self._agent._q_init)
            else:
                pulled = self._agent._qtable.get_qvalues(state)
        elif self._bootstrap == 'from_init':
            pulled = self._agent._qtable.get_np_init_qvalues(self._env._action_size, self._agent._q_init)
        return pulled

    def possible_concrete_states (self, state):
        possible_values = []
        for i in range(len(state)):
            s = state[i].split(",")
            temp = []
            for j in range (int(s[0]), int(s[1])):
                temp.append(j)
            possible_values.append(temp)
        return reduce(mul, [len(lst) for lst in possible_values], 1)

    def find_concrete_qtable_indices (self, concrete_states):
        indices = []
        for s in concrete_states:
            indices.append(self._env.state_to_index(s))
        return indices

    def update_n_abstract_state (self):
        self._n_abstract_states = len(self._tree._leaves)

    def update_abstraction(self, eval_log, abs_to_con, unstable_state=None, refine_levels=1):
        if unstable_state is None:
            eval = self.clean_eval(eval_log._table)
            if len(eval)>0:
                unstable_states, unstable_actions = self.find_k_unstable_state(eval)
                print("unstable states:",unstable_states)
                refine_levels_done = 0
                while len(unstable_states) > 0:
                    new_unstable_states = []
                    for i in range(len(unstable_states)):
                        s = unstable_states[i]
                        if refine_levels_done == 0:
                            a = unstable_actions[i]
                            new_states = self.update_tree(abs_to_con, s, a)
                        else:
                            new_states = self.update_tree(abs_to_con, s)
                        if i == 0 and refine_levels_done < refine_levels-1:
                            new_unstable_states.extend(new_states)
                    unstable_states = deepcopy(new_unstable_states)
                    refine_levels_done += 1
                self.update_n_abstract_state()
        else:
            self.update_tree(abs_to_con, unstable_state)
            self.update_n_abstract_state ()

    def is_divisible(self, state):
        valid = False
        node = self._tree.find_node(state)
        if node._is_root:
            return True
        split = node._parent._split
        indices = self._tree.state_to_split_indices(state, split)
        for i in range (len(self._vars_split_allowed)):
            if self._vars_split_allowed[i]:
                index = indices[i]
                lower = split [i][index]
                upper = split [i][index+1]
                if self._tree.is_refinable([lower, upper], i):
                    valid = True
                    break
        return valid

    def clean_eval(self, eval_in):
        # delete the states that are indivisible wrt to all the allowed split variables
        eval = deepcopy(eval_in)
        indivisible_states = []
        for state in eval:
            if not self.is_divisible(state): 
                indivisible_states.append(state)
        for s in indivisible_states:
            del eval[s]
        return eval

    def find_unstable_state (self, eval_log):
        max_value = -np.inf
        unstable_state = None
        for state in eval_log:
            std_temp = []
            for i in range (self._n_action_size):
                variation = pow(np.std(eval_log[state][i]),2) / np.average(eval_log[state][i])
                if np.isnan(variation): variation = 0
                std_temp.append(variation)
                max_current = max(std_temp)
            if max_current > max_value:
                max_value = max_current
                unstable_state = state
        self._current_counter_examples = [unstable_state]
        return [unstable_state]

    def get_minm_maxm_tderror(self, eval_log):
        all_values = []
        for state in eval_log:
            for action in eval_log[state]:
                all_values.extend(eval_log[state][action])
        if len(all_values) > 0:
            minm = min(all_values)
            maxm = max(all_values)
            return minm, maxm
        return None, None
    
    def normalize_eval(self, eval_log, minm, maxm):
        eval = deepcopy(eval_log)
        for state in eval_log:
            for action in eval_log[state]:
                eval[state][action] = []
                for eval_value in eval_log[state][action]: 
                    eval[state][action].append((eval_value - minm) / (maxm - minm))
        return eval

    def find_k_unstable_state (self, eval_log):
        var_dict = {}
        unstable_state_to_action = {}
        unstable_states = []

        minm, maxm = self.get_minm_maxm_tderror(eval_log)
        if minm is not None:
            eval_log = self.normalize_eval(eval_log, minm, maxm)

        for state in eval_log:
            std_temp = []
            for i in range (self._n_action_size):
                variation = pow(np.std(eval_log[state][i]),2) / abs(np.average(eval_log[state][i]))
                if np.isnan(variation): variation = 0
                std_temp.append(variation)
            max_current = max(std_temp)
            max_current_action = np.argmax(std_temp)
            var_dict [max_current] = state
            unstable_state_to_action[state] = max_current_action

        var_dict = dict(sorted(var_dict.items(),reverse=True))
        unstable_selected = list(var_dict.items())

        q = self.get_total_unstable_number(unstable_selected)
        self._unstable_cap.append(q)
        k = min (self._k, q)
        print("k_cap:", self._k, "refining:", k)
        if k==0:
            print("no unstable state selected")
        unstable_selected = unstable_selected[0:k]
        unstable_actions = list()
        for item in unstable_selected:
            unstable_states.append(item[1])
            unstable_actions.append(unstable_state_to_action[item[1]])
        self._current_counter_examples = unstable_states 
        return unstable_states, unstable_actions

    @ignore_warnings(category=ConvergenceWarning)
    def get_total_unstable_number(self, variation_values):
        if len(variation_values) > 0:
            X = []
            if variation_values[-1][0] < 1: base = 1 
            else: base = variation_values[-1][0]
            for i in range (len(variation_values)):
                temp = []
                item = variation_values[i][0]
                if item < 1: item  = 1
                v = int(item/base)
                temp.append(math.log(v,2))
                X.append(temp)
            X = np.array(X)
            X.reshape(-1, 1)
            kmeans = KMeans(n_clusters=min(len(variation_values),3), n_init="auto").fit(X)
            res = kmeans.predict(X)
            ref = res[0]
            num = 0
            for i in range(len(res)):
                if res[i] == ref: num += 1
            return num
        else:
            return 0

    def update_tree(self, abs_to_con, unstable_state, unstable_action=None, vector=None):
        if vector is None:
            if self.refinement_method == "aggressive":
                vector = self.get_to_split_variables_aggressive(unstable_state)
            elif self.refinement_method == "deliberative":
                vector = self.get_to_split_variables(abs_to_con, unstable_state, unstable_action)                

        temp = []
        node = self._tree.find_node (unstable_state)
        if node._parent is None: 
            split = self._state_ranges
            for i in range(len(split)): split[i] = list(split[i])
        else:
            split = node._parent._split
        
        new_leaf_nodes = []
        if 1 in vector:
            new_split, new_state_values = self._tree.update_split(unstable_state, split, vector)
            node._split = new_split # the node now has a different split compared to its parent 
            del self._tree._leaves[unstable_state] # the no longer is a leaf node
            new_leaf_nodes = list(itertools.product(*new_state_values))
            for s in new_leaf_nodes:
                if self._bootstrap == 'from_estimated_concrete':
                    self._agent.abs_to_con[s] = set()
                    if unstable_state in self._agent.abs_to_con:
                        possible_concrete_states = self._agent.abs_to_con[unstable_state]
                        for c in possible_concrete_states:
                            if utils.fallsWithin_global(c, s, self._tree._root):
                                self._agent.abs_to_con[s].add(c)
                self._agent.update_qtable(s)
                           
                new_node = self._tree.add_node (split = [], abs_state = s)
                new_node._parent = node
                node._child.append(new_node)
                self._tree._leaves[s] = new_node
                temp.append(s)
            if unstable_state in self._agent._qtable._qtable:
                self._agent._qtable.delete_state(unstable_state)
            self._gen_pool.append(temp)
        return new_leaf_nodes

    ############################ plotting STGs, CATs, heatmaps ##########################################
    def reuse_stgs(self, reuse_stgs_path, directory_path, state_mapping):
        f = open(reuse_stgs_path, "rb")
        self._stgs = pickle.load(f)
        self._stgs_nx = {}
        for abs_index,stg in self._stgs.items():
            graph = nx.from_dict_of_dicts(stg, create_using=nx.DiGraph)
            graph = nx.relabel_nodes(graph, state_mapping)
            for edge in graph.edges:
                graph.add_edge(edge[0], edge[1], capacity=200)
            self._stgs_nx[abs_index] = graph
            # nx.nx_pydot.write_dot(graph, directory_path+"/stg_"+str(abs_index)+".dot")

    def add_transition(self, stg, state_abs, action, new_state_abs):
        if state_abs not in stg:
            stg[state_abs] = {}
        if new_state_abs not in stg[state_abs]:
            stg[state_abs][new_state_abs] = {}
            stg[state_abs][new_state_abs]["label"] = []
        if action not in stg[state_abs][new_state_abs]["label"]:
            stg[state_abs][new_state_abs]["label"].append(action)
        return stg
    
    def update_stg(self, abs_index, state_abs, action, new_state_abs):
        if abs_index not in self._stgs:
            self._stgs[abs_index] = {}
        self.add_transition(self._stgs[abs_index], state_abs, action, new_state_abs)
    
    def update_local_stg(self, abs_index, trajectories):
        if abs_index not in self._local_stg:
            self._local_traj[abs_index] = []
            self._local_stg[abs_index] = {}
        self._local_traj[abs_index].extend(trajectories)
        for trajectory in trajectories:
            for transition in trajectory.trajectory:
                state_abs, action, option, next_state_abs = transition.state, transition.action, transition.option, transition.next_state
                self.add_transition(self._local_stg[abs_index], state_abs, action, next_state_abs)

    def compute_criticalscores_and_distances(self, abs_index):
        critical_scores = utils.compute_critical_frequency(self._local_traj[abs_index])
        distances = self._tree.compute_distances(self._local_stg[abs_index])
        return critical_scores, distances
    
    def plot_stgs(self, directory, abs_index, optionid=None):
        utils.plot_stgs(self._stgs, self._stgs[abs_index], directory, optionid)

    def plot_local_stg(self, directory, abs_index, optionid=None):
        critical_scores, distances = self.compute_criticalscores_and_distances(abs_index)
        graph, self._abstract_colors, self._abstract_best_actions = utils.plot_local_stg(self._local_stg[abs_index], directory, self.revise_qtable(), self._abstract_colors, self._abstract_best_actions, distances, critical_scores, optionid)
        return graph

    def revise_qtable(self):
        new_table = {}
        for key in self._tree._leaves:
            if key in self._agent._qtable._qtable:
                new_table[str(key)] = self._agent._qtable._qtable[key]
        return new_table

    def plot_cat(self, directory_path):
        file = open(directory_path+"/cat.pickle", 'wb')
        pickle.dump(self._tree, file)
        file.close()
        self._abstract_colors, self._abstract_best_actions = utils.plot_cat(self._tree, directory_path, self._local_stg, self.revise_qtable(), self._abstract_colors, self._abstract_best_actions)

    def get_all_mazes(self, env, maze_abstract):
        max_y = maze_abstract.shape[0] 
        max_x = maze_abstract.shape[1]
        mazes = []
        for relevant_state in env._relevant_states:
            value = relevant_state[2:]
            temp_maze = deepcopy(maze_abstract)
            for i in range (max_y):
                for j in range (max_x):
                    temp = [i,j]
                    temp += value
                    abs_state = self.state(temp)
                    temp_maze[i][j] = str(abs_state)
            mazes.append ([temp_maze, value])
        return mazes

    def plot_heatmaps(self, init_state, goal_state, heatmap_directory, index, qtable=None, optionid=None):
        if qtable is None:
            qtable = self.revise_qtable()

        if not os.path.exists(heatmap_directory+"/"):
            os.makedirs(heatmap_directory+"/")
        all_data = self.get_all_mazes(self._env, self._maze_abstract)
        if all_data != []:
            self._abstract_colors, self._abstract_best_actions = utils.plot_heatmaps(heatmap_directory, index, self._env, init_state, goal_state, all_data, qtable, self._abstract_colors, self._abstract_best_actions, optionid)
