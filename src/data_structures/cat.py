import networkx as nx
import numpy as np
import math
import copy

class CAT():

    class Abs_node():
        def __init__(self, split, abs_state, is_root=False):
            self._split = split
            self._state = abs_state
            self._parent = None
            self._child = []
            self._is_root = is_root

        def __hash__(self):
            return self._state.__hash__()
        
        def __eq__(self, other):
            return self._state.__hash__() == other._state.__hash__()
        
        def __lt__(self, other):
            return False
        
    def __init__(self, state_ranges, _vars_split_allowed_initially, is_int_variable, continuous_state=False):
        self.continuous_state = continuous_state
        self.is_int_variable = is_int_variable
        self._vars_split_allowed_initially = _vars_split_allowed_initially
        self._leaves = {}
        self._root_split, self._root_abs_state = self.initialize_root_node(state_ranges)
        self._root = self.add_node(self._root_split, self._root_abs_state, is_root=True)

    def initialize_root_node(self, state_ranges):
        split = []
        root_abs_state = []
        for i in range (len(state_ranges)):
            if not self.continuous_state:
                midpoint = (state_ranges[i][1] - state_ranges[i][0])/2 + state_ranges[i][0]
                if not midpoint - int(midpoint) == 0: midpoint = math.ceil(midpoint)
                midpoint = int(midpoint)
                min_value = state_ranges[i][0]
                max_value = state_ranges[i][1]
            else:
                midpoint = np.float32(round(state_ranges[i][0] + (state_ranges[i][1] - state_ranges[i][0])/2.0, 3)) #TODO: midpoint is rounded off to 3 decimals, check if needs to be changed
                min_value = np.float32(state_ranges[i][0])
                max_value = np.float32(state_ranges[i][1])
            if self._vars_split_allowed_initially[i] == 1:
                split.append ([min_value, midpoint, max_value])
            else:
                split.append ([min_value, max_value])
            root_abs_state.append(str(min_value) + "," + str(max_value))
        root_abs_state = tuple(root_abs_state)
        return split, root_abs_state
           
    def find_node (self, abs_state):
        found = None
        for leaf in self._leaves:
            if leaf == abs_state: 
                found = self._leaves[leaf]
                break
        return found

    def find_node_in_tree(self, discovered, root_node, abs_state):
        discovered.append(root_node)
        if root_node._state == abs_state:
            return True, root_node
        for child in root_node._child:
            if child not in discovered:
                found, node = self.find_node_in_tree(discovered, child, abs_state)
                if found:
                    return found, node
        return False, None

    def add_node (self, split, abs_state, is_root=False):
        node = self.Abs_node(split, abs_state, is_root)
        self._leaves[abs_state] = node
        return node
    
    def state_to_split_indices (self, state, split):
        # find out which split the state belongs to from a set of splits
        indices = []
        for i in range(len(state)): 
            s = state[i].split(",")
            s = [np.float32(s[0]), np.float32(s[1])]
            for j in range(len(split[i])-1):
                if np.allclose(s, split[i][j:j+2]):
                    indices.append(j)
                    break
        return indices 
    
    def is_refinable (self, interval, var_i):
        if not self.continuous_state or self.is_int_variable[var_i]:
            interval_allowed = 1
        else:
            interval_allowed =  0.2 #TODO: check if should change to something else
        if interval[1] - interval[0] > interval_allowed: return True
        else: return False
    
    def update_split(self, unstable_state, split_in, to_split_vector):
        split = copy.deepcopy(split_in)
        split = [list(x) for x in split]
        split_indices = self.state_to_split_indices(unstable_state, split)
        new_state_values = []
        for i in range(len(split_indices)):
            index = split_indices[i]
            # if we need to split the state variable
            if to_split_vector[i] == 1:
                if self.is_refinable( [split[i][index], split[i][index+1]], i): 
                    if not self.continuous_state:
                        new_split_point = (split[i][index+1] - split[i][index])/2 + split[i][index]
                        if not new_split_point - int(new_split_point) == 0: new_split_point = math.ceil(new_split_point)
                        new_split_point = int(new_split_point)
                    else:
                        new_split_point = np.float32(round(split[i][index] + (split[i][index+1] - split[i][index])/2.0, 3)) #TODO: midpoint is rounded off to 3 decimals, check if needs to be changed
                    split[i].append(new_split_point)
                    split[i].sort()
                    new_state_values.append([str(split[i][index]) + "," + str(new_split_point), str(new_split_point) + "," + str(split[i][index + 2]) ])
                else: new_state_values.append([str(split[i][index]) + "," + str(split[i][index + 1])])
            else:
                new_state_values.append([str(split[i][index]) + "," + str(split[i][index + 1])])
                split[i] = split_in[i]
        return split, new_state_values

    def state(self, state_con):
        state_abstract = self.state_recursive(state_con, self._root) 
        assert state_abstract in self._leaves
        return state_abstract 

    def states(self, state_con, variables):
        states_abstract = self.get_abstract_states(state_con, self._root, variables) 
        for state_abstract in states_abstract:
            assert state_abstract in self._leaves
        return states_abstract 

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
        if start_node._child == []: return self._root_abs_state
        if found: 
            return result
        else:
            if not flag:
                print (state_con)
            return self.state_recursive(state_con, temp_node)

    def fallsWithin(self, concrete_state, abstract_state, vars_i):
        concrete_state = np.array(concrete_state).take(vars_i)
        ranges = np.asarray(abstract_state).take(vars_i)
        a, b = [], []

        for item in ranges:
            item_list = item.split(",")
            a.append(np.float32(item_list[0]))
            b.append(np.float32(item_list[1]))

        if (concrete_state >= a).all() and (concrete_state < b).all():
            return True
        else:
            return False

    def get_abstract_states(self, state_con, variables):
        abstract_states = []
        for leaf in self._leaves:
            if self.fallsWithin(state_con, leaf, variables):
                abstract_states.append(leaf)
        return abstract_states

    def con_state_to_abs(self, state_con, split):
        state = []
        for i in range(len(state_con)):
            for j in range (len(split[i]) -1):
                if j == len(split[i])-2:
                    if state_con[i] >= np.float32(split[i][j]) and state_con[i] <= np.float32(split[i][j+1]):
                        state.append(str(split[i][j]) + ',' + str(split[i][j+1]) )
                        break
                else:
                    if state_con[i] >= np.float32(split[i][j]) and state_con[i] < np.float32(split[i][j+1]):
                        state.append(str(split[i][j]) + ',' + str(split[i][j+1]) )
                        break
        state = tuple(state)
        if len(state) == len(state_con): return state
        else: return None  
    
    def get_networkx_cat(self, root):
        graph = nx.DiGraph()
        stack = [root]
        while stack:
            temp = stack.pop()
            for child in temp._child:
                stack.append(child)
                node1 = temp._state
                node2 = child._state 
                graph.add_edge(node1, node2)
        return graph

    def distance_betn_abstract_states(self, state1, state2):
        if state1 == state2:
            return 0
        graph = self.get_networkx_cat(self._root)
       
        try:
            ancestor = nx.lowest_common_ancestor(graph, state1, state2)
            dist1 = nx.shortest_path_length(graph, source=ancestor, target=state1)
            dist2 = nx.shortest_path_length(graph, source=ancestor, target=state2)
        except Exception as e:
            print(str(e))
            print(state1, state2)
            print(list(nx.simple_cycles(graph)))
        dist = 0.9 * (self.max_depth - self.depths[ancestor] + 1) + (0.1 * (dist1+dist2)/2.0)
        return dist
    
    def distance_to_common_ancestor(self, state1, state2):
        if state1 == state2:
            return 0
        graph = self.get_networkx_cat(self._root)
       
        try:
            ancestor = nx.lowest_common_ancestor(graph, state1, state2)
            dist1 = nx.shortest_path_length(graph, source=ancestor, target=state1)
        except Exception as e:
            print(str(e))
            print(state1, state2)
            print(list(nx.simple_cycles(graph)))
        return dist1

    def compute_depths(self):
        self.depths = {}
        node = self._root
        self.depths[node._state] = 0
        queue = [node]
        self.max_depth = 0
        while len(queue) > 0:
            node = queue.pop(0)
            if node._state in self._leaves:
                node.is_leaf = True
            else:
                node.is_leaf = False
            for child in node._child:
                if child._state not in self.depths:
                    self.depths[child._state] = self.depths[node._state] + 1
                    queue.append(child)
                    if self.depths[child._state] > self.max_depth:
                        self.max_depth = self.depths[child._state]
    
    def compute_distances(self, local_stg):
        if len(local_stg) > 0:
            # self.compute_node_to_level()
            self.compute_depths()
        distances = dict()
        for state in local_stg:
            for next_state in local_stg[state]:
                distance = self.distance_betn_abstract_states(state, next_state)
                if state not in distances:
                    distances[state] = dict()
                if next_state not in distances[state]:
                    distances[state][next_state] = distance
        return distances
    
    def get_all_leafs(self, node):
        if len(node._child) == 0:
            return [node]
        leafs = []
        for child in node._child:
            leafs.extend(self.get_all_leafs(child))
        return leafs
    
    def get_all_leaf_siblings(self, node):
        parent = node._parent
        if parent is not None:
            children = parent._child
            leafs = []
            for child in children:
                leafs.extend(self.get_all_leafs(child))
            return leafs
        else:
            return []

    def is_child_of_state(self, intermediate_abs_state, abs_state):
        inter_node = self.find_node_in_tree([], self._root, intermediate_abs_state)[1]
        leafs = self.get_all_leafs(inter_node)
        for node in leafs:
            if node._state == abs_state:
                return True
        return False
            
