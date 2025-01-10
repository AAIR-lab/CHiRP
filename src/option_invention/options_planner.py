import heapq
import copy

from src.data_structures.option import Option

class OptionsPlanner():

    def __init__(self, options, cat, transition_library, invent_bridge_option=False, replanning=False):
        self.options = options
        self.cat = cat
        self.transition_library = transition_library
        self.augmented_abstract_transitions = {}
        self.invent_bridge_option = invent_bridge_option
        self.replanning = replanning
        self.depths = {}
        self.distances = {}
        self.heuristic = {}
        self.node_to_abs_node = {}
        self.cat.compute_depths()
        self.compute_depths()

    def search(self, init, goals):
        node = self.cat.find_node_in_tree([], self.cat._root, init)[1]
        self.frontier = []
        heapq.heappush(self.frontier, (0, [node, [(None, node)], 0]))
        self.visited = {}
        while len(self.frontier) > 0:
            node, plan, pathcost = heapq.heappop(self.frontier)[1]
            if node._state in goals:
                return plan
            if node not in self.visited:
                children = self.expand(node, pathcost, goals)
                self.visited[node] = pathcost
                for child, action, child_pathcost in children:
                    if child not in self.visited or child_pathcost < self.visited[child]:
                        heapq.heappush(self.frontier, (int(child_pathcost), [child, plan + [(action, child)], int(child_pathcost)]))
        return []

    def expand(self, node1, cost, goals):
        children = []
        if node1 in self.augmented_abstract_transitions:
            for node2 in self.augmented_abstract_transitions[node1]:
                if "option" in self.distances[node1][node2]:
                    action, distance = self.distances[node1][node2]["option"]
                else:
                    action, distance = self.distances[node1][node2]["action"]
                heuristic = self.get_heuristic(node2, goals)
                cost = cost + distance + heuristic
                children.append((node2, action, cost))
        return children
    
    def get_heuristic(self, node, goals):
        if node not in self.heuristic:
            heuristics = []
            for goal in goals:
                heuristic = self.cat.distance_betn_abstract_states(node._state, goal)
                heuristics.append(heuristic)
            self.heuristic[node] = min(heuristics)
        return self.heuristic[node]

    def compute_depths(self):
        node = self.cat._root
        self.depths[node] = 0
        queue = [node]
        self.max_depth = 0
        while len(queue) > 0:
            node = queue.pop(0)
            if node._state in self.cat._leaves:
                node.is_leaf = True
            else:
                node.is_leaf = False
            for child in node._child:
                if child not in self.depths:
                    self.depths[child] = self.depths[node] + 1
                    queue.append(child)
                    if self.depths[child] > self.max_depth:
                        self.max_depth = self.depths[child]

    def compute_state_to_abs_node(self):
        node = self.cat._root
        queue = [node]
        while len(queue) > 0:
            node = queue.pop(0)
            self.node_to_abs_node[node._state] = node
            for child in node._child:
                queue.append(child)

    def augment_abstract_transitions(self, augment_abstract=False, transitions={}):
        if not augment_abstract:
            transitions = {}
            if not self.replanning:
                for state1 in self.transition_library:
                    # node1 = self.cat.find_node_in_tree([], self.cat._root, state1)[1]
                    if state1 in self.node_to_abs_node:
                        node1 = self.node_to_abs_node[state1]
                        # if node1 is not None:
                        for state2 in self.transition_library[state1]:
                            # node2 = self.cat.find_node_in_tree([], self.cat._root, state2)[1]
                            if state2 in self.node_to_abs_node: #TODO
                                node2 = self.node_to_abs_node[state2]
                                # if node2 is not None:
                                if node1 not in transitions:
                                    transitions[node1] = {}
                                transitions[node1][node2] = {"action" : "action"}
                                self.augment_abstract_transition(node1, node2)

            for optionid, option in self.options.items():
                for state1 in option.initiation_set:
                    # node1 = self.cat.find_node_in_tree([], self.cat._root, state1)[1]
                    if state1 in self.node_to_abs_node:
                        node1 = self.node_to_abs_node[state1]
                        # if node1 is not None:
                        for state2 in option.termination_set:
                            # node2 = self.cat.find_node_in_tree([], self.cat._root, state2)[1]
                            if state2 in self.node_to_abs_node:
                                node2 = self.node_to_abs_node[state2]
                                if node2 is not None:
                                    self.augment_abstract_transition(node1, node2, option)
            return transitions

        if augment_abstract:
            for node1 in transitions:
                for node2 in transitions[node1]:
                    i = 0
                    number_of_levels_for_siblings = 1
                    while i < number_of_levels_for_siblings and node1 is not None and node1._parent is not None:
                        i += 1
                        siblings = self.cat.get_all_leafs(node1._parent)
                        for sibling in siblings:
                            self.augment_abstract_transition(sibling, node2)
                            if sibling not in self.distances:
                                self.distances[sibling] = {}
                            if node2 not in self.distances[sibling]:
                                self.distances[sibling][node2] = {}
                                self.distances[sibling][node2]["action"] = ("action", (self.max_depth - self.depths[node2._parent] + 1) * 10)
                        node1 = node1._parent
            return {}


    def augment_abstract_transition(self, node1, node2, action=None):
        if node1 not in self.augmented_abstract_transitions:
            self.augmented_abstract_transitions[node1] = {}
        if action is None:
            self.augmented_abstract_transitions[node1][node2] = {"action" : "action"}
        else:
            self.augmented_abstract_transitions[node1][node2] = {"option" : action}

    def compute_distances(self):
        for node1 in self.augmented_abstract_transitions:
            if node1 not in self.distances:
                self.distances[node1] = {}
            for node2 in self.augmented_abstract_transitions[node1]:
                if node2 not in self.distances[node1]:
                    self.distances[node1][node2] = {}
                for action in self.augmented_abstract_transitions[node1][node2]:
                    if action == "action":
                        self.distances[node1][node2]["action"] = ("action", (self.max_depth - self.depths[node2] + 1) * 10)
                    else:
                        option = self.augmented_abstract_transitions[node1][node2]["option"]
                        self.distances[node1][node2]["option"] = (option, (self.max_depth - self.depths[node2] + 1) * option.abstract_cost  * 10)
            
    def get_plan_over_options(self, state_abs, goal_states_abs):
        augment_abstract = False
        self.compute_state_to_abs_node()
        transitions = self.augment_abstract_transitions(augment_abstract)
        self.compute_distances()
        action_states = self.search(state_abs, goal_states_abs)
        if len(action_states) == 0 and not self.replanning:
            augment_abstract = True
            self.augment_abstract_transitions(augment_abstract, transitions)
            # self.compute_distances()
            action_states = self.search(state_abs, goal_states_abs)
        print(f"action_states {action_states}")
        actual_plan = []
        if len(action_states) > 0:
            # and (action_states[1][0] != "action" or action_states[-1][0] != "action"):
            start_node = action_states[0][1]
            end_node = None
            i = 0
            for action_state in action_states[1:]:
                i += 1
                if action_state[0] == "action":
                    end_node = action_state[1]
                if action_state[0] != "action" or i == len(action_states)-1:
                    if (start_node is not None and end_node is not None):
                        initiation_set = set([start_node._state])
                        termination_set = set([end_node._state])
                        bridge_option = Option(initiation_set, termination_set, optionid=len(self.options))
                        bridge_option._is_goal_option = False
                        bridge_option._is_bridge_option = True
                        actual_plan.append(bridge_option)
                        start_node = copy.deepcopy(end_node)
                        end_node = None
                if action_state[0] != "action" and action_state[0] != None:
                    actual_plan.append(action_state[0])
        return actual_plan