import networkx as nx
import pickle
import copy
import numpy as np
import math
import pydot

import hyper_param
from src.data_structures.option import Option
from src.misc import utils
from src.data_structures.qvalue_table import Qtable


class OptionsInventor():

    def __init__(self, cat, trajectories, stgs=None):
        self.cat = cat
        self.trajectories = trajectories
        self.stgs = stgs

    def get_pruned_cats(self, optimal_trajectory, high_freq_vars, low_freq_vars, reuse_directory):
        states = [tuple(item[0]) for item in optimal_trajectory] + [tuple(optimal_trajectory[-1][2])]
        cats = []
        for state in states:
            if len(cats) > 0 and self.cat.state(state) in cats[-1]._leaves:
                cats.append(copy.deepcopy(cats[-1]))
            else:
                cat = utils.build_relevant_cat(self.cat, state, high_freq_vars, low_freq_vars, reuse_directory)
                cats.append(copy.deepcopy(cat))
        return states, cats

    # ---------------------------- Abstraction distance function ------------------------------
    def depth_subtree(self, node):
        graph = utils.get_networkx_cat(node)
        path = nx.dag_longest_path(graph)
        return len(path)

    def node_match(self, u, v):
        if u == v and u != {}:
            return True
        return False

    def segregate(self, nodes1, nodes2):
        same_pairs = []
        different1 = []
        different2 = []
        data_dict = {}
        for node in nodes1:
            if node._state not in data_dict:
                data_dict[node._state] = {}
            data_dict[node._state][0] = node
        for node in nodes2:
            if node._state not in data_dict:
                data_dict[node._state] = {}
            data_dict[node._state][1] = node
        for value in data_dict.values():
            if 0 in value and 1 in value:
                same_pairs.append((value[0], value[1]))
            elif 0 in value:
                different1.append(value[0])
            else:
                different2.append(value[1])
        return same_pairs, different1, different2

    def custom_graph_edit_distance(self, cat1, cat2, node1, node2, distance):
        queue = [(node1,node2)]
        while len(queue) > 0:
            node_pair = queue.pop()
            same_pairs, different1, different2 = self.segregate(node_pair[0]._child, node_pair[1]._child)
            for node in different1:
                distance += self.depth_subtree(node)
            for node in different2:
                distance += self.depth_subtree(node)
            queue.extend(same_pairs)
        return distance

    def get_abstraction_distance_pruned_cats(self, cat1, cat2):
        distance = self.custom_graph_edit_distance(cat1, cat2, cat1._root, cat2._root, distance=0)
        return distance

    def get_abstraction_distances(self, states, cats):
        distances = []
        for i in range(len(states)-1):
            cat1 = cats[i]
            cat2 = cats[i+1]
            distance = self.get_abstraction_distance_pruned_cats(cat1, cat2)
            print(states[i], states[i+1], distance)
            distances.append((cat1, cat2, distance))
        return distances

    # ---------------------------- shorter options ------------------------------------------
    def get_distance(self, state1, state2):
        return self.cat.distance_betn_abstract_states(state1, state2) 

    def compute_state_distance_for_trajectory(self, optimal_trajectory):
        abstract_states = [tuple(item[0]) for item in optimal_trajectory] + [tuple(optimal_trajectory[-1][2])]
        dist_max = 0
        traj_dist = []
        for i in range(len(abstract_states)-1):
            dist = self.get_distance(abstract_states[i], abstract_states[i+1])
            # print(abstract_states[i], abstract_states[i+1], dist)
            traj_dist.append([abstract_states[i], abstract_states[i+1], dist])
            if dist > dist_max:
                dist_max = dist
        return traj_dist, dist_max

    
    def compute_salient_indices(self, optimal_trajectory, distances):
        salient_indices = []
        abstraction_dist_threshold = 0
        for i in range(1,len(optimal_trajectory)):
            # prev_trans, prev_dist = optimal_trajectory[i-1], distances[i-1]  #distances[i-1][2]
            next_trans, next_dist = optimal_trajectory[i], distances[i]      #distances[i][2]
            # if (prev_dist <= abstraction_dist_threshold < next_dist) or (prev_dist > abstraction_dist_threshold >= next_dist):
            if next_dist > abstraction_dist_threshold:
                if i-1 not in salient_indices and i >= 0:
                    salient_indices.append(i-1)
                if i not in salient_indices:
                    salient_indices.append(i)
        if len(optimal_trajectory)-1 not in salient_indices:
            salient_indices.append(len(optimal_trajectory)-1)
        return salient_indices

    def segment_trajectories(self, optimal_trajectory, optimal_abstract_trajectory, salient_indices):
        segment_abstract_trajectories = []
        segment_trajectories = []
        start_end_indices = []
        for i in range(len(salient_indices)):
            if i == 0:
                start_id = 0
            else:
                start_id = salient_indices[i-1]+1
            end_id = salient_indices[i]
            start_end_indices.append((start_id, end_id))
            # print(start_id, end_id)
            # segment_abstract_trajectories.append((optimal_abstract_trajectory[start_id:end_id+1], cats[start_id]))
        for start_end in start_end_indices:
            start_id, end_id = start_end[0], start_end[1]
            segment_abstract_trajectories.append(optimal_abstract_trajectory[start_id:end_id+1])
            segment_trajectories.append(optimal_trajectory[start_id:end_id+1])
        return segment_trajectories, segment_abstract_trajectories
    
    def further_segment_trajectories(self, segment_trajectories, segment_abstract_trajectories, traj_dist, dist_max, threshold):
        new_trajectories = []
        new_abstract_trajectories = []
        trajectory = []
        abstract_trajectory = []
        for i in range(len(segment_trajectories)):
            if traj_dist[i][2] > threshold or i == len(segment_trajectories)-1:
                trajectory.append(segment_trajectories[i])
                abstract_trajectory.append(segment_abstract_trajectories[i])
                if trajectory != []:
                    new_trajectories.append(trajectory)
                    new_abstract_trajectories.append(abstract_trajectory)
                trajectory = []
                abstract_trajectory = []
            else:
                trajectory.append(segment_trajectories[i])
                abstract_trajectory.append(segment_abstract_trajectories[i])
        return new_trajectories, new_abstract_trajectories

    # ------------------------- main function --------------------------------------------
    def construct_options(self, directory_path, env, envid, qtable_full={}, optionid=None, save_options=True, dont_break=False):
        optimal_trajectory = self.get_optimal_trajectory()
        optimal_abstract_trajectory = [[self.cat.state(transition.state), transition.action, self.cat.state(transition.next_state), transition.state, transition.next_state] for transition in optimal_trajectory] 
        # high_freq_vars_, low_freq_vars_ = self.extract_freq_vars(optimal_trajectory)
        high_freq_vars, low_freq_vars, vars_to_refinement = self.get_variable_refinement_frequency()

        qtable = Qtable()
        for key,value in qtable_full._qtable.items():
            if key in self.cat._leaves:
                qtable._qtable[key] = value

        # qvalues = [qtable._qtable[transition[0]] for transition in optimal_abstract_trajectory ]
        # qvalues_max = [np.max(list(actions_qvalues.values())) for actions_qvalues in qvalues]
        # qvalues_variation = [np.std(list(actions_qvalues.values())) for actions_qvalues in qvalues]
        # all = []
        # for i in range(len(optimal_abstract_trajectory)):
        #     if len(all) == 0:
        #         all.append([optimal_abstract_trajectory[i][0], (qvalues[i], qvalues_max[i], qvalues_variation[i])])
        #     if len(all) > 0 and all[-1][0] != optimal_abstract_trajectory[i][0]:
        #         all.append([optimal_abstract_trajectory[i][0], (qvalues[i], qvalues_max[i], qvalues_variation[i])])

        if env._train_option is not None and dont_break == True:
            option = env._train_option
            # initial_states_abs = [transition[0] for transition in optimal_abstract_trajectory]
            initial_states_abs = optimal_abstract_trajectory[0][0]
            option.initiation_set = set([initial_states_abs])
            goal_state_abs = optimal_abstract_trajectory[-1][2]
            if goal_state_abs in option.initiation_set:
                option.initiation_set.remove(goal_state_abs)
            option.termination_set = set([goal_state_abs])
            option.init_states = [optimal_trajectory[0].state]
            option.initialize_qtable(qtable)
            option.add_term_states(optimal_trajectory, optimal_abstract_trajectory)
            option.cat = self.cat
            states = [tuple(transition.state) for transition in optimal_trajectory] + [tuple(optimal_trajectory[-1].next_state)]
            option.policy_states = set(states)
            option.trajectories = [optimal_trajectory]
            option.abstract_trajectory = [optimal_abstract_trajectory]
            option.cost = len(states)
            option.abstract_cost = len(set(states))
            options = [option]
        else:
            segment_trajectories = []
            segment_abstract_trajectories = []
            if len(low_freq_vars) > 0:
                # states, cats = self.get_pruned_cats(optimal_trajectory, high_freq_vars, low_freq_vars, directory_path)
                # distances = self.get_abstraction_distances(states, cats)
                # maxdist = max([item[2] for item in distances])
                distances = self.get_distance_list(optimal_trajectory, low_freq_vars)
                salient_indices = self.compute_salient_indices(optimal_trajectory, distances)
                segment_trajectories, segment_abstract_trajectories = self.segment_trajectories(optimal_trajectory, optimal_abstract_trajectory, salient_indices)
            if len(segment_abstract_trajectories) == 0 or len(low_freq_vars) == 0:
                segment_abstract_trajectories = [optimal_abstract_trajectory]
                segment_trajectories = [optimal_trajectory]

            if hyper_param.ancestor_dist_threshold_factor != 1.0:
                self.cat.compute_depths()
                _, dist_max = self.compute_state_distance_for_trajectory(optimal_abstract_trajectory)
                threshold = dist_max * hyper_param.ancestor_dist_threshold_factor
                print("local saliency max and theshold: ", dist_max, threshold)
                new_segment_trajectories = []
                new_segment_abstract_trajectories = []
                for i in range(len(segment_abstract_trajectories)):
                    segment_abstract_trajectory = segment_abstract_trajectories[i]
                    segment_trajectory = segment_trajectories[i]
                    cost_segment_traj = len([tuple(transition.state) for transition in segment_trajectory] + [tuple(segment_trajectory[-1].next_state)])
                    if cost_segment_traj < 10:
                        threshold = dist_max
                    traj_dist, _ = self.compute_state_distance_for_trajectory(segment_abstract_trajectory)
                    new_segment_trajectories, new_segment_abstract_trajectories = self.further_segment_trajectories(segment_trajectory, segment_abstract_trajectory, traj_dist, dist_max, threshold=threshold)
            else:
                new_segment_trajectories = copy.deepcopy(segment_trajectories)
                new_segment_abstract_trajectories = copy.deepcopy(segment_abstract_trajectories)

            options = []
            for i in range(len(new_segment_abstract_trajectories)):
                # init_set, cost_set, traj_set = self.construct_initiation_sets(start_end, [segment_abstract_trajectory], [segment_trajectory])
                # partial_options = []
                # for pair, init in init_set.items():
                #     option = Option(init, set([pair[1]]))
                #     option.cost = cost_set[pair]
                #     option.trajectories = traj_set[pair]
                #     partial_options.append(option)
                # for option in partial_options:
                #     option.initialize_qtable(qtable)
                #     option.cat = self.cat
                #     option.add_term_states(segment_trajectory, segment_abstract_trajectory)
                # options.extend(partial_options)
                print(new_segment_abstract_trajectories[i][0][0], new_segment_abstract_trajectories[i][-1][2])
                initiation_set = set()
                initiation_set.add(tuple(new_segment_abstract_trajectories[i][0][0]))
                termination_set = set()
                termination_set.add(tuple(new_segment_abstract_trajectories[i][-1][2]))
                option = Option(initiation_set, termination_set)
                option.cost = len(new_segment_abstract_trajectories[i])
                states = set()
                for traj in new_segment_abstract_trajectories[i]:
                    states.add(tuple(traj[0][0]))
                option.abstract_cost = len(states)
                option.policy_states = set(states)
                option.trajectories = [new_segment_trajectories[i]]
                option.abstract_trajectories = [new_segment_abstract_trajectories[i]]
                option.initialize_qtable(qtable)
                option.cat = self.cat
                option.init_states = [new_segment_trajectories[i][0].state]
                option.add_term_states(new_segment_trajectories[i], new_segment_abstract_trajectories[i])
                options.append(option)

            # for option in options:
            #     option = self.learn_initiation_policy(option, self.cat, env, envid, directory_path)
        
            # utils.plot_option_path(options, self.cat, env, envid, directory_path, filename="catoptions_")

        if save_options:
            filename = "options"
            if optionid is not None:
                filename = "option"+str(optionid)+"_"+filename
            file = open(directory_path+"/"+filename+".pickle","wb")
            pickle.dump(options,file)
            file.close()
        # self.plot_optimal_trajectory_on_cat(directory_path)

        # self.build_option_model(options, env.var_to_name, directory_path)
        print("Options learned...")
        return options
    
    def get_variable_refinement_frequency(self):
        vars_to_refinement = {}
        node = self.cat._root
        queue = [node]
        visited = []
        while len(queue) > 0:
            node = queue.pop(0)
            visited.append(node)
            children = node._child
            if len(children) > 0:
                child = children[0]
                for i in range(len(child._state)):
                    if i not in vars_to_refinement:
                        vars_to_refinement[i] = 0
                    if child._state[i] != node._state[i]:
                        vars_to_refinement[i] += 1
            for child in children:
                queue.append(child)
        maxm = max(vars_to_refinement.values())
        high_feq_vars = []
        low_feq_vars = []
        for var in vars_to_refinement:
            if vars_to_refinement[var] > maxm/2:
                high_feq_vars.append(var)
            else:
                low_feq_vars.append(var)
        return high_feq_vars, low_feq_vars, vars_to_refinement
    

    def build_option_model(self, options, var_to_name, directory_path):
        string = "(define (domain domain_name)\n"
        predicates = set()
        optionid = 0
        option_pre_eff = {}
        for option in options:
            changing_vars = set()
            abs_term = list(option.termination_set)[0]
            print(abs_term)
            # for init in option.initiation_set:
            #     for i in range(len(init)):
            #         if not self.lie_within(term[i], init[i]):
            #             changing_vars.add(i)
            init = option.cat.state(option.trajectories[0][0])
            term = option.term_states[0]
            for state in option.trajectories[0]:
                for i in range(len(state)):
                    if state[i] != term[i]:
                        changing_vars.add(i)
            pre = "(and "
            eff = "(and "
            for var in changing_vars:
                if var not in [0,1,2,3]:
                    predicates.add(var_to_name[var])
                    valpre = init[var].split(",")[0]
                    valeff = abs_term[var][0]
                    print(var, var_to_name[var], valpre, valeff)
                    if valpre == "0":
                        pre += "(not ({} ?p)) ".format(var_to_name[var])
                    else:
                        pre += "({} ?p) ".format(var_to_name[var])
                    if valeff == "0":
                        eff += "(not ({} ?p)) ".format(var_to_name[var])
                    else:
                        eff += "({} ?p) ".format(var_to_name[var])
            pre += ")"
            eff += ")"
            option_pre_eff[optionid] = (pre, eff)
            # string += "option{} \n \t precondition: {} \n \t\t   effect: {}".format(optionid, pre, eff) +"\n\n"
            optionid += 1 

        string += "(:predicates \n\t"
        for pred in predicates:
            pred += " ?p"
            string += " ("+pred+") "
        string += "\n)\n"

        for op, pre_eff in option_pre_eff.items():
            string += "(:action option{}\n".format(op)
            string += "\t:parameters (?p)\n".format()
            string += "\t:precondition {}\n".format(pre_eff[0])
            string += "\t:effect {}\n".format(pre_eff[1])
            string += ")\n"

        string += "\n)"
        f = open(directory_path + "/heatmaps/option_model.pddl", "w")
        f.write(string)
        f.close()

    def lie_within(self, range1, range2):
        low1, high1 = range1.split(",")
        low2, high2 = range2.split(",")
        if (low1 < low2 and high2 < high2) or (low2 < low1 and high2 < high1):
            return True
        return False

    def get_distance_list(self, optimal_trajectory, low_freq_vars):
        current_abs_range = self.get_low_freq_var_range(optimal_trajectory[0].state, low_freq_vars)
        # print(current_abs_range)
        distance = []
        for transition in optimal_trajectory[1:]:
            s0 = transition.state
            if current_abs_range is not None and self.low_freq_var_falls_within(s0, current_abs_range, low_freq_vars):
                distance.append(0)
                continue
            else:
                distance.append(1)
                current_abs_range = self.get_low_freq_var_range(s0, low_freq_vars)
                # print(current_abs_range)
        g = optimal_trajectory[-1].next_state
        if current_abs_range is not None and self.low_freq_var_falls_within(g, current_abs_range, low_freq_vars):
            distance.append(0)
        else:
            distance.append(1)
            # print(current_abs_range)
        return distance

    def get_low_freq_var_range(self, state, low_freq_vars):
        low_freq_vars = np.array(low_freq_vars)
        state = np.array(state)
        for leaf in self.cat._leaves:
            l = np.array(list(map(lambda x: list(map(float, x.split(','))), leaf)))
            if self.low_freq_var_falls_within(state, l, low_freq_vars):
                return l

        return None

    def low_freq_var_falls_within(self, state, current_abs_range, low_freq_vars):
        root = np.array(list(map(lambda x: list(map(float, x.split(','))), self.cat._root._state)))
        state = np.array(state)
        if np.all(state[low_freq_vars] >= current_abs_range[low_freq_vars, 0]):
            if np.all(state[low_freq_vars] < current_abs_range[low_freq_vars, 1]):
                return True
            elif np.all(np.logical_or(state[low_freq_vars] < current_abs_range[low_freq_vars, 1],  state[low_freq_vars] == current_abs_range[low_freq_vars, 1])):
                mask = state[low_freq_vars] == current_abs_range[low_freq_vars, 1]
                if np.all(state[low_freq_vars][mask] == root[low_freq_vars,1][mask]):
                    return True
        
        return False



    # ---------------------------- processing -----------------------------------
    def extract_freq_vars(self, optimal_trajectory):
        num_vars = len(optimal_trajectory[0].state)
        high_freq_vars = np.arange(num_vars)
        low_freq_vars = []
        if num_vars > 2:
            freq = {}
            for var_i in range(num_vars):
                if var_i not in freq:
                    freq[var_i] = 0
                for transition in optimal_trajectory:
                    if self.var_differs(transition.state, transition.next_state, var_i):
                        freq[var_i] += 1
            avg_freq = sum(freq.values())/len(freq)
            high_freq_vars, low_freq_vars = [], []
            for var_i, freq in freq.items():
                if freq >= avg_freq:
                    high_freq_vars.append(var_i)
                else:
                    low_freq_vars.append(var_i)
        return high_freq_vars, low_freq_vars

    def get_vars_differ(self, state1, state2):
        vars = []
        for var_i in range(len(state1)):
            if self.var_differs(state1, state2, var_i):
                vars.append(var_i)
        return vars

    def var_differs(self, state1, state2, var_i):
        if state1[var_i] != state2[var_i]:
            return True
        return False

    def process_trajectory(self, optimal_trajectory):
        # remove cycles
        processed_trajectory = []
        for transition in optimal_trajectory.trajectory:
            if transition.state != transition.next_state:
                if len(processed_trajectory) == 0:
                    processed_trajectory.append(transition)
                elif processed_trajectory[-1].state == transition.next_state and processed_trajectory[-1].next_state == transition.state:
                    del processed_trajectory[-1]
                else:
                    processed_trajectory.append(transition)
        if len(processed_trajectory) == 0:
            processed_trajectory = optimal_trajectory.trajectory
        return processed_trajectory

    def get_optimal_abstract_trajectory(self):
        minm = 1000000
        optimal_trajectory = []
        for traj in self.trajectories["abstract_trajectories"]:
            traj = self.process_trajectory(traj)
            if len(traj) < minm:
                optimal_trajectory = traj
                minm = len(traj)
        # optimal_trajectory = self.process_trajectory(optimal_trajectory)
        return optimal_trajectory

    def get_optimal_trajectory(self):
        minm = 1000000
        optimal_trajectory = []
        for traj in self.trajectories["trajectories"]:
            traj = self.process_trajectory(traj)
            if len(traj) < minm:
                optimal_trajectory = traj
                minm = len(traj)
        # optimal_trajectory = self.process_trajectory(optimal_trajectory)
        return optimal_trajectory

    def convert_to_tuple(self, a):
        a = a.split(' ->')[0]
        a = a.replace('(','')
        a = a.replace(')','')
        a = a.replace('\'','')
        a = a.split(", ")
        return tuple(a)  

    def transform_options(self, options, state_mapping):
        transformed_options = []
        for option in options:
            option_copy = copy.deepcopy(option)
            init = copy.deepcopy(option_copy.initiation_set)
            for state in init:
                new_state = state_mapping[state]
                option_copy.initiation_set.remove(state)
                option_copy.initiation_set.add(new_state)
            term = copy.deepcopy(option_copy.termination_set)
            for state in term:
                new_state = state_mapping[state]
                option_copy.termination_set.remove(state)
                option_copy.termination_set.add(new_state)
            transformed_options.append(option_copy)
        return transformed_options

    # -------------------------- plotting ------------------------------------------
    def find_color(self, graph, options):
        N = len(options)
        colors = []
        for i in range(N):
            colors.extend(["red","green","blue","orange","purple"])

        i = 0
        node_to_color = {}
        option_to_color = {}
        for option in options:
            # nodes = option.initiation_set.union(option.termination_set)
            nodes = option.initiation_set
            for node in nodes:
                graph.add_node(node, fillcolor=colors[i], style="filled")
                node_to_color[node] = colors[i]
            option_to_color[option] = colors[i]
            i += 1
        
        # print("\n\n")
        for node in graph.nodes:
            if node not in node_to_color.keys():
                if node not in node_to_color:
                    graph.add_node(node, fillcolor="white", style="filled")
                else:
                    end_nodes = list(graph[node].keys())
                    if node in end_nodes:
                        end_nodes.remove(node)
                    end_node = end_nodes[0]
                    graph.add_node(node, fillcolor=node_to_color[end_node], style="filled")
                    
        abstract_to_color = {}
        for node, color in node_to_color.items():
            abs_state = node
            if ' ->' in node:
                abs_state = node.split(" ->")[0]
            abstract_to_color[abs_state] = color
        return graph, abstract_to_color

    def color_options(self, options, graph, directory_path, envid):
        options_init_colored_graph, node_to_color = self.find_color(graph, options)
        filename = "env"+str(envid)+"_local_stg_options_new"
        # if optionid is not None:
        #     filename = "option"+str(optionid)+"_"+filename
        nx.nx_pydot.write_dot(options_init_colored_graph, directory_path+"/"+filename+".dot")

    def get_initial_node_in_graph(self, graph, initial_state):
        initial_state = "('"+"', '".join(initial_state[0])+"')" #convert tuple state to string
        for node in graph.nodes:
            if initial_state in node:
                initial_state = node
                break
        return initial_state

    def plot_optimal_trajectory_on_cat(self, directory_path):
        # plot CAT with transitions at the leaves of the most optimal trajectory found
        cat_path = directory_path+"/cat.dot"
        graph = nx.MultiDiGraph(nx.nx_pydot.read_dot(cat_path))
        node_pos = nx.drawing.nx_pydot.pydot_layout(graph)

        optimal_abstract_trajectory = self.get_optimal_abstract_trajectory()
        start_end = [self.get_initial_node_in_graph(graph, state) for state in optimal_abstract_trajectory]
        for i in range(len(start_end)-1):
            graph.add_edge(start_end[i],start_end[i+1],color="orange",width=100.0)

        nx.set_node_attributes(graph, node_pos, "pos")
        cat_path = directory_path+"/cat_new.dot"
        nx.nx_pydot.write_dot(graph, cat_path)


