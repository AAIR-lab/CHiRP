import pickle
import copy
import os
import re
import networkx as nx
from src.data_structures.option import Option
import numpy as np
import random
import matplotlib.colors as mcolors

from src.misc import utils
from src.data_structures.cat import CAT
from src.misc.results import Results 

def convert_to_tuple(a):
    a = a.replace('(','')
    a = a.replace(')','')
    a = a.replace('\'','')
    a = a.split(", ")
    return tuple(a)  

def is_root(graph, node):
    if graph.in_degree(node) == 0:
        return True
    return False

def reuse_tree(reuse_cat_path, is_int_variable, continuous_state=False):
    '''
        reads a CAT from dot into abstract tree data structure
    '''
    graph = nx.MultiDiGraph(nx.nx_pydot.read_dot(reuse_cat_path))
    label_mapping = {}
    state_ranges = []
    for node in graph.nodes:
        pattern = re.compile(r'\(.*\)')
        re_obj = re.search(pattern, node)
        if re_obj:
            state_tup = re_obj.group(0)
            state_tup = convert_to_tuple(state_tup)
            # if "root" in node:
            if is_root(graph, node):
                root_state_tup = copy.deepcopy(state_tup)
                for item in state_tup:
                    item_list = item.split(",")
                    new_item_list = []
                    for n in item_list:
                        new_item_list.append(np.float32(n))
                    state_ranges.append(tuple(new_item_list))
            label_mapping[node] = state_tup
    graph = nx.relabel_nodes(graph, label_mapping)
    
    reuse_tree = CAT(state_ranges, [1 for i in range(len(state_ranges))], is_int_variable, continuous_state)
    root_node = reuse_tree._root
    mapping = {}
    mapping[root_state_tup] = root_node
    
    stack = [root_state_tup]
    visited = []
    leaves = []
    while stack:
        node1_tup = stack.pop()
        if node1_tup in visited:
            continue
        visited.append(node1_tup)
        
        children = [item[1] for item in list(graph.out_edges(node1_tup))]
        for node2_tup in children:
            if node1_tup == root_state_tup:
                new_node1 = mapping[root_state_tup]
            else:
                if node1_tup in mapping:
                    new_node1 = mapping[node1_tup]
                else:
                    new_node1 = reuse_tree.add_node(split = [], abs_state = node1_tup)
                    mapping[node1_tup] = new_node1
            if node2_tup in mapping:
                new_node2 = mapping[node2_tup]
            else:
                new_node2 = reuse_tree.add_node(split = [], abs_state = node2_tup)
                mapping[node2_tup] = new_node1
            
            if new_node1._state in reuse_tree._leaves:
                del reuse_tree._leaves[new_node1._state]
            
            new_node1._child.append(new_node2)
            new_node2._parent = new_node1
            if new_node1._parent:
                vector = list()
                for i in range(len(new_node1._state)):
                    if new_node1._state[i] != new_node2._state[i]:
                        vector.append(1)
                    else:
                        vector.append(0)
                new_node1._split, _ = reuse_tree.update_split(new_node1._state, new_node1._parent._split, vector)
            mapping[node1_tup] = new_node1
            mapping[node2_tup] = new_node2
            
            if node2_tup not in visited:
                stack.append(node2_tup)
        if len(children) == 0:
            leaves.append(mapping[node1_tup])
    reuse_tree._leaves = leaves
    reuse_tree._root = mapping[root_state_tup]
    return reuse_tree

def compute_critical_frequency(trajectories):
    criticality_frequency = dict()
    all_states = set()
    for traj in trajectories:
        states = set()
        for s_a_ns in traj.trajectory:
            states.add(s_a_ns.state)
        states.add(traj.trajectory[-1].next_state)
        for state in states:
            if state not in criticality_frequency:
                criticality_frequency[state] = 0
            criticality_frequency[state] += 1
        all_states.update(states)
    for state in criticality_frequency:
        criticality_frequency[state] = round(criticality_frequency[state]/len(trajectories),3)
    return criticality_frequency


def update_option_sets(option, updated_abstractions):
    #  as abstractions in CAT have been updated
    if len(updated_abstractions) > 0:
        init = copy.deepcopy(option.initiation_set)
        term = copy.deepcopy(option.termination_set)
        for old_abs_state in updated_abstractions:
            if old_abs_state in init:
                option.initiation_set.remove(old_abs_state)
                for new_abs_state in updated_abstractions[old_abs_state]:
                    option.initiation_set.add(new_abs_state)
            if old_abs_state in term:
                option.termination_set.remove(old_abs_state)
                for new_abs_state in updated_abstractions[old_abs_state]:
                    option.termination_set.add(new_abs_state)
    return option

def update_options_to_single_term(abstract, option):
    # termination sets of options should contain a single state
    if len(option.termination_set) > 1:
        if len(option.term_states) > 0:
            term_abs_states = list(set([abstract.state(term_state) for term_state in option.term_states]))
            assert len(term_abs_states) > 0
            new_termination_set = set()
            for term_abs_state in term_abs_states:
                new_termination_set.add(term_abs_state)
            option.termination_set = new_termination_set
        else:
            pass
    return option

def plot_options_graph(directory, options):
    graph = nx.DiGraph()
    for source_option in options:
        for target_option in options:
            source = source_option.termination_set
            target = target_option.initiation_set
            if len(source.intersection(target)) > 0: 
            # chain_options, found_plan = Search.get_plan_over_options(source, target, options)
            # if found_plan:
                start_node = source_option.get_goal_abs_state()
                end_node = target_option.get_goal_abs_state()
                graph.add_edge(start_node, end_node, allow_duplicates=False)
    nx.nx_pydot.write_dot(graph, directory+"/options.dot")
    return graph

########################### plot CATs ##############################################
def rgb2hex(rgb):
    hex = '#' + ''.join(f'{i:02X}' for i in rgb)
    assert len(hex)==7
    # hex = colorsys.rgb_to_hex(rgb)
    return hex

def augment_data(graph, qtable, best_actions, distances=None, criticality_frequency=None):
    mapping = dict()
    for node in graph.nodes:
        # if "root" in node: 
        if is_root(graph, node):
            mapping[node] = str(node) + " -> "
        else:
            action, best_actions = Results.get_best_action(str(node), qtable, best_actions)
            mapping[node] = str(node) + " -> "+ str(action)

    if criticality_frequency:
        new_mapping = dict()
        new_label = dict()
        for edge in graph.edges:
            node1 = edge[0]
            node2 = edge[1]
            if node1 not in new_mapping:
                freq = 0.0
                if node1 in criticality_frequency:
                    freq = criticality_frequency[node1]
                new_mapping[node1] = mapping[node1] + " [" + str(freq) +"]"
            if node2 not in new_mapping:
                freq = 0.0
                if node2 in criticality_frequency:
                    freq = criticality_frequency[node2]
                new_mapping[node2] = mapping[node2] + " [" + str(freq) +"]"

            distance = distances[node1][node2]
            new_label[(new_mapping[node1],new_mapping[node2])] = [graph[node1][node2]["label"], distance]

        new_graph = nx.relabel_nodes(graph, new_mapping)

        for node1_node2_tup, label in new_label.items():
            new_graph[node1_node2_tup[0]][node1_node2_tup[1]]["label"] = label
    else:
        new_graph = nx.relabel_nodes(graph, mapping)
    return new_graph, best_actions

def augment_transition_function(graph, tree, local_stg):
    for node1, value in local_stg.items():
        if tree.find_node(node1):
            for node2, actions in value.items():
                if tree.find_node(node2):
                    graph.add_edge(node1, node2, label=sorted(list(actions)), color="orange")
    return graph

def get_networkx_cat(root):
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

def plot_cat(tree, directory, local_stg={}, qtable=None, abstraction_colors={}, best_actions={}, augment_options=False, abstract_to_color={}):
    graph = get_networkx_cat(tree._root)
    if qtable:
        qmax, qmin, qavg = Results.get_max_min(qtable)

    for node in graph.nodes:
        # if "root" in node: 
        if is_root(graph, node):
            graph.add_node(node)
        else: 
            # if augment_options and str(node) in abstract_to_color:
            #     color = abstract_to_color[str(node)]
            #     graph.add_node(node, fillcolor=color, style="filled")
            if qtable:
                # color, abstraction_colors = Results.get_color_for_abstract_state(str(node), qtable, qmax, qmin, qavg, abstraction_colors)
                # color = rgb2hex(color)
                # graph.add_node(node, fillcolor=color, style="filled")
                graph.add_node(node)
            # print("color not found")
        if tree.find_node(node):
            graph.add_node(node, shape="box")

    if augment_options:
        # pos = nx.nx_pydot.graphviz_layout(graph)
        graph = augment_transition_function(graph, tree, local_stg)

    graph, best_actions = augment_data(graph, qtable, best_actions)

    if not os.path.exists(directory+"/"):
        os.makedirs(directory+"/")
        
    if augment_options:
        # graph = nx.draw(graph, pos=pos)
        # nx.drawing.nx_agraph.write_dot(graph, directory+"/cat_options_"+str(index)+".dot")
        nx.nx_pydot.write_dot(graph, directory+"/cat_options.dot")
    else:
        nx.nx_pydot.write_dot(graph, directory+"/cat.dot")
    file = open(directory+"/cat_nx.pickle", 'wb')
    pickle.dump(graph, file)
    file.close()
    return abstraction_colors, best_actions

############################ plot stg #################################################
def plot_local_stg(local_stg, directory, qtable, abstraction_colors, best_actions, distances=None, frequency=None, optionid=None, plot_local_stg=True):
    graph = nx.from_dict_of_dicts(local_stg, create_using=nx.DiGraph)
    qmax, qmin, qavg = Results.get_max_min(qtable)

    for edge in graph.edges:
        graph.add_edge(edge[0], edge[1], capacity=200)
    
    for node in graph.nodes:
        # if "root" in node: 
        if is_root(graph, node):
            graph.add_node(node)
        else: 
            color, abstraction_colors = Results.get_color_for_abstract_state(str(node), qtable, qmax, qmin, qavg, abstraction_colors)
            color = rgb2hex(color)
            graph.add_node(node, fillcolor=color, style="filled")
            # print("color not found")

    graph, best_actions = augment_data(graph, qtable, best_actions, distances, frequency)

    if plot_local_stg:
        filename = "local_stg"
        if optionid is None:
            filepath = directory+"/"+filename+".pickle"
            file = open(filepath, "wb")
            pickle.dump(graph,file)
            file.close()
    return graph, abstraction_colors, best_actions

def plot_stgs(stgs, stg, directory, optionid=None):
    graph = nx.from_dict_of_dicts(stg, create_using=nx.DiGraph)
    for edge in graph.edges:
        graph.add_edge(edge[0], edge[1], capacity=200)
        
    filename = "stgs"
    if optionid is None:
        filepath = directory+"/"+filename+".pickle"
        file = open(filepath, "wb")
        pickle.dump(stgs,file)
        file.close()

     
def find_path_in_stgs(stgs_nx, state_abs, goal_state_abs):
    found_path = False
    path = []
    for index in range(len(stgs_nx)-1,0,-1):
        stg = stgs_nx[index]
        paths = list(nx.all_simple_paths(stg, state_abs, goal_state_abs))
        if len(paths) > 0:
            path = nx.shortest_path(stg, state_abs, goal_state_abs)
            found_path = True
            break
    return path

############################ plot heatmaps #########################################
def plot_heatmaps(directory, index, env, init_state, goal_state, all_data, qtable, abstract_colors, abstract_best_actions, optionid=None):
    main_maze = env._maze
    init = init_state[:2]
    goal = goal_state[:2]
    obj_locs = env._obj_locs
    qmax, qmin, qavg = Results.get_max_min(qtable)    
    for data in all_data:
        maze_abs = data[0]
        file_name_policy = directory+ "/"
        if optionid is not None:
            file_name_policy += "option"+str(optionid)+"_"
            file_name_policy += str(data[1]) + "_policy.png"
        else:
            file_name_policy += str(data[1]) + "_policy.png"
        abstract_colors, abstract_best_actions = Results.get_qtable_heatmap(main_maze, maze_abs, 40, qtable, qmax, qmin, qavg, file_name_policy, init, goal, obj_locs, abstract_colors, abstract_best_actions, label=True)
    return abstract_colors, abstract_best_actions

def plot_option_heatmap(env, init_state, goal_state, maze_abs, directory, filename, qtable):
    main_maze = env._maze
    init = init_state[:2]
    goal = goal_state[:2]
    obj_locs = env._obj_locs
    file_path = directory+ "/" + filename
    Results.get_option_heatmap(main_maze, maze_abs, 40, qtable, file_path, init, goal, obj_locs)

def make_rgb_transparent(rgb, alpha = 0.5):
    r, g, b = rgb[0], rgb[1], rgb[2]
    return [
        255 - (255-r)*(1-alpha),
        255 - (255-g)*(1-alpha),
        255 - (255-b)*(1-alpha)
    ]

def plot_option_path(chain_options, cat, env, env_id, dir_path, plot_individually=False, filename=""):  
    maze_abs = np.chararray(env._maze.shape, itemsize = 100 ,unicode=True)
    maze_abs[:] = ''
    max_y = maze_abs.shape[0] 
    max_x = maze_abs.shape[1]

    colors = [[elem*255 for elem in list(color)] for name, color in mcolors.BASE_COLORS.items()]
    colors.remove([0,0,0])
    colors.remove([255,255,255])
    for color in copy.deepcopy(colors):
        for i in range(len(color)):
            if color[i] == 0:
                color[i] += 70
        colors.append(color)

    if plot_individually == False:
        qtable = {}
        k = 0
        new_chain_options = {}
        for option in chain_options:
            term = list(option.termination_set)[0][2:]
            if term not in new_chain_options:
                new_chain_options[term] = []
            new_chain_options[term].append(option)

        for term, chain_options in new_chain_options.items():
            maze_abs = np.chararray(env._maze.shape, itemsize = 100 ,unicode=True)
            maze_abs[:] = ''
            max_y = maze_abs.shape[0] 
            max_x = maze_abs.shape[1]

            filepath = filename
            for option in chain_options:
                color = colors[k%len(colors)]
                for abs_state in list(option.termination_set):
                    # abs_state = abs_state[:env._goal_vars[0]]
                    abs_state = abs_state[:2]
                    qtable[str(abs_state)] = color
                    x_min_max = [float(item) for item in abs_state[0].split(",")]
                    y_min_max = [float(item) for item in abs_state[1].split(",")]
                    for i in range(int(x_min_max[0]),int(x_min_max[1])):
                        for j in range(int(y_min_max[0]), int(y_min_max[1])):
                            maze_abs[i][j] = str(abs_state)
                k += 1

            for option in chain_options:
                abs_state = list(option.termination_set)[0]
                # abs_state = abs_state[:env._goal_vars[0]]
                abs_state = abs_state[:2]
                color = list(qtable[str(abs_state)])
                color = make_rgb_transparent(color, 0.8)
                for abs_state in list(option.initiation_set):
                    if str(abs_state) not in qtable:
                        # abs_state = abs_state[:env._goal_vars[0]]
                        abs_state = abs_state[:2]
                        x_min_max = [float(item) for item in abs_state[0].split(",")]
                        y_min_max = [float(item) for item in abs_state[1].split(",")]
                        for i in range(int(x_min_max[0]),int(x_min_max[1])):
                            for j in range(int(y_min_max[0]), int(y_min_max[1])):
                                if maze_abs[i][j] == "":
                                    maze_abs[i][j] = str(abs_state)
                                    qtable[str(abs_state)] = color

            filepath += str(term)+".png"
            plot_option_heatmap(env, env._init_state, env._goal_state, maze_abs, dir_path+"/heatmaps", filepath, qtable)
    else:
        k = 0
        for option in chain_options:
            new_filename = copy.deepcopy(filename)
            maze_abs = np.chararray(env._maze.shape, itemsize = 100 ,unicode=True)
            maze_abs[:] = ''
            max_y = maze_abs.shape[0] 
            max_x = maze_abs.shape[1]
            qtable_copy = dict()
            color = colors[i%len(colors)]
            for abs_state in list(option.termination_set):
                if len(env._goal_vars) > 0:
                    abs_state = abs_state[:env._goal_vars[0]]
                qtable_copy[str(abs_state)] = color
                x_min_max = [float(item) for item in abs_state[0].split(",")]
                y_min_max = [float(item) for item in abs_state[1].split(",")]
                for i in range(int(x_min_max[0]), int(x_min_max[1])):
                    for j in range(int(y_min_max[0]), int(y_min_max[1])):
                        maze_abs[i][j] = str(abs_state)
            color = make_rgb_transparent(color, 0.5)
            for abs_state in list(option.initiation_set):
                if len(env._goal_vars) > 0:
                    abs_state = abs_state[:env._goal_vars[0]]
                qtable_copy[str(abs_state)] = color
                x_min_max = [float(item) for item in abs_state[0].split(",")]
                y_min_max = [float(item) for item in abs_state[1].split(",")]
                for i in range(int(x_min_max[0]), int(x_min_max[1])):
                    for j in range(int(y_min_max[0]), int(y_min_max[1])):
                        if maze_abs[i][j] == "":
                            maze_abs[i][j] = str(abs_state)
            
            if option._qtable == {}:
                new_filename = "bridgeoptionpath_"+str(k)+".png"
            else:
                new_filename = "optionpath_"+str(k)+".png"
            plot_option_heatmap(env, env._init_state, env._goal_state, maze_abs, dir_path+"/heatmaps", new_filename, qtable_copy)
            k += 1
        if len(chain_options) == 0:
            qtable_copy = dict()
            filename = "empty_optionpath.png"
            plot_option_heatmap(env, env._init_state, env._goal_state, maze_abs, dir_path+"/heatmaps", filename, qtable_copy)

# ----------------------------- build pruned CATs ------------------------------
# a concrete state falls within an abstract_state
def fallsWithin(concrete_state, abstract_state, vars_i):
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

# two abstract states overlap i.e. values overlap for each variable
def overlap(state1, state2):
    l1, u1 = [], []
    for item in state1:
        item_list = item.split(",")
        l1.append(np.float32(item_list[0]))
        u1.append(np.float32(item_list[1]))

    l2, u2 = [], []
    for item in state2:
        item_list = item.split(",")
        l2.append(np.float32(item_list[0]))
        u2.append(np.float32(item_list[1]))

    overlap = True
    for i in range(len(l1)):
        if (l1[i] <= l2[i] and u1[i] > l2[i]) or (l2[i] <= l1[i] and u2[i] > l1[i]):
            overlap = True
        else:
            return False
    return overlap

def build_relevant_cat(cat, concrete_state, high_freq_vars, low_freq_vars, directory=None):
    cat = copy.deepcopy(cat)

    # get all abstract states that satisfy concrete state for low_freq_vars 
    abstract_states = cat._leaves
    relevant_nodes = set()
    relevant_states = set()
    for abstract_state in abstract_states:
        if fallsWithin(concrete_state, abstract_state, low_freq_vars):
            node = cat.find_node_in_tree([], cat._root, abstract_state)[1]
            relevant_states.add(node._state)
            relevant_nodes.add(node)

    # add all their parent nodes
    all_nodes = copy.deepcopy(relevant_nodes)
    for node in relevant_nodes:
        temp_node = node
        while temp_node != cat._root:
            temp_node = temp_node._parent
            all_nodes.add(temp_node)
            relevant_states.add(temp_node._state)

    # find remaining abstract states in the CAT
    irrelevant_states = set()
    for state in abstract_states:
        if state not in relevant_states:
            irrelevant_states.add(state)
            
    # find remaining nodes in the CAT (not necessarily an abstract state) as these are irrelevant
    irrelevant_nodes = set()
    for state in irrelevant_states:
        node = cat.find_node_in_tree([], cat._root, state)[1]
        irrelevant_nodes.add(node)

    # delete all the irrelevant nodes in the CAT
    for node in irrelevant_nodes:
        temp_node = node
        while temp_node._state not in relevant_states:
            parent = temp_node._parent
            for node in parent._child:
                if node._state == temp_node._state:
                    parent._child.remove(temp_node)
            if temp_node._state in cat._leaves:
                del cat._leaves[temp_node._state]
            temp_node = parent
    graph = utils.get_networkx_cat(cat._root)
    nx.nx_pydot.write_dot(graph, directory+"/relevant_cat_"+str(concrete_state)+".dot")
    return cat

def argmax_rand_tie_breaker(data):
    max_value = np.max(data)
    max_indices = []
    for i in range(len(data)):
        if np.allclose(data[i], max_value):
            max_indices.append(i)
    index = random.randint(0,len(max_indices)-1)
    res = max_indices[index]
    return res

def fallsWithin_global(con_state, abs_state, root):
    for i,con_var in enumerate(con_state):
        if not float(abs_state[i].split(',')[0])<=con_var<float(abs_state[i].split(',')[1]):
            if not con_var==float(abs_state[i].split(',')[1]):
                return False
            elif not float(abs_state[i].split(',')[1]) == float(root._state[i].split(',')[1]):
                return False
    return True

def custom_sort_key(item):
    tuple_values = tuple(map(float, item[0].split(',')))
    return tuple_values