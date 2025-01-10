import hyper_param
import pickle
import copy
import os
import random

from src.abstraction.abstraction import Abstraction
from src.misc import utils
from src.data_structures.qvalue_table import Qtable

def update_option_library(env_directory, option_path, reuse_cat_path, env_id):
    updated_options = {}
    with open(env_directory+"/cat.pickle", "rb") as f:
        latest_cat = pickle.load(f)

    if os.path.exists(option_path):
        with open(option_path, 'rb') as f:
            prev_options = pickle.load(f)
        
        more_abs_to_refined_abs = two_abs_mapping(reuse_cat_path, latest_cat, env_id)
        for optionid, option in prev_options.items():
            option.cat = latest_cat
            option_init_with_new_abs = []
            option_term_with_new_abs = []
            for abs_init_state in option.initiation_set:
                if abs_init_state in more_abs_to_refined_abs:
                    new_abs_init_states = more_abs_to_refined_abs[abs_init_state]
                    for new_abs_init_state in new_abs_init_states:
                        for transition in option.trajectories[0]:
                            if utils.fallsWithin_global(transition.state, new_abs_init_state, latest_cat._root):
                                option_init_with_new_abs.append(new_abs_init_state)
            option.initiation_set = set(option_init_with_new_abs)
            for abs_term_state in option.termination_set:
                if abs_term_state in more_abs_to_refined_abs:
                    new_abs_term_states = more_abs_to_refined_abs[abs_term_state]
                    for new_abs_term_state in new_abs_term_states:
                        for term_state in option.term_states:
                            if utils.fallsWithin_global(term_state, new_abs_term_state, latest_cat._root):
                                option_term_with_new_abs.append(new_abs_term_state)
                                break
            option.termination_set = set(option_term_with_new_abs)
            new_qtable = Qtable()
            for abs_state in option._qtable._qtable:
                if abs_state not in latest_cat._leaves:
                    if abs_state in more_abs_to_refined_abs:
                        new_abs_states = more_abs_to_refined_abs[abs_state]
                        for new_abs_state in new_abs_states:
                            new_qtable._qtable[new_abs_state] = option._qtable._qtable[abs_state]
                else:
                    new_qtable._qtable[abs_state] = option._qtable._qtable[abs_state]
            option._qtable = new_qtable
            if option._qtable._qtable != {}:
                option._is_bridge_option = False
        for optionid, option in prev_options.items():
            updated_options[optionid] = option
    return updated_options


def update_option_library_varying(env_directory, option_path, reuse_cat_path, env_id, initial_abstract_varying):
    updated_options = []
    with open(env_directory+"/cat.pickle", "rb") as f:
        latest_cat = pickle.load(f)
    state_mapping = initial_abstract_varying.state_mapping
    if os.path.exists(option_path):
        with open(option_path, 'rb') as f:
            prev_options = pickle.load(f)
        
        more_abs_to_refined_abs = two_abs_mapping_varying(initial_abstract_varying, latest_cat, env_id)
        for option in prev_options:
            option.cat = latest_cat
            option_init_with_new_abs = []
            option_term_with_new_abs = []
            for abs_init_state in option.initiation_set:

                if abs_init_state in state_mapping and state_mapping[abs_init_state] in more_abs_to_refined_abs:
                    new_abs_init_states = more_abs_to_refined_abs[state_mapping[abs_init_state]]
                    option_init_with_new_abs.extend(new_abs_init_states)
            option.initiation_set = set(option_init_with_new_abs)
            for abs_term_state in option.termination_set:
                if abs_term_state in state_mapping and state_mapping[abs_term_state] in more_abs_to_refined_abs:
                    new_abs_term_states = more_abs_to_refined_abs[state_mapping[abs_term_state]]
                    option_term_with_new_abs.extend(new_abs_term_states)
            option.termination_set = set(option_term_with_new_abs)
            new_qtable = {}
            for abs_state in option._qtable:
                if abs_state in state_mapping:
                    if state_mapping[abs_state] not in latest_cat._leaves:
                        if state_mapping[abs_state] in more_abs_to_refined_abs:
                            new_abs_states = more_abs_to_refined_abs[state_mapping[abs_state]]
                            for new_abs_state in new_abs_states:
                                new_qtable[new_abs_state] = option._qtable[abs_state]
                    else:
                        new_qtable[state_mapping[abs_state]] = option._qtable[abs_state]
            option._qtable = new_qtable
            if option._qtable != {}:
                option._is_bridge_option = False
        updated_options.extend(prev_options)
    return updated_options

def two_abs_mapping(reuse_cat_path, less_abs, env_id):
    more_abs_to_refined_abs = {}
    if hyper_param.varying_map:
        prev_map_id = hyper_param.problem_args[env_id-1]["map_id"]
        prev_map_name = hyper_param.map_params[prev_map_id]["map_name"]
        env = hyper_param.get_env('optioncats', prev_map_name, env_id-1)
    else:
        map_name = hyper_param.map_name
        env = hyper_param.get_env('optioncats', map_name, env_id)
    prev_abs = Abstraction(env, agent_con = None, agent = None, k_cap=hyper_param.k_cap, continuous_state=hyper_param.continuous_state)
    prev_abs.reuse_tree(reuse_cat_path, regenerate_tree=hyper_param.varying_map, plot_cat=False)
    for leaf in less_abs._leaves:
        if leaf not in prev_abs._tree._leaves:
            l = tuple(float(coord) for part in leaf for coord in part.split(','))
            random_concrete_leaf = tuple(random.uniform(l[i],l[i+1]) for i in range(0, len(l),2))
            new_abs = prev_abs.state(random_concrete_leaf)
        else:
            new_abs = leaf
        if new_abs not in more_abs_to_refined_abs:
            more_abs_to_refined_abs[new_abs] = []
        more_abs_to_refined_abs[new_abs].append(leaf)
    return more_abs_to_refined_abs

def two_abs_mapping_varying(initial_cat, less_abs,  env_id):
    more_abs_to_refined_abs = {}
    for leaf in less_abs._leaves:
        if leaf not in initial_cat._tree._leaves:
            l = tuple(int(coord) for part in leaf for coord in part.split(','))
            random_concrete_leaf = tuple(random.uniform(l[i],l[i+1]) for i in range(0, len(l),2))
            new_abs = initial_cat.state(random_concrete_leaf)
        else:
            new_abs = leaf
        if new_abs not in more_abs_to_refined_abs:
            more_abs_to_refined_abs[new_abs] = []
        more_abs_to_refined_abs[new_abs].append(leaf)
    return more_abs_to_refined_abs

def fallsWithin(con_state, abs_state, root):
    for i,con_var in enumerate(con_state):
        if not int(abs_state[i].split(',')[0])<=con_var<int(abs_state[i].split(',')[1]):
            if not con_var==int(abs_state[i].split(',')[1]):
                return False
            elif not int(abs_state[i].split(',')[1]) == int(root._state[i].split(',')[1]):
                return False
    return True

