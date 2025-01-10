import hyper_param
import pickle
import copy
import os
import random

random.seed(78*hyper_param.trial + 13)

def update_prev_transition_abstraction(prev_transition_data, transition_dict, abstract):
    for con_state in prev_transition_data['concrete_transitions']:
        if con_state not in transition_dict['concrete_transitions']:
            transition_dict['concrete_transitions'][con_state] = {}
        abs_state = abstract.state(con_state)
        if abs_state not in transition_dict['abstract_transitions']:
            transition_dict['abstract_transitions'][abs_state] = {}
        
        for con_next_state in prev_transition_data['concrete_transitions'][con_state]:
            abs_next_state = abstract.state(con_next_state)
            if abs_next_state not in transition_dict['abstract_transitions'][abs_state]:
                transition_dict['abstract_transitions'][abs_state][abs_next_state] = {'label':[]}
            if con_next_state not in transition_dict['concrete_transitions'][con_state]:
                transition_dict['concrete_transitions'][con_state][con_next_state] = {'label':[]}
            transition_dict['abstract_transitions'][abs_state][abs_next_state]['label'].extend(prev_transition_data['concrete_transitions'][con_state][con_next_state]['label'])
            transition_dict['concrete_transitions'][con_state][con_next_state]['label'].extend(prev_transition_data['concrete_transitions'][con_state][con_next_state]['label'])
    return transition_dict

def update_transition_library(env_directory, env_id, log, abstract, first_env):
    abs_transition_dict = {}
    if len(log.concrete_transition_dict)< hyper_param.max_transitions_to_store:
        concrete_transition_samples = log.concrete_transition_dict
    else:
        sampled_keys = random.sample(list(log.concrete_transition_dict.keys()), hyper_param.max_transitions_to_store)
        concrete_transition_samples = {key: log.concrete_transition_dict[key] for key in sampled_keys}

    for con_state in concrete_transition_samples:
        abs_state = abstract.state(con_state)
        if abs_state not in abs_transition_dict:
            abs_transition_dict[abs_state] = {}
        
        for con_next_state in concrete_transition_samples[con_state]:
            abs_next_state = abstract.state(con_next_state)
            if abs_next_state not in abs_transition_dict[abs_state]:
                abs_transition_dict[abs_state][abs_next_state] = {'label':[]}
            abs_transition_dict[abs_state][abs_next_state]['label'] = concrete_transition_samples[con_state][con_next_state]['label']

    transition_dict = {'concrete_transitions': concrete_transition_samples, 'abstract_transitions': abs_transition_dict}
    transition_path = env_directory + "/transitions.pickle"
    with open(transition_path, 'wb') as f:
        pickle.dump(transition_dict, f)
        
    if not hyper_param.varying_map:
        prev_transition_path = hyper_param.directory+f"/env_{env_id-1}/transition_library.pickle"
        prev_transition_data = {'concrete_transitions': {}, 'abstract_transitions': {}}
        if os.path.exists(prev_transition_path) and not first_env:
            with open(prev_transition_path, 'rb') as f:
                prev_transition_data = pickle.load(f)
        transition_library = update_prev_transition_abstraction(prev_transition_data, transition_dict, abstract)
    else:
        transition_library = copy.deepcopy(transition_dict)
        
    with open(env_directory+"/transition_library.pickle","wb") as file:
        pickle.dump(transition_library, file)
