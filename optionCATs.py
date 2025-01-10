import numpy as np
import random 
import time
import os, copy, sys
import shared_args

shared_args.method = sys.argv[1]
shared_args.domain = sys.argv[2]
shared_args.trial = sys.argv[3]

import hyper_param
from src.agent.learning import *
from src.misc.log import Log_experiments
from src.abstraction.abstraction import Abstraction 
from src.data_structures.option import Option
from src.methods.method_catrl import CATRL
from src.methods.method_chirp import CHiRP
from src.abstraction.update_option_library import *
from src.abstraction.update_transition_library import *

random.seed(78*hyper_param.trial + 13)
np.random.seed(78*hyper_param.trial + 13)


def delete_previous_large_files(env_id):
    paths = []
    files_to_delete = ["local_stg.pickle","stgs.pickle","optionNone_OptionCATs_evaluation.pickle","optionNone_OptionCATs_learning.pickle","trajectories.pickle","transitions.pickle","transition_library.pickle","qtable.pickle"]
    for file in files_to_delete:
        paths.append(hyper_param.directory+"/env_"+str(env_id)+"/"+file)
    for path in paths:
        if os.path.exists(path):
            os.remove(path)

def reuse_options(options_path, cat, action_size, state_mapping=None):
    options = []
    with open(options_path,'rb') as file:
        options_pkl = pickle.load(file)
    for optionid, option in options_pkl.items():
        if state_mapping is not None:
            new_initiation_set = set()
            for state in option.initiation_set:
                if state in state_mapping:
                    new_state = state_mapping[state]
                    new_initiation_set.add(new_state)
            new_termination_set = set()
            for state in option.termination_set:
                if state in state_mapping:
                    new_state = state_mapping[state]
                    new_termination_set.add(new_state)
        else:
            new_initiation_set = option.initiation_set
            new_termination_set = option.termination_set
        new_option = Option(new_initiation_set, new_termination_set, optionid=optionid, action_size=action_size)
        qtable = Qtable()
        for state, qval in option._qtable._qtable.items():
            if state_mapping is not None:
                if state in state_mapping:
                    new_state = state_mapping[state]
                    qtable._qtable[new_state] = qval
            else:
                qtable._qtable[state] = qval
        new_option.initialize_qtable(qtable)
        new_option.init_states = option.init_states
        new_option.term_states = option.term_states  
        new_option.cat = cat
        new_option.cost = option.cost
        new_option.abstract_cost = option.abstract_cost
        new_option.trajectories = option.trajectories
        if new_option._qtable != {}:
            new_option._is_bridge_option = False
        options.append(new_option)
    return options

def update_option_transition_libraries(env_id, env, option_path, env_directory, reuse_cat_path, log, abstract, first_env, final_steps, initial_abstract_varying):
    options_library = {}
    if not hyper_param.varying_map:                           
        if os.path.exists(option_path) and not first_env:
            prev_options = update_option_library(env_directory, option_path, reuse_cat_path, env_id)
            for optionid, option in prev_options.items():
                option.optionid = optionid
                options_library[optionid] = option
        update_transition_library(env_directory, env_id, log, abstract, first_env)
    else:
        # initial_abstract_varying.state_mapping  (cat1 - cat2 initial)    
        # two_abs_mapping (cat2 initial - cat2 final)     
        # find cat1 - cat2 final      
        if os.path.exists(option_path) and not first_env:
            prev_options = update_option_library_varying(env_directory, option_path, reuse_cat_path, env_id, initial_abstract_varying)
            options_library.extend(prev_options)
        update_transition_library(env_directory, env_id, log, abstract, first_env)

    if os.path.exists(env_directory+"/options.pickle"):
        with open(env_directory+"/options.pickle", "rb") as f:
            options = pickle.load(f)
            for option in options:
                optionid = env._action_size + len(options_library)
                option.optionid = optionid
                options_library[optionid] = option

    with open(env_directory+"/options_library.pickle","wb") as file:
        pickle.dump(options_library, file)
    with open(env_directory+'/final_steps.pickle', 'wb') as file:
        pickle.dump({env_id:final_steps}, file)

def learn_problems_continually():
    approach_name = 'optioncatplanrl_continual_transfer'
    start_problem_from = hyper_param.start_problem_from
    first_env = False
    fallback = False
    initial_abstract_varying = None
    if start_problem_from == 0:
        first_env = True
        fallback = True
    start_time = time.time()
    for env_id in range(start_problem_from, len(hyper_param.problem_args)):
        print("\n\n##################### Env {} ###########################".format(env_id))
        map_name = hyper_param.map_name
        file_name = hyper_param.map_name + "_" + approach_name + "_env" + str(env_id)
        env = hyper_param.get_env('optioncats', map_name, env_id)
        env_directory = hyper_param.directory+"/env_"+str(env_id)
        if not os.path.exists(env_directory+"/"):
            os.makedirs(env_directory+"/")
        log = Log_experiments()   

        if not hyper_param.continuous_state:
            agent_con = Agent(env, decay=hyper_param.decay_con)
        else:
            agent_con = None

        if hyper_param.method == "chirp": 
            agent = AbstractAgent(env, decay=hyper_param.decay)

        abstract = Abstraction(env = env, agent_con = agent_con, agent = agent, k_cap=hyper_param.k_cap, init_abs_level=hyper_param.init_abs_level, bootstrap = hyper_param.bootstrap, continuous_state=hyper_param.continuous_state)
        
        if first_env:
            reuse_cat_path = None
            abstract.initialize_tree()
        else:
            if hyper_param.varying_map:
                reuse_cat_path = hyper_param.directory+f"/env_{env_id-1}/cat.dot"
            else:
                reuse_cat_path = hyper_param.directory+f"/env_{env_id-1}/cat.pickle"
            abstract.reuse_tree(reuse_cat_path, env_directory, regenerate_tree=hyper_param.varying_map)
        abstract.update_n_abstract_state()
        agent._abstract = abstract

        option_path = hyper_param.directory+f"/env_{env_id-1}/options_library.pickle"
        eval_log = {}
        epi_i, final_steps = 0, 0
        abs_index = 0
        if os.path.exists(option_path) and not first_env:

            action_size = env._action_size
            if hyper_param.varying_map:
                options_library = reuse_options(option_path, abstract._tree, action_size, abstract.state_mapping)
                initial_abstract_varying = copy.deepcopy(abstract)
            else:
                options_library = reuse_options(option_path, abstract._tree, action_size, state_mapping=None)

            if hyper_param.method == "chirp":
                chirp = CHiRP(env, env_id, agent, agent_con, abstract, options_library, log, hyper_param.directory, file_name, eval_log)
                chirp.initialize_problem(hyper_param.problem_args[env_id], hyper_param.step_max, env_id)
                final_steps, abs_index, fallback = chirp.main(final_steps, abs_index)
        
        if first_env or fallback:
            agent._abstract._k = hyper_param.k_cap
            catrl = CATRL(env, agent, agent_con, abstract, log, hyper_param.directory, hyper_param.update_abs_interval, hyper_param.evaluate_interval, hyper_param.refine_levels, file_name='OptionCATs')
            catrl.initialize_problem(hyper_param.problem_args[env_id], hyper_param.step_max, env_id)
            epi_i, final_steps, abs_index = catrl.learn_cat_and_catoptions(hyper_param.problem_args[env_id], hyper_param.step_max, env_id, epi_i, final_steps, abs_index)

        update_option_transition_libraries(env_id, env, option_path, env_directory, reuse_cat_path, log, abstract, first_env, final_steps, initial_abstract_varying)
        first_env = False
        current_time = time.time()
        print("Time taken (s): ", round(current_time - start_time,2))


if __name__=="__main__":
    learn_problems_continually()
