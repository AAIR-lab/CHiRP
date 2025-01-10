import os
import numpy as np
from environments.envs.mazeworld_continuous import MazeWorldContinuous, MazeWorldContinuousEnvironment
from environments.envs.roomsworld_continuous import FourRoomsWorldContinuousEnvironment, FourRoomsWorldContinuous
from environments.envs.officeworld_continuous import OfficeWorldContinuous, OfficeWorldContinuousEnvironment
from environments.envs.minecraft_continuous import MineCraftEnvironment
from environments.envs.taxiworld_continuous import TaxiWorldContinuous, TaxiWorldContinuousEnvironment
import shared_args

domain = shared_args.domain
method = shared_args.method
trial = int(shared_args.trial)

np.random.seed(78*trial)
max_transitions_to_store = 10000
start_problem_from = 0
init_abs_level = 1
refine_levels = 1
refinement_method = "deliberative"
varying_map = False
print_info = True

if "_continuous" in domain:
    continuous_state = True
    plot_heatmaps = False
else:
    continuous_state = False
    plot_heatmaps = True


if domain == "maze_continuous":
    map_name = "maze_24x24"
    directory_map_name = map_name
    ancestor_dist_threshold_factor=0.95
    update_abs_interval = 100
    evaluate_interval = 100

    # for 24x24
    step_max = 500
    option_stepmax = 250 #only for optionRL
    adaptive_step_factor = 10

    epsilon_min = 0.05
    epsilon = 1.0
    epsilon_con = epsilon
    alpha = 0.05
    alpha_con = alpha
    gamma = 0.99
    k_cap = 5
    option_k_cap = 2

#     for 24x24
    decay = 0.997
    option_decay = 0.994
    goal_option_decay = 0.991
    decay_con = decay
    catrl_uai_decay = decay

    episode_max = 3000
    total_timesteps = 1500000
    option_training_epi_threshold = 500
    
    def get_env(approach_name, map_name, env_id):
        if approach_name == 'optioncats':
            env = MazeWorldContinuous(map_name)
        else:
            env = MazeWorldContinuousEnvironment(map_name, include_goal_variable=False)
        return env

    #     for 24x24
    problem_args = {
        0: { "init_vars": [23.01, 18.59], "goal_vars": [8, 22]},
        1: { "init_vars": [2.84, 19.12], "goal_vars": [2, 9]},
        2: { "init_vars": [15.54, 20.28], "goal_vars": [15, 4]},
        3: { "init_vars": [15.31, 1.13], "goal_vars": [14, 16]},
        4: { "init_vars": [23.02, 21.94], "goal_vars": [21, 5]},
        5: { "init_vars": [19.47, 1.31], "goal_vars": [5, 23]},
        6: { "init_vars": [18.73, 11.19], "goal_vars": [5, 6]},
        7: { "init_vars": [2.23, 21.36], "goal_vars": [21, 21]},
        8: { "init_vars": [19.61, 12.90], "goal_vars": [2, 4]},
        9: { "init_vars": [2.05, 10.03], "goal_vars": [21, 9]},
        10: { "init_vars": [0.84, 18.34], "goal_vars": [17, 9]},
        11: { "init_vars": [7.74, 7.05], "goal_vars": [19, 10]},
        12: { "init_vars": [16.21, 1.35], "goal_vars": [21, 18]},
        13: { "init_vars": [22.42, 18.16], "goal_vars": [15, 4]},
        14: { "init_vars": [19.71, 19.63], "goal_vars": [10, 6]},
        15: { "init_vars": [0.06, 11.22], "goal_vars": [12, 17]},
        16: { "init_vars": [1.69, 19.14], "goal_vars": [4, 11]},
        17: { "init_vars": [16.70, 14.08], "goal_vars": [0, 4]},
        18: { "init_vars": [11.41, 2.85], "goal_vars": [14, 20]},
        19: { "init_vars": [0.92, 4.99], "goal_vars": [13, 15]},

    }
    bootstrap = 'from_estimated_concrete'
    
if domain == "fourrooms_continuous":
    map_name = "rooms_33x33"
    directory_map_name = map_name
    ancestor_dist_threshold_factor=0.95
    update_abs_interval = 100
    evaluate_interval = 100

    step_max = 800      #for 33x33
    k_cap = 5
    option_k_cap = 2 # only used when finetuning
    adaptive_step_factor = 10

    epsilon_min = 0.05
    epsilon = 1.0
    epsilon_con = epsilon
    alpha = 0.05
    alpha_con = alpha
    gamma = 0.999
    k_cap = 5
    decay = 0.998
    catrl_uai_decay = 0.9991
    decay_con = decay
    option_decay = decay
    catrl_uai_decay = decay
    goal_option_decay = decay

    episode_max = 3000
    total_timesteps = 2000000
    option_training_epi_threshold = 500

    def get_env(approach_name, map_name, env_id):
        if approach_name == 'optioncats':
            env = FourRoomsWorldContinuous(map_name)
        else:
            env = FourRoomsWorldContinuousEnvironment(map_name)
        return env
    problem_args = {
                0: { "init_vars":[26.02, 22.17], "goal_vars":[10, 2]}, 
                1: { "init_vars":[13.13, 0.75], "goal_vars":[26, 14]}, 
                2: { "init_vars":[0.78, 5.53], "goal_vars":[2, 32]}, 
                3: { "init_vars":[0.31, 6.41], "goal_vars":[14, 15]}, 
                4: { "init_vars":[26.29, 5.13], "goal_vars":[29, 23]}, 
                5: { "init_vars":[19.85, 4.46], "goal_vars":[10, 27]}, 
                6: { "init_vars":[3.16, 1.76], "goal_vars":[25, 5]}, 
                7: { "init_vars":[28.64, 31.06], "goal_vars":[2, 11]}, 
                8: { "init_vars":[17.13, 27.47], "goal_vars":[0.4, 24]}, 
                9: { "init_vars":[2.65, 9.98], "goal_vars":[20, 26]}, 
                10: { "init_vars":[29.35, 2.12], "goal_vars":[1, 9]}, 
                11: { "init_vars":[5.45, 9.19], "goal_vars":[25, 22]}, 
                12: { "init_vars":[7.62, 13.07], "goal_vars":[31, 14]}, 
                13: { "init_vars":[11.28, 30.03], "goal_vars":[30, 20]}, 
                14: { "init_vars":[31.16, 18.40], "goal_vars":[23, 3]}, 
                15: { "init_vars":[29.25, 19.42], "goal_vars":[0.4, 21]}, 
                16: { "init_vars":[10.42, 16.47], "goal_vars":[12, 23]}, 
                17: { "init_vars":[7.17, 3.46], "goal_vars":[0.5, 30]}, 
                18: { "init_vars":[24.89, 21.53], "goal_vars":[18, 5]}, 
                19: { "init_vars":[13.73, 25.29], "goal_vars":[23, 13]}, 
            }
    bootstrap = 'from_estimated_concrete'

if domain == "minecraft_continuous":
    ancestor_dist_threshold_factor=1.0
    update_abs_interval = 100
    evaluate_interval = 100
    goal_option_update_abs_interval = 100
    goal_option_evaluate_interval = 100
    goal_option_eval_success_threshold = 0.9

    map_name = "makeAxe"
    k_cap = 5
    option_k_cap = 2
    step_max = 1000
    option_stepmax = step_max
    goal_option_stepmax = step_max
    directory_map_name = map_name

    epsilon_min = 0.05
    epsilon = 1.0
    epsilon_con = epsilon
    alpha = 0.05
    alpha_con = alpha
    gamma = 1
    adaptive_step_factor = 10

    decay = 0.999
    catrl_uai_decay = 0.999
    decay_con = 0.999
    option_decay = 0.999
    goal_option_decay = decay

    episode_max = 3000
    total_timesteps = 3000000
    # bootstrap = 'from_concrete' #from_concrete for discrete domains
    bootstrap = 'from_estimated_concrete'
    option_training_epi_threshold = 200

    def get_env(approach_name, map_name, env_id):
        task = problem_args[env_id]["task"]
        return MineCraftEnvironment(map_name, task)

    # taxi 30x30 problems
    problem_args = {
     0: { "init_vars": [11.46, 12.29, 18, 20, 0, 0, 0, 0, 0], "locs": [4, 10, 12, 7, 3, 2, 2, 12], "goal_vars": [2, 12], "task":"makeIronAxe"},
     1: { "init_vars": [1.31, 7.12, 8, 11, 0, 0, 0, 0, 0], "locs": [4, 10, 12, 7, 3, 2, 2, 12], "goal_vars": [2, 12], "task":"makeStoneAxe"},
     2:{ "init_vars": [7.34, 19.46, 10, 18, 0, 0, 0, 0, 0], "locs": [4, 10, 12, 7, 3, 2, 2, 12], "goal_vars": [2, 12], "task":"makeIronAxe"},
    3:{ "init_vars": [15.40, 19.64, 18, 20, 0, 0, 0, 0, 0], "locs": [4, 10, 12, 7, 3, 2, 2, 12], "goal_vars": [2, 12], "task":"makeStoneAxe"},
    4:{ "init_vars": [3.52, 9.54, 18, 20, 0, 0, 0, 0, 0], "locs": [4, 10, 12, 7, 3, 2, 2, 12], "goal_vars": [2, 12], "task":"makeStoneAxe"},
    5:{ "init_vars": [8.04, 3.42, 8, 11,  0, 0, 0, 0, 0], "locs": [4, 10, 12, 7, 3, 2, 2, 12], "goal_vars": [2, 12], "task":"makeIronAxe"},
    6:{ "init_vars": [14.30, 18.27, 10, 18,  0, 0, 0, 0, 0], "locs": [4, 10, 12, 7, 3, 2, 2, 12], "goal_vars": [2, 12], "task":"makeStoneAxe"},
    7:{ "init_vars": [12.83, 11.53, 1, 3,  0, 0, 0, 0, 0], "locs": [4, 10, 12, 7, 3, 2, 2, 12], "goal_vars": [2, 12], "task":"makeIronAxe"},
    8:{ "init_vars": [20.92, 21.52, 18, 20,  0, 0, 0, 0, 0], "locs": [4, 10, 12, 7, 3, 2, 2, 12], "goal_vars": [2, 12], "task":"makeIronAxe"},
    9:{ "init_vars": [16.45, 19.03, 8, 11,  0, 0, 0, 0, 0], "locs": [4, 10, 12, 7, 3, 2, 2, 12], "goal_vars": [2, 12], "task":"makeStoneAxe"},
    10:{ "init_vars": [17.18, 14.49, 10, 18,  0, 0, 0, 0, 0], "locs": [4, 10, 12, 7, 3, 2, 2, 12], "goal_vars": [2, 12], "task":"makeIronAxe"},
    11:{ "init_vars": [19.41, 15.92, 1, 3,  0, 0, 0, 0, 0], "locs": [4, 10, 12, 7, 3, 2, 2, 12], "goal_vars": [2, 12], "task":"makeStoneAxe"},
    12:{ "init_vars": [19.21, 12.48, 18, 20,  0, 0, 0, 0, 0], "locs": [4, 10, 12, 7, 3, 2, 2, 12], "goal_vars": [2, 12], "task":"makeStoneAxe"},
    13:{ "init_vars": [8.25, 20.82, 8, 11,  0, 0, 0, 0, 0], "locs": [4, 10, 12, 7, 3, 2, 2, 12], "goal_vars": [2, 12], "task":"makeIronAxe"},
    14:{ "init_vars": [16.92, 4.69, 10, 18,  0, 0, 0, 0, 0], "locs": [4, 10, 12, 7, 3, 2, 2, 12], "goal_vars": [2, 12], "task":"makeStoneAxe"},
    15:{ "init_vars": [3.92, 2.73, 1, 3,  0, 0, 0, 0, 0], "locs": [4, 10, 12, 7, 3, 2, 2, 12], "goal_vars": [2, 12], "task":"makeIronAxe"},
    16:{ "init_vars": [12.1, 1.32, 18, 20,  0, 0, 0, 0, 0], "locs": [4, 10, 12, 7, 3, 2, 2, 12], "goal_vars": [2, 12], "task":"makeIronAxe"},
    17:{ "init_vars": [15.59, 17.98, 8, 11,  0, 0, 0, 0, 0], "locs": [4, 10, 12, 7, 3, 2, 2, 12], "goal_vars": [2, 12], "task":"makeStoneAxe"},
    18:{ "init_vars": [7.23, 10.10, 10, 18,  0, 0, 0, 0, 0], "locs": [4, 10, 12, 7, 3, 2, 2, 12], "goal_vars": [2, 12], "task":"makeIronAxe"},
    19:{ "init_vars": [7.19, 2.09, 1, 3,  0, 0, 0, 0, 0], "locs": [4, 10, 12, 7, 3, 2, 2, 12], "goal_vars": [2, 12], "task":"makeStoneAxe"},
}

if domain == "taxi_pass1_continuous":
    ancestor_dist_threshold_factor=1.0
    update_abs_interval = 100
    evaluate_interval = 100

    map_name = "taxi_30x30"
    k_cap = 5
    option_k_cap = 2 # only used when finetuning option
    adaptive_step_factor = 4

    step_max = 1000
    directory_map_name = map_name

    epsilon_min = 0.05
    epsilon = 1.0
    epsilon_con = epsilon
    alpha = 0.05
    alpha_con = alpha
    gamma = 1

    # for 30x30
    decay = 0.999 #0.997
    catrl_uai_decay = 0.9991
    decay_con = 0.999
    option_decay = 0.999
    goal_option_decay = decay

    episode_max = 4000
    total_timesteps = 4000000
    bootstrap = 'from_estimated_concrete' #from_concrete for discrete domains
    option_training_epi_threshold = 200

    def get_env(approach_name, map_name, env_id):
        if approach_name == 'optioncats':
            env = TaxiWorldContinuous(map_name, passenger_n=1)
        else:
            env = TaxiWorldContinuousEnvironment(map_name)
        return env

    problem_args = {
        0: { "init_vars": [27.2, 15.5, 2], "goal_vars": [3]},
        1: { "init_vars": [23.4, 11.7, 1], "goal_vars": [3]},
        2: { "init_vars": [0.6, 4.9, 2], "goal_vars": [4]},
        3: { "init_vars": [12.9, 6.2, 4], "goal_vars": [2]},
        4: { "init_vars": [27.8, 28.9, 1], "goal_vars": [2]},
        5: { "init_vars": [29.2, 16.5, 2], "goal_vars": [3]},
        6: { "init_vars": [11.7, 26.3, 1], "goal_vars": [4]},
        7: { "init_vars": [29.4, 28.6, 2], "goal_vars": [1]},
        8: { "init_vars": [26.8, 0.6, 2], "goal_vars": [4]},
        9: { "init_vars": [10.2, 26.5, 1], "goal_vars": [2]},
        10: { "init_vars": [0.3, 7.2, 4], "goal_vars": [1]},
        11: { "init_vars": [23.9, 2.8, 4], "goal_vars": [3]},
        12: { "init_vars": [9.4, 17.1, 3], "goal_vars": [2]},
        13: { "init_vars": [14.5, 8.6, 3], "goal_vars": [1]},
        14: { "init_vars": [14.1, 1.8, 2], "goal_vars": [3]},
        15: { "init_vars": [14.6, 13.4, 2], "goal_vars": [4]},
        16: { "init_vars": [9.7, 17.6, 2], "goal_vars": [1]},
        17: { "init_vars": [27.5, 1.6, 3], "goal_vars": [4]},
        18: { "init_vars": [26.7, 25.2, 1], "goal_vars": [3]},
        19: { "init_vars": [24.3, 29.8, 2], "goal_vars": [1]},
    }

if domain == "office_continuous":

    map_name = "office_11x15"
    ancestor_dist_threshold_factor=1.0
    refine_levels = 1
    update_abs_interval = 100
    evaluate_interval = 100
    goal_option_update_abs_interval = 100
    goal_option_evaluate_interval = 100
    goal_option_eval_success_threshold = 0.95
    directory_map_name = map_name

    step_max = 800
    option_stepmax = step_max
    goal_option_stepmax = step_max
    adaptive_step_factor = 10

    epsilon_min = 0.05
    epsilon = 1.0
    epsilon_con = 1.0
    alpha = 0.05
    alpha_con = 0.05
    gamma = 0.99
    k_cap = 5
    option_k_cap = k_cap
    decay = 0.9991
    option_decay = 0.999
    goal_option_decay = 0.999
    decay_con = decay
    catrl_uai_decay = decay

    episode_max = 4000
    total_timesteps = 4000000
    option_training_epi_threshold = 200

    bootstrap = 'from_estimated_concrete'

    def get_env(approach_name, map_name, env_id):
        if approach_name == 'optioncats':
            env = OfficeWorldContinuous(map_name)
        else:
            env = OfficeWorldContinuousEnvironment(map_name, include_goal_variable=False)
        return env
    
    problem_args = {0: {"init_vars": [9.12,2.86], "goal_vars": [5,5], }, 
            1: {"init_vars": [5.23,9.35], "goal_vars": [1,13], },
            2: {"init_vars": [1.91,2.03], "goal_vars": [10,2], },
            3: {"init_vars": [6.19,10.43], "goal_vars": [6,6], },
            4: {"init_vars": [8.01,4.49], "goal_vars": [5,9], },
            5: {"init_vars": [4.40,14.70], "goal_vars": [2,8], },
            6: {"init_vars": [2.48,12.16], "goal_vars": [6,2], },
            7: {"init_vars": [8.07, 9.50], "goal_vars":[0, 1], },
            8: {"init_vars": [8.23, 6.39], "goal_vars":[6, 2], },
            9: {"init_vars": [6.71, 12.64], "goal_vars":[8, 12], },
            10: {"init_vars": [6.31, 9.56], "goal_vars":[9, 8], },
            11: {"init_vars": [1.61, 5.66], "goal_vars":[10, 5], },
            12: {"init_vars": [0.84, 14.16], "goal_vars":[4, 8], },
            13: {"init_vars": [0.19, 10.41], "goal_vars":[0, 5], },
            14: {"init_vars": [6.77, 13.22], "goal_vars":[3, 9], },
            15: {"init_vars": [9.54, 8.41], "goal_vars":[4, 6], },
            16: {"init_vars": [0.37, 8.19], "goal_vars":[9, 0], },
            17: {"init_vars": [6.29, 1.21], "goal_vars":[2, 10], },
            18: {"init_vars": [1.06, 7.72], "goal_vars":[4, 0], },
            19: {"init_vars": [9.04, 7.93], "goal_vars":[1, 14],},
            }

basepath = os.getcwd()
directory = basepath + f"/results/{directory_map_name}/{method}/trial_{str(trial)}"
option_critic_directory = basepath + f"/results/{directory_map_name}/optioncritic/trial_{str(trial)}"

if method=="catrl" and continuous_state:
    refinement_method = "aggressive"
    bootstrap = "from_ancestor"
