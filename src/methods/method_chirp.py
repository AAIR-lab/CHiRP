import copy
import os
import shutil
import networkx as nx

from src.misc import utils
from src.agent.learning import *
from src.data_structures.option import Option
from src.misc.log import Log_experiments
from src.methods.method_catrl import CATRL
from src.option_invention import options_planner
import hyper_param

class CHiRP:
    def __init__(self, env, env_id, agent, agent_con, abstract, option_library, log, directory_path, file_name, prev_eval_log):
        self.env = env
        self.env_id = env_id
        self.agent = agent
        self.agent_con = agent_con
        self.abstract = abstract
        self.log = log
        self.directory_path = directory_path
        self.heatmap_directory_path = directory_path+"/heatmaps"
        self.file_name = file_name
        self.prev_option_library = option_library
        self.option_library = {}
        for option in option_library:
            self.option_library[option.optionid] = option
        self.new_options = []
        self.option_unreachable_count = {}
        self.training_reward = 0
        self.training_steps = 0
        self.training_success = 0
        self.fixed_plan = []
        self.iteration_info = {}
        self.ignore_options = {}

        self.total_timesteps = hyper_param.total_timesteps
        self.episode_max = hyper_param.episode_max
        self.stepmax = hyper_param.step_max
        self.epsilon = hyper_param.epsilon
        self.evaluate_interval = hyper_param.evaluate_interval
        self.update_abs_interval = hyper_param.update_abs_interval

        self.option_decay = hyper_param.option_decay
        self.goal_option_decay = hyper_param.goal_option_decay
        self.option_k_cap = hyper_param.option_k_cap
        # self.goal_option_k_cap = hyper_param.goal_option_k_cap
        self.option_training_epi_threshold = hyper_param.option_training_epi_threshold
        
        self.catrl = CATRL(self.env, self.agent, self.agent_con, self.abstract, self.log, self.directory_path, self.file_name, prev_eval_log)
        self.catrl.episode_max = self.episode_max
        self.catrl.total_timesteps = self.total_timesteps

    def initialize_problem(self, args, stepmax, env_id):
        self.env.initialize_problem(args, stepmax)
        self.env_id = env_id
        self.step_max = self.env._step_max
        self.env_directory_path = self.directory_path+"/env_"+str(self.env_id)
        self.catrl.initialize_problem(args, stepmax, env_id)
        if not os.path.exists(self.env_directory_path+"/"):
            os.makedirs(self.env_directory_path+"/")
        self.heatmap_directory_path = self.env_directory_path+"/heatmaps"
        # if not os.path.exists(self.heatmap_directory_path+"/"):
        if os.path.exists(self.heatmap_directory_path+"/"):
            shutil.rmtree(self.heatmap_directory_path+"/")
        os.makedirs(self.heatmap_directory_path+"/")
        self.initialize_transition_library()

    def initialize_transition_library(self):
        current_path = copy.deepcopy(self.env_directory_path)
        prev_env_directory_path = current_path.split("env_")[0] + "env_" + str(self.env_id-1)
        f = open(prev_env_directory_path+"/transition_library.pickle", "rb")
        # f = open(prev_env_directory_path+"/transitions.pickle", "rb")
        transition_library = pickle.load(f)
        self.transition_library = transition_library['abstract_transitions']

    def compute_plan(self, state_abs, threshold, invent_bridge_option, ignore_options, replanning=False):
        cat = self.abstract._tree
        for optionid in copy.deepcopy(list(self.option_library.keys())):
            if len(self.option_library[optionid].termination_set) == 0:
                del self.option_library[optionid]
        new_options = {}
        for opid, op in copy.deepcopy(self.option_library).items():
            if opid not in ignore_options:
                new_options[opid] = op

        # utils.plot_option_path(list(new_options.values()), cat, self.env, self.env_id, self.env_directory_path, filename="option_library")
        goal_states_abs = [self.abstract.state(self.env._goal_state)]

        search = options_planner.OptionsPlanner(new_options, cat, self.transition_library, invent_bridge_option=invent_bridge_option, replanning=replanning)
        chain_options = search.get_plan_over_options(state_abs, goal_states_abs)

        learned_options = []
        for option in chain_options:
            if option._qtable._qtable!={}: learned_options.append(1)
            else: learned_options.append(0)

        print("plan:",chain_options)
        return chain_options, learned_options

    def main(self, final_steps, abs_index):
        iteration = 0
        success_rate = 0
        final_steps = 0
        new_catoptions = []
        while success_rate < 0.9 and final_steps < self.total_timesteps:
            print(f"\n########################  iteration {iteration} ###########################\n")
            self.iteration_info[iteration] = {}
            iteration_success, final_steps, success_plan, new_catoptions, abs_index, fallback = self.main_iteration(final_steps, abs_index, iteration, new_catoptions)
            if fallback:
                return final_steps, abs_index, fallback
            self.iteration_info[iteration]["success"] = iteration_success
            self.iteration_info[iteration]["final_steps"] = final_steps
            self.iteration_info[iteration]["new_catoptions"] = copy.deepcopy(new_catoptions)
            self.iteration_info[iteration]["abstract"] = copy.deepcopy(self.abstract)
            self.iteration_info[iteration]["success_plan"] = copy.deepcopy(success_plan)
            print(f"#### iteration_success {iteration_success} in steps {final_steps}")
            print(f"#### Now evaluating...")
            success_rate, trajectories_data, local_stg = self.evaluate_plan(iteration, success_plan, final_steps)
            print(f"#### Result: Env {self.env_id} iteration_success {success_rate} in final_steps {final_steps}")
            iteration += 1
            if iteration == 10:
                fallback = True
                return final_steps, abs_index, fallback

        
        last_iteration = len(self.iteration_info)-1
        new_options = self.iteration_info[last_iteration]["new_catoptions"]
        with open(self.env_directory_path+"/options.pickle","wb") as file:
            pickle.dump(new_options, file)

        self.abstract = self.iteration_info[last_iteration]["abstract"]
        self.abstract.plot_cat(self.env_directory_path)

        if hyper_param.plot_heatmaps:
            init_state = self.env._init_state
            for option in self.new_invented_options:
                qtable = {}
                for key,value in option._qtable.items():
                    if key in option.cat._leaves:
                        qtable[str(key)] = value
                self.abstract.plot_heatmaps(init_state, self.env._goal_state, self.heatmap_directory_path, index=0, qtable=qtable, optionid=option.optionid)
        utils.plot_option_path(list(self.option_library.values()), self.abstract._tree, self.env, self.env_id, self.env_directory_path, filename="new_option_library")
        utils.plot_option_path(list(new_options), self.abstract._tree, self.env, self.env_id, self.env_directory_path, plot_individually=False, filename="succ_option_plan")
        return final_steps, abs_index, fallback


    def main_iteration(self, final_steps, abs_index, iteration, new_catoptions):
        fallback = False
        state = self.env._init_state
        initial_state = copy.deepcopy(self.env._init_state)
        initial_abstract_state = self.abstract.state(state)
        current_state_abs = copy.deepcopy(initial_abstract_state)
        success_plan = []
        initialplan = []
        iteration_success = False
        replan = False

        while final_steps < self.total_timesteps:
            # 1. compute plan from current abstract state
            if iteration == 0 and state == initial_state:
                invent_bridge_option = True
                plan, learned_options = self.compute_plan(current_state_abs, threshold=1, invent_bridge_option=invent_bridge_option, ignore_options=self.ignore_options, replanning=replan)
                plan_found = True if len(plan) > 0 else False
                initialplan = copy.deepcopy(plan)
                print(f"\nComputed plan: {plan} \ninitialstate {self.env._init_state} goalstate {self.env._goal_state}") 
                print(f"Learned options: {learned_options}")
            elif iteration==0 and replan:
                invent_bridge_option = False
                plan, learned_options = self.compute_plan(current_state_abs, threshold=1, invent_bridge_option=invent_bridge_option, ignore_options=self.ignore_options, replanning=replan)
                plan_found = True if len(plan) > 0 else False
                print(f"\nComputed plan: {plan} \ninitialstate {self.env._init_state} goalstate {self.env._goal_state}") 
                print(f"Learned options: {learned_options}")
            elif state == initial_state:
                plan = copy.deepcopy(self.fixed_plan)
                plan_found = True if len(plan) > 0 else False
                print(f"\nLearned plan: {plan} \ninitialstate {self.env._init_state} goalstate {self.env._goal_state}")

            if iteration == 0 and state == initial_state and not plan_found:
                fallback = True
                return iteration_success, final_steps, success_plan, new_catoptions, abs_index, fallback
            
            if plan_found:
                for i in range(len(plan)):
                    option = plan[i]
                    finetune = False

                    # 2. Execute option if succ prob > 0.9 else finetune.
                    if not option._is_bridge_option:
                        print(f"\nSimulating option {option} from state {state}")
                        print(f"initialstate {self.env._init_state} goalstate {self.env._goal_state} state{state}") 
                        print(f"Plan: {initialplan}") 
                        print(f"Success plan: {success_plan}")
                        succ_prob, next_state = self.simulate_option(option, state)
                        if succ_prob > 0.9:
                            state = copy.deepcopy(next_state)
                            current_state_abs = self.abstract.state(state)
                            success_plan.append(option)
                            print(f"Successfully executed option: {option} success: {succ_prob} \nstate:{state}")
                        else:
                            finetune = True

                    # 3. Learn or finetine policy for option. If fails, then replan from current abstract state.
                    # 4. add to option library, update CAT, update representations for all options.
                    if (option._is_bridge_option or finetune) and final_steps < self.total_timesteps:
                        print(f"\nLearning/finetuning option {option} from state {state}")
                        print(f"initialstate {self.env._init_state} goalstate {self.env._goal_state} state {state}") 
                        print(f"Plan: {initialplan}") 
                        print(f"Success plan: {success_plan}")
                        new_options, cat, updated_abstractions, next_state, success, final_steps, couldnt_learn = self.learn_option(option, state, abs_index, final_steps, iteration)
                        if success:
                            state = copy.deepcopy(next_state)
                            current_state_abs = self.abstract.state(state)
                            success_plan.extend(new_options)
                            print(f"Successully learned: {new_options} \nstate: {state}")
                            print(f"initialstate {self.env._init_state} goalstate {self.env._goal_state} state {state}") 
                            print(f"Plan: {initialplan}") 
                            print(f"Success plan: {success_plan}")
                            if option._is_bridge_option:
                                self.abstract._tree = copy.deepcopy(cat)
                                new_catoptions.extend(new_options)
                                new_catoptions = self.update_option_abstractions(updated_abstractions, new_catoptions)
                                success_plan = self.update_option_abstractions(updated_abstractions, success_plan)  
                        elif iteration == 0:
                            replan = True
                            self.ignore_options[option.optionid] = option
                            break

                if not replan:
                    break
                if option._is_bridge_option and couldnt_learn:
                    break
            else:
                break
        
        # 4. If plan is not found, create option from current state to goal states.
        # 4. add to option library, update CAT, update representations for all options.
        if final_steps < self.total_timesteps and not self.env.is_goal_state(state):
            goal_option = self.build_goal_option_set()
            print(f"\nLearning goal_option {goal_option} from state: {state}")
            print(f"initialstate {self.env._init_state} goalstate {self.env._goal_state} state {state}") 
            print(f"Plan: {initialplan}") 
            print(f"Success plan: {success_plan}")
            new_options, cat, updated_abstractions, next_state, success, final_steps, couldnt_learn = self.learn_option(goal_option, state, abs_index, final_steps, iteration)
            if success:
                iteration_success = True
                state = copy.deepcopy(next_state)
                current_state_abs = self.abstract.state(state)
                success_plan.extend(new_options)
                print(f"Successully learned: {new_options} \nstate_reached: {state}")
                print(f"initialstate {self.env._init_state} goalstate {self.env._goal_state} state {state}") 
                print(f"Plan: {initialplan}") 
                print(f"Success plan: {success_plan}")
                for op in new_options:
                    if op.cost >= 10 and len(op.termination_set)>0:
                        new_catoptions.append(op)
                        self.abstract._tree = copy.deepcopy(cat)
                    new_catoptions = self.update_option_abstractions(updated_abstractions, new_catoptions)
                    success_plan = self.update_option_abstractions(updated_abstractions, success_plan)

        if self.env.is_goal_state(state):
            iteration_success = True
            self.fixed_plan = copy.deepcopy(success_plan)
        if iteration == 0 and not iteration_success:
            self.fixed_plan = []

        return iteration_success, final_steps, success_plan, new_catoptions, abs_index, fallback


    def learn_option(self, option, init_state, abs_index, final_steps, iteration):
        # 1. seed policy and set hyperparams for option
        if option._is_goal_option:
            if len(self.fixed_plan) == 0:
                dont_break = False
            else:
                dont_break = True
            epsilon = self.epsilon
            option_decay = self.option_decay
            option_stepmax = self.stepmax #self.goal_option_stepmax
            option_k_cap = self.abstract._k #self.goal_option_k_cap 
            option_update_abs_interval = self.update_abs_interval #self.goal_option_update_abs_interval
            option_evaluate_interval =  self.evaluate_interval #self.goal_option_evaluate_interval
            option_training_epi_threshold = self.episode_max
        elif option._is_bridge_option:
            dont_break = False
            epsilon = self.epsilon
            option_decay = self.option_decay
            option_stepmax = self.stepmax
            option_k_cap = self.abstract._k
            option_update_abs_interval = self.update_abs_interval
            option_evaluate_interval = self.evaluate_interval
            option_training_epi_threshold = self.option_training_epi_threshold
        else: #finetune
            dont_break = True
            epsilon = 0.4
            option_decay = 0.9995
            option_stepmax = self.stepmax #self.option_stepmax
            option_k_cap = self.option_k_cap
            option_update_abs_interval = self.update_abs_interval
            option_evaluate_interval = self.evaluate_interval
            option_training_epi_threshold = self.option_training_epi_threshold

        option_env = copy.deepcopy(self.env)
        option_env._train_option = option
        option_agent = IntraOptionAbstractAgent(option_env, epsilon=epsilon, decay=option_decay)
        if not option._is_goal_option and not option._is_bridge_option: #finetune
            option_agent._qtable = self.seed_qtable(option)
        if hyper_param.continuous_state:
            option_agent_con = None
        else:
            option_agent_con = Agent(option_env, decay=self.agent_con._decay)
        option_abstract = copy.deepcopy(self.abstract)
        option_abstract._env = option_env
        option_abstract._agent = option_agent
        option_abstract._agent_concrete = option_agent_con
        option_abstract._k = option_k_cap
        option_agent._abstract = option_abstract
        option_log = Log_experiments()
        
        # 2. CAT+RL for option
        catrl = CATRL(option_env, option_agent, option_agent_con, option_abstract, option_log, self.directory_path)
        catrl.option_training_epi_threshold = option_training_epi_threshold
        catrl.update_abs_interval = option_update_abs_interval
        catrl.evaluate_interval=option_evaluate_interval
        catrl.total_timesteps=self.total_timesteps
        args = copy.deepcopy(hyper_param.problem_args[self.env_id])
        args["init_vars"] = init_state
        catrl.initialize_problem(args, option_stepmax, self.env_id)
        catrl.optionid = option.optionid

        trajectories_data = {}
        trajectories_data["trajectories"] = []
        new_options = []
        updated_abstractions = []
        couldnt_learn = False
        success = False
        i = 0
        print(f"\n----------------------- Env id:{self.env_id} iteration {iteration} option_learning optionid: {option.optionid} {option.termination_set} -----------------------")
        while not couldnt_learn and i < self.episode_max and final_steps < self.total_timesteps:
            init_state = self.env.reset()
            i, final_steps, abs_index, learn_options, couldnt_learn, trajectories_data, success_state = catrl.catrl(i, final_steps, abs_index, mode = "\toption_learning")
            success_state = None
            if learn_options:
                while success_state is None:
                    _, _, _, success_state = catrl.evaluate_policy(i, abs_index, n_eval=1)
                success = True
                # 3. invent options 
                new_options, updated_abstractions, trajectories_data, final_steps = catrl.learn_catoptions(catrl.env, catrl.env_id, catrl.env_directory_path, catrl.abstract._tree, trajectories_data, catrl.agent._qtable, final_steps=final_steps, abs_index=abs_index, dont_break=dont_break, qtable_con=None)
                break
            elif final_steps > self.total_timesteps:
                couldnt_learn = True
                print('Oops, our approach could not learn within given timesteps /\(-|-)/\\')
                break
   
        cat = option_abstract._tree  
        if learn_options:
            for op in new_options:
                op.cat = option_abstract._tree

        print("\tSuccess rate:", catrl.log.latest_eval_success(), " in episodes:",i, " in timesteps ", final_steps)
        print(f"----------------------- Env id: {self.env_id} iteration {iteration} Ended option_learning optionid: {option.optionid} {[op.termination_set for op in new_options]} -----------------------\n")
        return new_options, cat, updated_abstractions, success_state, success, final_steps, couldnt_learn


    def update_option_abstractions(self, updated_abstractions, options):
        if len(updated_abstractions) > 0:
            for i in range(len(options)):
                option = options[i]
                option = utils.update_option_sets(option, updated_abstractions)
                options[i] = utils.update_options_to_single_term(self.abstract, option)
        return options
    
    def update_option_library_abstractions(self, updated_abstractions):
        if len(updated_abstractions) > 0:
            for i in self.option_library.keys():
                option = self.option_library[i]
                option = utils.update_option_sets(option, updated_abstractions)
                option = utils.update_options_to_single_term(self.abstract, option) 

    def seed_qtable(self, option):
        old_qtable = option._qtable
        new_qtable = Qtable()

        old_leaves = set() #TODO: adding new
        for s in option.policy_states:
            if s in option.cat._leaves:
                old_leaves.add(s)

        updated_abstractions = {}
        for old_leaf in old_leaves:
            if old_leaf not in self.abstract._tree._leaves:
                _, node = self.abstract._tree.find_node_in_tree([], self.abstract._tree._root, old_leaf)
                if node is not None:
                    new_leaves = [temp_node._state for temp_node in self.abstract._tree.get_all_leafs(node)]
                    updated_abstractions[old_leaf] = new_leaves
            else:
                updated_abstractions[old_leaf] = [old_leaf]

        for old_abs in updated_abstractions:
            if old_abs in old_qtable._qtable:
                for new_abs in updated_abstractions[old_abs]:
                    new_qtable._qtable[new_abs] = old_qtable._qtable[old_abs]

        return new_qtable

    def build_goal_option_set(self):
        initiation_set = set([])
        termination_set = set([])
        new_option = Option(initiation_set, termination_set, optionid=len(self.option_library))
        new_option._is_goal_option = True
        return new_option
    
    def execute_option(self, option, state, option_env=None, steps=0, print_info=False, mode='learning'):
        state_abs = option.state(state)
        reward = 0
        steps = steps
        goal_success = False
        option_success = False
        traj = []
        abs_traj = []

        if (option._is_goal_option and self.env.is_goal_state(state)):
            goal_success = True
            option_success = True
            return state, state_abs, steps, reward, goal_success, option_success, traj, abs_traj
        elif (not option._is_goal_option and option.is_termination_set(state_abs)):
            option_success = True
            return state, state_abs, steps, reward, goal_success, option_success, traj, abs_traj
        
        if option_env is None:
            self.env._train_option = None
        # if print_info:
        #     print("Executing option",option.termination_set)
        while steps < self.env._step_max:
            # get policy prescribed by the option
            action = option.policy(state)
            if action is None: # falling out of option
                print("Falling out of option as no action prescribed...")
                break

            # execute policy prescribed by the option in the env
            if option_env is None:
                new_state, temp_r, _, success = self.env.step(action)
            else:
                new_state, temp_r, _, success = option_env.step(action)        
            goal_success = success["success"]
            # if print_info:
            #     print(state, option.state(state), action, new_state, option.state(new_state))
            reward += temp_r
            steps += 1

            # update state and state_abs
            traj.append([copy.deepcopy(state), copy.deepcopy(action), copy.deepcopy(new_state)])
            abs_traj.append([copy.deepcopy(state_abs), copy.deepcopy(action), copy.deepcopy(self.abstract.state(new_state))])
            state = copy.deepcopy(new_state)
            state_abs = option.state(new_state)
            if mode == 'evaluation' and goal_success:
                break
            # find if option's termination is reached
            if (option._is_goal_option and self.env.is_goal_state(state)):
                option_success = True
                break
            elif (not option._is_goal_option and option.is_termination_set(state_abs)):
                option_success = True
                break
        return state, state_abs, steps, reward, goal_success, option_success, traj, abs_traj

    def simulate_option(self, option, initstate):
        succ_prob = 0
        next_state = copy.deepcopy(initstate)
        for i in range(100):
            option_env = copy.deepcopy(self.env)
            option_args = copy.deepcopy(hyper_param.problem_args[self.env_id])
            option_args["init_vars"] = initstate
            option_env.initialize_problem(option_args, self.step_max)
            state, state_abs, steps, reward, goal_success, option_success, traj, abs_traj = self.execute_option(option, initstate)
            succ_prob += int(option_success)
            if option_success:
                next_state = copy.deepcopy(state)
        return succ_prob/100.0, next_state

    def evaluate_plan(self, iteration, chain_options, final_steps):
        print("\n\n----------------------- End id: {} Executing plan of options....-----------------------".format(self.env_id))
        self.log.log_transfer_info(self.env_id, final_steps, self.abstract._n_abstract_states, total_options=len(self.option_library), useful_options=len(chain_options))
        reward_list = []
        success_list = []
        steps_list = []
        trajectories = []
        abstract_trajectories = []
        local_stg = {}
        print("Plan",chain_options)
        total_success = 0.0
        success_rate = 0.0
        eval_epi_i = 0
        n_eval = 100
        if len(chain_options) > 0: #TODO: adding new
            chain_options[-1].is_goal_option = True
        while eval_epi_i < n_eval:
            eval_epi_i += 1
            mode = "optioncats"
            traj = []
            abs_traj = []
            args = copy.deepcopy(hyper_param.problem_args[self.env_id])
            self.env.initialize_problem(args, self.env._step_max)
            state = self.env.reset()
            steps = 0
            reward  = 0
            init_state = copy.deepcopy(state)
            goal_success = False
            state_abs = self.abstract.state(state)
            for option in chain_options:
                init_state = copy.deepcopy(state)
                # print("\tinit state for option {} is {}".format(option.termination_set, state))
                state, state_abs, steps, reward, goal_success, option_success, op_traj, op_abs_traj = self.execute_option(option, init_state, steps=steps, print_info=True, mode='evaluation')
                if not option_success and not goal_success:
                    print(f"\t Could not reach option term: {option} from init_state: {init_state} state:{state}")
                    # for transition in op_traj:
                    #     print(transition)
                traj.extend(op_traj)
                abs_traj.extend(op_abs_traj)
                reward += reward
                if goal_success:
                    trajectories.append(copy.deepcopy(traj))
                    abstract_trajectories.append(copy.deepcopy(abs_traj))
                    break
                # if not option_success:
                #     assert False
            reward_list.append(reward)
            success_list.append(int(goal_success))
            steps_list.append(steps)
            total_success += int(goal_success)
            print("Env: {} {} episode: {}  final_steps: {} #abs: {}  success: {} steps: {} reward: {} state: {}".format(self.env_id, mode, eval_epi_i, final_steps, self.abstract._n_abstract_states, int(goal_success), steps, reward, state))
        success_rate = total_success/n_eval
        self.log.log_transfer_eval(iteration, reward_list, success_list, steps_list, success_rate)
        self.log.save_transfer_result(self.env_directory_path)
        print("\n\nSuccess rate achieved is {}".format(success_rate))
        print("final_steps:",final_steps)
        if len(chain_options) > 0: #TODO: adding new
            chain_options[-1].is_goal_option = False
        trajectories_data = {}
        trajectories_data["trajectories"] = trajectories
        trajectories_data["abstract_trajectories"] = abstract_trajectories
        for traj in abstract_trajectories:
            for transition in traj:
                state1, action, state2 = transition[0], transition[1], transition[2]
                if state1 not in local_stg:
                    local_stg[state1] = {}
                if state2 not in local_stg[state1]:
                    local_stg[state1][state2] = {}
                    local_stg[state1][state2]["label"] = []
                if action not in local_stg[state1][state2]["label"]:
                    local_stg[state1][state2]["label"].append(action)
        return success_rate, trajectories_data, local_stg

        


