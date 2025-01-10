import numpy as np
import copy
import os

from src.misc.log import Log_experiments
from src.agent.learning import *
from src.misc import utils
from src.option_invention import options_inventor
from src.data_structures.cat import CAT
from src.data_structures.trajectory import Sample, Transition, Trajectory

class CATRL:
    def __init__(self, env, agent, agent_con, abstract, log, directory_path, update_abs_interval=100, evaluate_interval=100, refine_levels=1, prev_eval_log={}, file_name = 'ThisShouldNotBeTheName'):
        self.env = env
        self.agent = agent
        self.agent_con = agent_con
        self.abstract = abstract
        self.log = log
        self.directory_path = directory_path
        self.file_name = file_name
        self.update_abs_interval_initial = copy.deepcopy(update_abs_interval)
        self.evaluate_interval_initial = copy.deepcopy(evaluate_interval)
        self.update_abs_interval = update_abs_interval
        self.evaluate_interval = evaluate_interval
        self.refine_levels = refine_levels
        self.optionid = None
        self.training_steps = 0
        self.cumulative_reward = 0.0
        self.cumulative_success = 0
        self.log_abs_eval = False
        self.agent.initialize_tderror()
        self.old_leaves = copy.deepcopy(list(self.abstract._tree._leaves.keys()))
        self.prev_eval_log = prev_eval_log
        self.last_avg_eval_success = 0.0
        self.couldnt_learn = False
        self.total_succ_trajectories = 0
        self.eval_success_threshold = 0.9
        self.too_small_option = False

        self.episode_max = hyper_param.episode_max
        self.option_training_epi_threshold = hyper_param.episode_max
        self.total_timesteps=hyper_param.total_timesteps
        self.adaptive_step_factor=hyper_param.adaptive_step_factor

    def initialize_problem(self, args, stepmax, env_id):
        self.env.initialize_problem(args, stepmax)
        self.env_id = env_id
        self.initial_step_max = self.env._step_max
        # self.step_max = self.env._step_max
        self.env_directory_path = self.directory_path+"/env_"+str(self.env_id)
        if not os.path.exists(self.env_directory_path+"/"):
            os.makedirs(self.env_directory_path+"/")
        self.heatmap_directory_path = self.env_directory_path+"/heatmaps"
        if not os.path.exists(self.heatmap_directory_path+"/"):
            os.makedirs(self.heatmap_directory_path+"/")

    def learn_cat_and_catoptions(self, args, stepmax, env_id, epi_i, final_steps, abs_index):
        self.initialize_problem(args, stepmax, env_id)
        self.trajectories_data = None
        directly_learn_options = False
        init_state = self.env.reset()
        if hyper_param.plot_heatmaps:
            self.abstract.plot_heatmaps(init_state, self.env._goal_state, self.heatmap_directory_path, abs_index, optionid=self.optionid)
        if directly_learn_options:
            learn_options = True
        while(final_steps < self.total_timesteps):
            if not directly_learn_options:
                init_state = self.env.reset()
                epi_i, final_steps, abs_index, learn_options, _, _, _ = self.catrl(epi_i, final_steps, abs_index)
            if learn_options:
                if not directly_learn_options:
                    self.abstract.plot_cat(self.env_directory_path)
                    self.log.save_qtable(self.agent._qtable, self.env_directory_path, self.optionid)
                    self.log.plot_save_performance(self.training_steps, self.env_directory_path, self.file_name, self.optionid)
                    self.abstract.plot_stgs(self.env_directory_path, abs_index, self.optionid)
                    # local_stg_graph = self.abstract.plot_local_stg(self.env_directory_path, abs_index, self.optionid)
                    # self.trajectories_data, _, _, success_state = self.evaluate_policy(epi_i, abs_index, n_eval=200)
                    self.log.save_trajectories(self.trajectories_data, self.env_directory_path, self.optionid)
                    # assert final_steps < self.total_timesteps, 'Oops, our approach could not learn within given timesteps /\(-|-)/\\'
                    _, _, _, final_steps = self.learn_catoptions(self.env, self.env_id, self.env_directory_path, cat=self.abstract._tree, trajectories_data=self.trajectories_data, qtable=self.agent._qtable, final_steps=final_steps, abs_index=abs_index)
                else:
                     _, _, _, final_steps = self.learn_catoptions(self.env, self.env_id, self.env_directory_path, final_steps=final_steps, abs_index=abs_index)
                break
        return epi_i, final_steps, abs_index

    def learn_catoptions(self, env, env_id, env_directory_path=None, cat=None, trajectories_data=None, qtable=None, dont_break=False, qtable_con=None, final_steps=0, abs_index=0):
        print("\nLearning options..")
        if env_directory_path is None:
            env_directory_path = self.env_directory_path
        if cat is None:
            f = open(env_directory_path+"/cat.pickle", "rb")
            cat_copy = pickle.load(f)
            cat = CAT([])
            cat.continuous_state = cat_copy.continuous_state
            cat._leaves = cat_copy._leaves
            cat._root_split = cat_copy._root_split
            cat._root_abs_state = cat_copy._root_abs_state
            cat._root = cat_copy._root
        if trajectories_data is None:
            f = open(env_directory_path+"/trajectories.pickle", "rb")
            trajectories_data = pickle.load(f)
        if qtable is None:
            f = open(env_directory_path+"/qtable.pickle", "rb")
            qtable = pickle.load(f)
        inventor = options_inventor.OptionsInventor(cat, trajectories_data)
        options_learned = inventor.construct_options(env_directory_path, env, env_id, qtable, dont_break=dont_break)
        updated_abstractions = {}
        for old_leaf in self.old_leaves:
            if old_leaf not in self.abstract._tree._leaves:
                _, node = self.abstract._tree.find_node_in_tree([], self.abstract._tree._root, old_leaf)
                new_leaves = [temp_node._state for temp_node in self.abstract._tree.get_all_leafs(node)]
                updated_abstractions[old_leaf] = new_leaves
        if qtable_con is not None:
            for option in options_learned:
                option.qtable_con = qtable_con
        
        for option in options_learned:
            # if self.env._train_option is not None and dont_break == True:
            #     option = self.learn_initiation(option, trajectories_data)
            # else:
            #     option, final_steps = self.learn_initiation_policy(option, final_steps=final_steps)
            option = self.learn_initiation(option, option.abstract_trajectories)
            

        utils.plot_option_path(options_learned, self.abstract._tree, env, env_id, env_directory_path, filename=f"catoptions_{self.optionid}")
        if self.env._train_option is None:
            for i in range(len(options_learned)):
                utils.plot_option_path([options_learned[i]], self.abstract._tree, env, env_id, env_directory_path, filename=f"catoptions_{i}")

        filename = "options"
        file = open(env_directory_path+"/"+filename+".pickle","wb")
        pickle.dump(options_learned,file)
        file.close()
        return options_learned, updated_abstractions, trajectories_data, final_steps
    
    def learn_initiation_policy(self, option, final_steps=0, abs_index=0):
        dont_break = True
        epsilon = 0.4
        option_decay = 0.98
        option_stepmax = hyper_param.step_max #self.option_stepmax
        option_k_cap = hyper_param.option_k_cap
        option_update_abs_interval = self.update_abs_interval
        option_evaluate_interval = self.evaluate_interval
        option_training_epi_threshold = self.option_training_epi_threshold
        option_env = copy.deepcopy(self.env)
        option_env._train_option = option
        option_agent = IntraOptionAbstractAgent(option_env, epsilon=epsilon, decay=option_decay)

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
        catrl.env = option_env
        catrl.agent = option_agent
        catrl.agent_con = option_agent_con
        catrl.abstract = option_abstract
        catrl.log = option_log
        catrl.total_timesteps = self.total_timesteps

        catrl.option_training_epi_threshold = option_training_epi_threshold
        catrl.update_abs_interval = option_update_abs_interval
        catrl.evaluate_interval=option_evaluate_interval
        catrl.total_timesteps=self.total_timesteps
        args = copy.deepcopy(hyper_param.problem_args[self.env_id])
        args["init_vars"] = option.init_states[0]
        catrl.initialize_problem(args, option_stepmax, self.env_id)
        catrl.optionid = option.optionid

        trajectories_data = {}
        trajectories_data["trajectories"] = []
        # new_options = []
        updated_abstractions = []
        couldnt_learn = False
        success = False
        i = 0
        print(f"\n----------------------- Env id:{self.env_id} option_learning optionid: {option.optionid} {option.termination_set} -----------------------")
        while not couldnt_learn and i < self.episode_max and final_steps < self.total_timesteps:
            init_state = self.env.reset()
            i, final_steps, abs_index, learn_options, couldnt_learn, trajectories_data, success_state = catrl.catrl(i, final_steps, abs_index, mode = "\toption_learning")
            success_state = None
            if learn_options:
                while success_state is None:
                    _, _, _, success_state = catrl.evaluate_policy(i, abs_index, n_eval=1)
                success = True
                # 3. invent options 
                # final_steps += catrl.evaluation_steps
                # local_stg_graph = catrl.abstract.plot_local_stg(catrl.env_directory_path, abs_index, catrl.optionid)
                option = self.learn_initiation(option, trajectories_data)
                # new_options, updated_abstractions, trajectories_data = catrl.learn_catoptions(catrl.env, catrl.env_id, catrl.env_directory_path, catrl.abstract._tree, trajectories_data, catrl.agent._qtable, dont_break=dont_break, qtable_con=None, final_steps=final_steps, abs_index=abs_index)
                break
            elif final_steps > self.total_timesteps:
                couldnt_learn = True
                print('Oops, our approach could not learn within given timesteps /\(-|-)/\\')
                break

        option.cat = option_abstract._tree
        option._qtable = option_agent._qtable

        # end_states = set()
        # for traj in trajectories_data["trajectories"]:
        #     end_states.add(tuple(traj.trajectory[-1].next_state))

        print("\tSuccess rate:", catrl.log.latest_eval_success(), " in episodes:",i, " in timesteps ", final_steps)
        print(f"----------------------- Env id: {self.env_id} Ended option_learning optionid: {option.optionid} {option.termination_set} -----------------------\n")
        
        return option, final_steps
    
    def learn_initiation(self, option, trajectories_data):
        print("Learning initiation...")
        if len(list(option.initiation_set)) > 0:
            node = self.abstract._tree.find_node(list(option.initiation_set)[0])
            transition_graph = {}
            # for trajectory in trajectories_data["abstract_trajectories"]:
            for trajectory in trajectories_data:
                # for transition in trajectory.trajectory:
                for transition in trajectory:
                    # state = transition.state
                    state = transition[0]
                    # next_state = transition.next_state
                    next_state = transition[2]
                    if state not in transition_graph:
                        transition_graph[state] = {}
                    if next_state not in transition_graph[state]:
                        transition_graph[state][next_state] = {}
                    option.policy_states.add(state)
                    option.policy_states.add(next_state)
            i = 0
            number_of_levels_for_siblings = 2
            while i < number_of_levels_for_siblings and node is not None and node._parent is not None:
                i += 1
                siblings = self.abstract._tree.get_all_leafs(node._parent)
                sibling_abstract_states = [node._state for node in siblings]
                for sibling_abstract_state in sibling_abstract_states:
                    if sibling_abstract_state in transition_graph and sibling_abstract_state in self.abstract._tree._leaves:
                        option.initiation_set.add(sibling_abstract_state)
                node = node._parent
        print("Learned initiation...")
        return option

    def catrl(self, epi_i, final_steps, abs_index, mode="learning"):
        epi_i += 1
        learn_options = False
        refine_abs = True
        self.trajectories_data = {}
        self.trajectories_data["trajectories"] = []

        if epi_i==1:
            if self.abstract.state(self.env._init_state) == self.abstract.state(self.env._goal_state):
                self.abstract.init_goal_abs_state = copy.deepcopy(self.abstract.state(self.env._init_state))
            while self.abstract.state(self.env._init_state) == self.abstract.state(self.env._goal_state):
                prev_goal_abs_state = self.abstract.state(self.env._goal_state)
                self.abstract.update_tree(self.agent.abs_to_con, self.abstract.state(self.env._goal_state), vector=[1 for i in range(len(self.env._init_state))])
                print(f"{self.abstract.state(self.env._init_state)=}, {self.abstract.state(self.env._goal_state)=}")
                if prev_goal_abs_state == self.abstract.state(self.env._goal_state):
                    break
                self.too_small_option = True
                self.adaptively_set_option_params(epi_i)
        success_state, _, _, _, final_steps, _  = self.abstract_qlearning_episode(epi_i, final_steps, abs_index, mode)
        if (epi_i % (self.update_abs_interval/2) == 0):
            self.log_abs_eval = True

        if (epi_i % self.evaluate_interval == 0):
            self.trajectories_data, most_unstable_state, new_avg_eval_success, success_state = self.evaluate_policy(epi_i, abs_index)
            if new_avg_eval_success > self.last_avg_eval_success:
                self.last_avg_eval_success = copy.deepcopy(new_avg_eval_success)
                refine_abs = False
            # self.log.plot_save_performance(self.training_steps, self.env_directory_path, self.file_name, self.optionid)
            if hyper_param.plot_heatmaps:
                init_state = self.env.reset()
                self.abstract.plot_heatmaps(init_state, self.env._goal_state, self.heatmap_directory_path, abs_index, optionid=self.optionid)
            if self.log.latest_eval_success() >= self.eval_success_threshold:
                learn_options = True
                # self.log.plot_save_performance(self.training_steps, self.directory_path, self.file_name, self.optionid)

        if (epi_i % self.update_abs_interval == 0) and (not learn_options) and refine_abs:
            # self.log.plot_save_performance(self.training_steps, self.env_directory_path, self.file_name, self.optionid)
            self.abstract.plot_cat(self.env_directory_path)
            if hyper_param.plot_heatmaps:
                self.abstract.plot_heatmaps(init_state, self.env._goal_state, self.heatmap_directory_path, abs_index, optionid=self.optionid)
            # self.abstract.plot_stgs(self.env_directory_path, abs_index, self.optionid)
            # self.abstract.plot_local_stg(self.env_directory_path, abs_index, self.optionid)
            # with open(self.env_directory_path+"/eval_log.pickle","wb") as file:
            #     pickle.dump(eval_log, file)
            if 0.0 <= self.log.recent_learning_success() < self.eval_success_threshold or (self.log.recent_learning_success() >= self.eval_success_threshold and self.log.latest_eval_success() < self.eval_success_threshold):
                print("\nUpdating abstraction..")
                self.abstract.update_abstraction(self.agent._tderror_table, self.agent.abs_to_con, refine_levels=self.refine_levels)
                self.abstract.update_n_abstract_state()
                abs_index += 1
                self.agent.initialize_tderror()
                self.log_abs_eval = False

        return epi_i, final_steps, abs_index, learn_options, self.couldnt_learn, self.trajectories_data, success_state

    def add_abstract_to_concrete_states(self, abs_state, state):
        if abs_state not in self.agent.abs_to_con:
            self.agent.abs_to_con[abs_state] = set()
        self.agent.abs_to_con[abs_state].add(self.env.state_to_index(state))
        # abs_state.add_concrete_state(state)
    
    def abstract_qlearning_episode(self, epi_i, final_steps, abs_index, mode="learning", plot_succ_trajectory=False):
        state = self.env.reset()
        success_state = copy.deepcopy(state)
        state_abs = self.abstract.state(state)
        self.add_abstract_to_concrete_states(state_abs, state)
        self.agent.update_qtable(state_abs)    # use bootstrapped qvalues from concrete table as initialization
        abstract_trajectories = []
        abs_traj = Trajectory()
        done = False
        success = False
        reward = 0
        steps = 0
        self.agent.initialize_tderror_state(state_abs)
        while (not done) and (steps < self.env._step_max) and (final_steps < self.total_timesteps):
            action = self.agent.policy(state_abs)
            new_state_abs = copy.deepcopy(state_abs)
            r = 0
            while new_state_abs == state_abs:
                new_state, temp_r, done, success = self.env.step(action, self.abstract)
                success = success["success"]
                new_state_abs = self.abstract.state(new_state)
                self.training_steps += 1
                final_steps += 1
                # self.cumulative_reward += temp_r
                self.cumulative_success += int(success)
                self.log.log_step(self.training_steps, temp_r, self.cumulative_success)
                reward += temp_r
                # print(state, action, new_state)
                self.add_abstract_to_concrete_states(new_state_abs, new_state)
                self.agent.update_qtable(new_state_abs)
                if self.abstract._bootstrap == 'from_concrete': 
                    sample = Sample(state, action, new_state, temp_r)
                    # sample = [copy.deepcopy(state), copy.deepcopy(new_state), copy.deepcopy(action), copy.deepcopy(temp_r)]
                    self.agent_con.train(sample)     
                elif self.abstract._bootstrap == 'from_estimated_concrete':
                    sample = Sample(tuple(state), action, new_state_abs, temp_r)
                    self.agent.estimate_concrete_qvalues(sample)
                r += temp_r
                steps += 1
                self.total_succ_trajectories += int(success)
                if success:
                    success_state = copy.deepcopy(new_state)
                if plot_succ_trajectory:
                    abs_transition = Transition(state_abs, action, action, new_state_abs)
                    abs_traj.append_transition(abs_transition)
                    # abs_traj.append([copy.deepcopy(state_abs), copy.deepcopy(action), copy.deepcopy(new_state_abs)])
                    if success:
                        abstract_trajectories.append(abs_traj)
                # if new_state_abs != state_abs:
                #     self.log.log_concrete_transition(tuple(state), tuple(new_state), action)
                if state == new_state or done or steps >= self.env._step_max:
                    break
                state = copy.deepcopy(new_state)
                # state = copy.deepcopy(new_state)
                # if done or steps >= self.step_max:
                #     break
            self.abstract.update_stg(abs_index, state_abs, action, new_state_abs)
            if plot_succ_trajectory and success:
                self.abstract.update_local_stg(abs_index, abstract_trajectories)
            self.add_abstract_to_concrete_states(new_state_abs, new_state)
            self.agent.update_qtable(new_state_abs)
            sample = Sample(state_abs, action, new_state_abs, r)
            # sample = [copy.deepcopy(state_abs), copy.deepcopy(new_state_abs), copy.deepcopy(action), copy.deepcopy(r)]
            self.agent.train(sample)
            self.agent.initialize_tderror_state(sample.next_state)
            if self.log_abs_eval:
                self.agent.log_values(sample)
            last_state_abs = copy.deepcopy(state_abs)
            state_abs = copy.deepcopy(new_state_abs)       
        self.agent.decay()
        self.log.log_episode(epi_i, final_steps, reward, success, steps)
        self.log.log_abs_size(epi_i, self.abstract._n_abstract_states)
        # recent_success = self.log.recent_learning_success()
        # self.agent._epsilon = 1.0 - recent_success
        self.log.print_info(self.env_id, epi_i, self.agent._action_size, final_steps, self.agent._epsilon, reward, success, steps, self.training_steps, self.abstract._n_abstract_states, mode=mode, option_in_plan=False, print_info=hyper_param.print_info, state=new_state)

        self.adaptively_set_option_params(epi_i)
        return success_state, reward, success, steps, final_steps, None

    def adaptively_set_option_params(self, epi_i):
        if self.env._train_option is not None:
            if self.too_small_option:
                if self.agent._epsilon > 0.4:
                    self.agent._epsilon = 0.4
                self.env._step_max = int(self.initial_step_max//10)
            else:
                if self.total_succ_trajectories == 0 and epi_i > self.option_training_epi_threshold:
                    self.couldnt_learn = True
                # adaptively set stepmax
                recent_success = self.log.recent_learning_success()
                if recent_success > 0.1:
                    recent_steps = self.log.recent_learning_steps()
                    # minimum_success_steps = self.log.minimum_learning_success_steps()
                    self.env._step_max = min(self.initial_step_max, 1.5*recent_steps)
                    # self.env._step_max = min(self.initial_step_max, 5*minimum_success_steps)
                else:
                    self.env._step_max = self.initial_step_max
                # adaptively set k_cap
                if recent_success != 0:
                    minimum_success_steps = self.log.minimum_learning_success_steps()
                    if minimum_success_steps < self.initial_step_max/10:
                        if self.agent._epsilon > 0.4:
                            self.agent._epsilon = 0.4
                        self.abstract._k = int(self.abstract._k_initial)
                        self.env._step_max = min(self.initial_step_max, self.adaptive_step_factor *minimum_success_steps)
                        self.update_abs_interval = copy.deepcopy(int(self.update_abs_interval_initial))
                        self.evaluate_interval = copy.deepcopy(int(self.evaluate_interval_initial))
                    else:
                        self.abstract._k = self.abstract._k_initial
                        self.env._step_max = self.initial_step_max
                        self.update_abs_interval = copy.deepcopy(self.update_abs_interval_initial)
                        self.evaluate_interval = copy.deepcopy(self.evaluate_interval_initial) 
                # couldn't learn if exp small and recent succ still 0
                # if self.agent._epsilon < 0.1 and recent_success == 0:
                #     self.couldnt_learn = True       

    def evaluate_policy(self, epi_i, abs_index, n_eval=100):
        trajectories = []
        abstract_trajectories = []
        unstable_states_to_count = {}
        most_unstable_state = None
        total_reward_list = []
        steps_list = []
        success_list = []
        self.evaluation_steps = 0
        success_state = None
        for epi_j in range(n_eval):
            traj = Trajectory()
            abs_traj = Trajectory()
            state = self.env.reset()
            state_abs = self.abstract.state(state)
            self.add_abstract_to_concrete_states(state_abs, state)
            self.agent.update_qtable(state_abs) ##TODO: 
            done = False
            reward = 0
            steps = 0
            # print("\n\n")
            while (not done) and (steps < self.initial_step_max):
                action = self.agent.policy(state_abs, sample=False)
                new_state_abs = copy.deepcopy(state_abs)
                r = 0
                while new_state_abs == state_abs:
                    new_state, temp_r, done, success = self.env.step(action, self.abstract)
                    success = success["success"]
                    # self.env.render()
                    # print(state, action, new_state)
                    new_state_abs = self.abstract.state(new_state)
                    self.add_abstract_to_concrete_states(new_state_abs, new_state)
                    self.agent.update_qtable(new_state_abs)
                    r += temp_r
                    steps += 1 
                    self.evaluation_steps += 1
                    transition = Transition(state, action, action, new_state)
                    traj.append_transition(transition)
                    # traj.append([copy.deepcopy(state), copy.deepcopy(action), copy.deepcopy(action), copy.deepcopy(new_state)])
                    abs_transition = Transition(state_abs, action, action, new_state_abs)
                    abs_traj.append_transition(abs_transition)
                    # abs_traj.append([copy.deepcopy(state_abs), copy.deepcopy(action), copy.deepcopy(action), copy.deepcopy(new_state_abs)])
                    if success:
                        success_state = copy.deepcopy(new_state)
                        trajectories.append(traj)
                        abstract_trajectories.append(abs_traj)
                    if new_state_abs != state_abs:
                        self.log.log_concrete_transition(tuple(state), tuple(new_state), action)
                    if state == new_state or done or steps >= self.initial_step_max:
                        break
                    state = copy.deepcopy(new_state)
                self.add_abstract_to_concrete_states(new_state_abs, new_state)
                self.agent.update_qtable(new_state_abs)
                state = copy.deepcopy(new_state)
                self.abstract.update_stg(abs_index, state_abs, action, new_state_abs)
                state_abs = copy.deepcopy(new_state_abs)
                reward += r  
            total_reward_list.append(reward)
            steps_list.append(steps)
            success_list.append(int(success))
            if not done:
                if state_abs not in unstable_states_to_count:
                    unstable_states_to_count[state_abs] = 0.0
                unstable_states_to_count[state_abs] += 1

        # self.env._env.close()
        # self.env.reinitialize_env_render(None)
        self.log.log_eval(epi_i, total_reward_list, success_list, steps_list)
        self.abstract.update_local_stg(abs_index, abstract_trajectories)
        
        sorted_unstable_states = dict(sorted(unstable_states_to_count.items(),reverse=True))
        if len(sorted_unstable_states.keys()) > 0:
            most_unstable_state_and_frequency = list(sorted_unstable_states.items())[0] 
            if most_unstable_state_and_frequency[1] > 5:
                most_unstable_state = most_unstable_state_and_frequency[0]
        # print("Trajectories:",len(abstract_trajectories))
        # print("Most unstable state:",most_unstable_state)
        most_unstable_state = None
        print("\nEvaluation episodes:",self.log._eval_data["episode"])
        print("Mean success rates: {}".format([np.mean(list_) for list_ in self.log._eval_data["success_list"]]))
        self.trajectories_data = {}
        self.trajectories_data["trajectories"] = trajectories
        self.trajectories_data["abstract_trajectories"] = abstract_trajectories
        return self.trajectories_data, most_unstable_state, np.mean(success_list), success_state

class Qlearning:
    def __init__(self, env, agent, abstract, log):
        self.env = env
        # self.step_max = self.env._step_max
        self.agent = agent
        self.abstract = abstract
        self.log = log

    def qlearning_episode(self, epi_i, init_state):
        state = self.env.reset(init_state)
        done = False
        success = False
        steps = 0
        reward = 0
        while not done and steps < self.env._step_max:
            self.agent.update_qtable(state)
            action = self.agent.policy(state)
            new_state, temp_r, done, success = self.env.step(action, self.abstract)
            success = success["success"]
            reward += temp_r
            # print(state, action, new_state)
            self.agent.update_qtable(new_state)
            steps += 1
            sample = [copy.deepcopy(state), copy.deepcopy(new_state), copy.deepcopy(action), copy.deepcopy(temp_r)]     
            self.agent.train(*sample)
            state = copy.deepcopy(new_state)
            if done or steps >= self.env._step_max:
                break 
        self.agent.decay()
        self.log.log_episode(epi_i, reward, success, steps)
        self.log.print_info(self.env_id, epi_i, self.agent._epsilon, reward, success, steps, self.training_steps, self.abstract._n_abstract_states, "qlearning", print_info=hyper_param.print_info, state=new_state)
        return reward, success, steps

