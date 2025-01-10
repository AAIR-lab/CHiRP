import numpy as np
import copy
import os

from src.agent.learning import *
from option_invention import options_inventor
from src.data_structures.trajectory import Sample, Transition, Trajectory


class CATRL:
    def __init__(self, env, agent, agent_con, abstract, log, episode_max, directory_path, update_abs_interval=100, update_abs_episodes=100, evaluate_interval=50, refine_levels=1, file_name='catrl', total_timesteps= hyper_param.total_timesteps):
        self.env = env
        self.agent = agent
        self.agent_con = agent_con
        self.abstract = abstract
        self.log = log
        self.episode_max = episode_max
        self.directory_path = directory_path
        self.file_name = file_name
        self.update_abs_interval = update_abs_interval
        self.update_abs_episodes = update_abs_episodes
        self.evaluate_interval = evaluate_interval
        self.refine_levels = refine_levels
        self.optionid = None
        self.training_steps = 0
        self.cumulative_success = 0
        self.total_timesteps = total_timesteps

    def initialize_problem(self, args, stepmax, env_id):
        self.env.initialize_problem(args, stepmax)
        self.step_max = self.env._step_max
        self.env_id = env_id
        self.step_max = self.env._step_max
        # self.env_directory_path = self.directory_path+"/env_"+str(self.env_id)
        # if not os.path.exists(self.env_directory_path+"/"):
        #     os.makedirs(self.env_directory_path+"/")
        self.heatmap_directory_path = self.directory_path+"/heatmaps"
        if not os.path.exists(self.heatmap_directory_path+"/"):
            os.makedirs(self.heatmap_directory_path+"/")

    # def learn_catoptions(self, args, stepmax, epi_i, abs_index):
    #     self.initialize_problem(args, stepmax, env_id)
    #     while(epi_i  <= self.episode_max):
    #         init_state = self.env.reset()
    #         epi_i, abs_index, learn_options = self.catrl(epi_i, abs_index, init_state)
    #         if learn_options:
    #             self.log.plot_save_performance(self.training_steps, self.directory_path, self.file_name, self.optionid)
    #             # options_learned, _ = self.learn_options(epi_i, abs_index, init_state)
    #             break
    #     return epi_i, abs_index

    def catrl(self, epi_i, final_steps, abs_index, init_state):
        epi_i += 1
        learn_options = False

        _, _, _, _, _, final_steps  = self.abstract_qlearning_episode(epi_i, final_steps, abs_index, init_state, [], [], "abs_learning")

        if (epi_i % self.evaluate_interval == 0):
            _, _, _ = self.evaluate_policy(epi_i, abs_index, init_state)
            # self.log.plot_save_performance(self.training_steps, self.directory_path, self.file_name, self.optionid)
            if hyper_param.plot_heatmaps:
                self.abstract.plot_heatmaps(init_state, self.env._goal_state, self.heatmap_directory_path, abs_index, optionid=self.optionid)
            # if self.log.recent_learning_success() >= 0.9 and self.log.latest_eval_success() >= 0.9:
            if self.log.latest_eval_success() >= 0.9:
                learn_options = True
                self.abstract.plot_cat(self.directory_path)
                self.log.plot_save_performance(self.training_steps, self.directory_path, self.file_name, self.optionid)


        if (epi_i % self.update_abs_interval == 0) and (not learn_options):
            if self.log.recent_learning_success() < 0.9 or (self.log.recent_learning_success() >= 0.9 and self.log.latest_eval_success() < 0.9):
                print("\nEvaluating abstraction..")
                self.agent.initialize_tderror()
                batch_con, batch_abs = [], []
                for j in range(self.update_abs_episodes):
                    # _, _, _ = self.evaluate_stochastic_policy(epi_i, abs_index, init_state, "abs_evaluation")
                    epi_i += 1
                    _, _, _, batch_con, batch_abs, final_steps = self.abstract_qlearning_episode(epi_i, final_steps, abs_index, init_state, batch_con, batch_abs, "abs_evaluation")
                self.agent.batch_train(batch_abs)
                if self.abstract._bootstrap == 'from_concrete': 
                    self.agent_con.batch_train(batch_con)

                if (epi_i % self.evaluate_interval == 0):
                    _, _, most_unstable_state = self.evaluate_policy(epi_i, abs_index, init_state)
                self.log.plot_save_performance(self.training_steps, self.directory_path, self.file_name, self.optionid)
                if hyper_param.plot_heatmaps:
                    self.abstract.plot_heatmaps(init_state, self.env._goal_state, self.heatmap_directory_path, abs_index, optionid=self.optionid)
                # self.abstract.plot_stgs(self.directory_path, abs_index, self.optionid)
                # self.abstract.plot_local_stg(self.directory_path, abs_index, self.optionid)
                if self.log.recent_learning_success() < 0.9 or (self.log.recent_learning_success() >= 0.9 and self.log.latest_eval_success() < 0.9):
                    print("\nUpdating abstraction..")
                    self.abstract.update_abstraction(self.agent._tderror_table, self.agent.abs_to_con, refine_levels=self.refine_levels)
                    # if most_unstable_state in self.abstract._tree._leaves:
                    #     self.abstract.update_abstraction({}, unstable_state=most_unstable_state, refine_levels=self.refine_levels)
                    self.abstract.update_n_abstract_state()
                    abs_index += 1

            self.abstract.plot_cat(self.directory_path)

        return epi_i,  abs_index, learn_options, final_steps
    
    def abstract_qlearning_episode(self, epi_i, final_steps, abs_index, init_state, batch_con, batch_abs, mode):
        # state = self.env.reset(init_state)
        state = self.env.reset()
        state_abs = self.abstract.state(state)
        self.agent.update_qtable(state_abs)    # use bootstrapped qvalues from concrete table as initialization
        done = False
        success = False
        reward = 0
        steps = 0
        if mode=="abs_evaluation":
            self.agent.initialize_tderror_state(state_abs)
        if state_abs not in self.agent.abs_to_con:
            self.agent.abs_to_con[state_abs] = []
        self.agent.abs_to_con[state_abs].append(state)
        while (not done) and (steps < self.step_max) and (final_steps < self.total_timesteps):
            action = self.agent.policy(state_abs)
            new_state_abs = copy.deepcopy(state_abs)
            r = 0
            while new_state_abs == state_abs:
                new_state, temp_r, done, info = self.env.step(action, self.abstract)
                success = info["success"]
                self.training_steps += 1
                final_steps += 1
                self.cumulative_success += int(success)
                self.log.log_step(self.training_steps, temp_r, self.cumulative_success)
                reward += temp_r
                # print(state, action, new_state)
                new_state_abs = self.abstract.state(new_state)
                # if new_state_abs not in self.agent.abs_to_con:
                #     self.agent.abs_to_con[new_state_abs] = []
                # self.agent.abs_to_con[new_state_abs].append(new_state)

                self.agent.update_qtable(new_state_abs)
                if self.abstract._bootstrap == 'from_concrete': 
                    sample = Sample(state, action, new_state, temp_r)
                    if mode=="abs_learning":
                        self.agent_con.train(sample)
                    if mode=="abs_evaluation":
                        batch_con.append(sample)              
                r += temp_r
                steps += 1
                if state == new_state or done or steps >= self.step_max:
                    break
                state = copy.deepcopy(new_state)
                # state = copy.deepcopy(new_state)
                # if done or steps >= self.step_max:
                #     break
            self.abstract.update_stg(abs_index, state_abs, action, new_state_abs)
            self.agent.update_qtable(new_state_abs)
            sample = Sample(state_abs, action, new_state_abs, r)
            if mode=="abs_learning":
                self.agent.train(sample)
            if mode=="abs_evaluation":
                batch_abs.append(sample)
                self.agent.initialize_tderror_state(sample.next_state)
                self.agent.log_values(sample)
            state_abs = copy.deepcopy(new_state_abs)
        if mode=="abs_learning":       
            self.agent.decay()
        self.log.log_episode(epi_i, final_steps, reward, success, steps)
        self.log.log_abs_size(epi_i, self.abstract._n_abstract_states)
        self.log.print_info(self.env_id, epi_i, self.agent._action_size, final_steps, self.agent._epsilon, reward, success, steps, self.training_steps, self.abstract._n_abstract_states, mode, False, print_info=hyper_param.print_info, state=new_state)
        return reward, success, steps, batch_con, batch_abs, final_steps 

    def learn_options(self, epi_i, abs_index, init_state):
        print("\nLearning options..")
        # self.log.plot_save_performance(self.training_steps, self.directory_path, self.file_name, self.optionid)
        trajectories, abstract_trajectories, _ = self.evaluate_policy(epi_i, abs_index, init_state, n_eval=200)
        self.abstract.plot_cat(self.directory_path)
        self.abstract.plot_stgs(self.directory_path, abs_index, self.optionid)
        graph = self.abstract.plot_local_stg(self.directory_path, abs_index, self.optionid)
        initial_state_abs = self.abstract.state(init_state)
        if self.env._train_option is None or self.env._train_option._is_goal_option:
            # goal_state_abs = self.abstract.state(self.env._goal_state)
            goal_states_abs = self.env.get_goal_abstract_states(self.abstract._tree._leaves)
        else:
            goal_states_abs = self.env._train_option.get_goal_abs_state()
        self.log.save_qtable(self.agent._qtable, self.directory_path, self.optionid)
        self.log.save_trajectories(trajectories, abstract_trajectories, self.directory_path, self.optionid)
        options_learned = options_inventor.construct_options(graph, initial_state_abs, goal_states_abs, self.directory_path, self.agent._qtable, self.optionid)
        if hyper_param.plot_heatmaps:
            self.abstract.plot_heatmaps(init_state, self.env.goal_state, self.heatmap_directory_path, abs_index, optionid=self.optionid)
        return options_learned, initial_state_abs, goal_states_abs
    
    def evaluate_stochastic_policy(self, epi_i, abs_index, init_state, mode):
        # state = self.env.reset(init_state)
        state = self.env.reset()
        state_abs = self.abstract.state(state)
        self.agent.update_qtable(state_abs)    # use bootstrapped qvalues from concrete table as initialization
        done = False
        success = False
        reward = 0
        steps = 0
        if mode=="abs_evaluation":
            self.agent.initialize_tderror_state(state_abs)
        while (not done) and (steps < self.step_max):
            action = self.agent.policy(state_abs)
            new_state_abs = copy.deepcopy(state_abs)
            r = 0
            while new_state_abs == state_abs:
                new_state, temp_r, done, info = self.env.step(action, self.abstract)
                success = info["success"]
                # self.total_steps += 1
                # self.log.log_step(self.total_steps, temp_r, success)
                reward += temp_r
                # print(state, action, new_state)
                new_state_abs = self.abstract.state(new_state)
                self.agent.update_qtable(new_state_abs)            
                r += temp_r
                steps += 1
                if state == new_state or done or steps >= self.step_max:
                    break
                state = copy.deepcopy(new_state)
            self.agent.update_qtable(new_state_abs)
            sample = [copy.deepcopy(state_abs), copy.deepcopy(new_state_abs), copy.deepcopy(action), copy.deepcopy(r)]
            if mode=="abs_evaluation":
                self.agent.initialize_tderror_state(sample[1])
                self.agent.log_values(sample[0], sample[1], sample[2], sample[3])
            state_abs = copy.deepcopy(new_state_abs)
        self.log.print_info(self.env_id, epi_i, self.agent._epsilon, reward, success, steps, self.training_steps, self.abstract._n_abstract_states, mode, print_info=hyper_param.print_info, state=new_state)
        return reward, success, steps
    
    def evaluate_policy(self, epi_i, abs_index, init_state, n_eval=100):
        trajectories = []
        abstract_trajectories = []
        unstable_states_to_count = {}
        most_unstable_state = None
        total_reward_list = []
        steps_list = []
        success_list = []
        for epi_j in range(n_eval):
            traj = []
            abs_traj = []
            # state = self.env.reset(init_state)
            state = self.env.reset()
            state_abs = self.abstract.state(state)
            self.agent.update_qtable(state_abs) ##TODO: 
            done = False
            reward = 0
            steps = 0
            while (not done) and (steps < self.env._step_max):
                action = self.agent.policy(state_abs, sample=False)
                new_state_abs = copy.deepcopy(state_abs)
                r = 0
                while new_state_abs == state_abs:
                    new_state, temp_r, done, info = self.env.step(action, self.abstract)
                    success = info["success"]
                    # print(state, action, new_state)
                    if new_state_abs not in self.agent.abs_to_con:
                        self.agent.abs_to_con[new_state_abs] = []
                    self.agent.abs_to_con[new_state_abs].append(new_state)
                    
                    new_state_abs = self.abstract.state(new_state)
                    self.agent.update_qtable(new_state_abs)
                    r += temp_r
                    steps += 1 
                    traj.append([copy.deepcopy(state), copy.deepcopy(action), copy.deepcopy(new_state)])
                    abs_traj.append([copy.deepcopy(state_abs), copy.deepcopy(action), copy.deepcopy(new_state_abs)])
                    if success:
                        trajectories.append(copy.deepcopy(traj))
                        abstract_trajectories.append(copy.deepcopy(abs_traj))
                    if state == new_state or done or steps >= self.env._step_max:
                        break
                    state = copy.deepcopy(new_state)
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

        self.log.log_eval(epi_i, total_reward_list, success_list, steps_list)
        # self.abstract.update_local_stg(abs_index, abstract_trajectories)
        
        # sorted_unstable_states = dict(sorted(unstable_states_to_count.items(),reverse=True))
        # if len(sorted_unstable_states.keys()) > 0:
        #     most_unstable_state_and_frequency = list(sorted_unstable_states.items())[0] 
        #     if most_unstable_state_and_frequency[1] > 5:
        #         most_unstable_state = most_unstable_state_and_frequency[0]
        print("Trajectories:",len(abstract_trajectories))
        print("Most unstable state:",most_unstable_state)
        most_unstable_state = None

        print("\nEvaluation episodes:",self.log._eval_data["episode"])
        print("Mean success rates: {}".format([np.mean(list_) for list_ in self.log._eval_data["success_list"]]))
        return trajectories, abstract_trajectories, most_unstable_state

class Qlearning:
    def __init__(self, env, agent, abstract, log):
        self.env = env
        self.step_max = self.env._step_max
        self.agent = agent
        self.abstract = abstract
        self.log = log

    def qlearning_episode(self, epi_i, init_state):
        # state = self.env.reset(init_state)
        state = self.env.reset()
        done = False
        success = False
        steps = 0
        reward = 0
        while not done and steps < self.step_max:
            self.agent.update_qtable(state)
            action = self.agent.policy(state)
            new_state, temp_r, done, success = self.env.step(action, self.abstract)
            reward += temp_r
            # print(state, action, new_state)
            self.agent.update_qtable(new_state)
            steps += 1
            sample = [copy.deepcopy(state), copy.deepcopy(new_state), copy.deepcopy(action), copy.deepcopy(temp_r)]     
            self.agent.train(*sample)
            state = copy.deepcopy(new_state)
            if done or steps >= self.step_max:
                break 
        self.agent.decay()
        self.log.log_episode(epi_i, reward, success, steps)
        self.log.print_info(self.env_id, epi_i, self.agent._epsilon, reward, success, steps, self.training_steps, self.abstract._n_abstract_states, "qlearning", state=new_state)
        return reward, success, steps

