import numpy as np
import matplotlib.pylab as plt
import pickle
import os
class Log_experiments:
    def __init__(self, moving_number = 100):
        self._episode_data = {'reward': [], 'success': [], 'steps': [], 'episode': [], 'timesteps': []}
        self._step_data = {'reward': [], 'success': [], 'training_steps': []}
        self._eval_data = {'reward_list': [], 'success_list': [], 'steps_list': [], 'episode': []}
        self._abs_size_data = {'abs_size': [], 'episode': []}
        self._transfer_result = {}
        self._transfer_eval = {}
        self._epi_moving_number = moving_number 
        self._steps_moving_number = 1000
        self.concrete_transition_dict = {}

    def log_episode(self, episode, timesteps, reward, success, steps):
        self._episode_data['reward'].append(reward)
        self._episode_data['success'].append(int(success))
        self._episode_data['steps'].append(steps)
        self._episode_data['episode'].append(episode)
        self._episode_data['timesteps'].append(timesteps)

    def log_step(self, training_steps, reward, success):
        self._step_data['reward'].append(reward)
        self._step_data['success'].append(int(success))
        self._step_data['training_steps'].append(training_steps)

    def log_eval(self, episode, reward_list, success_list, steps_list):
        self._eval_data["episode"].append(episode)
        self._eval_data["reward_list"].append(reward_list)
        self._eval_data["success_list"].append(success_list)
        self._eval_data["steps_list"].append(steps_list)

    def log_abs_size(self, episode, abs_size):
        self._abs_size_data["episode"].append(episode)
        self._abs_size_data["abs_size"].append(abs_size)

    def log_transfer_info(self, env_id, training_steps, abs_size, total_options, useful_options):
        self._transfer_result["env_id"] = env_id
        self._transfer_result["training_steps"] = training_steps
        self._transfer_result["abs_size"] = abs_size
        self._transfer_result["total_options"] = total_options
        self._transfer_result["useful_options"] = useful_options
    
    def log_transfer_eval(self, epi_i, reward_list, success_list, steps_list, success_rate):
        self._transfer_eval[epi_i] = {}
        self._transfer_eval[epi_i]["reward_list"] = reward_list
        self._transfer_eval[epi_i]["success_list"] = success_list
        self._transfer_eval[epi_i]["steps_list"] = steps_list
        self._transfer_eval[epi_i]["success_rate"] = success_rate
        self._transfer_result["eval_data"] = self._transfer_eval

    def save_transfer_result(self, directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        path = directory_path+"/transfer_results.pickle"
        with open(path, "wb") as output_file:
            pickle.dump(self._transfer_result, output_file)
        output_file.close()

    def print_info(self, envid, epi_i, action_size, final_steps, epsilon, reward, success, steps, training_steps, num_abs, mode, option_in_plan, print_info=True, state=None):
        if print_info:
            recent_success = self.recent_learning_success()
            recent_steps = self.recent_learning_steps()
            recent_reward = self.recent_learning_reward()
            print("{} ## envid: {} total_steps: {} train_steps: {} epi: {}  #actions: {} epsilon: {:.3f}  #abs: {}  success: {} op_in_plan: {} reward: {} steps: {} rec_success: {}  rec_steps: {} state: {}".format(mode, envid, final_steps, training_steps, epi_i, action_size, epsilon, num_abs, int(success), int(option_in_plan), reward, steps, recent_success, recent_steps, state))

    def latest_eval_success(self):
        eval_succ = 0.0
        if len(self._eval_data["success_list"]) > 0:
            eval_succ = [np.mean(list_) for list_ in self._eval_data["success_list"]][-1]
        return eval_succ

    def recent_learning_success(self):
        return self.recent_rate(self._episode_data['success'], self._epi_moving_number)
    
    def recent_learning_reward(self):
        return self.recent_rate(self._episode_data['reward'], self._epi_moving_number)
    
    def recent_learning_steps(self):
        return self.recent_rate(self._episode_data['steps'], self._epi_moving_number)
    
    def minimum_learning_success_steps(self):
        success_indices = [i for i in range(len(self._episode_data['success'])) if self._episode_data['success'][i] == 1]
        steps = [self._episode_data['steps'][i] for i in success_indices]
        return np.min(steps)

    def recent_rate(self, data, last):
        size = len(data)
        if last < size: 
            x = size - last
        else: 
            x = 0
        data = data[x: size]
        total = 0.0
        for item in data:
            total += item
        recent_rate = round(total/len(data), 3)
        return recent_rate 

    def get_moving_average_x_y(self, x, y, moving_number):
        x_m = []
        y_m = []
        for i in range (moving_number, len(x)):
            sum_temp = 0
            for j in range (i - moving_number, i):
                sum_temp += y[j]
            sum_temp /= moving_number
            y_m.append(sum_temp)
            x_m.append(i)
        return x_m, y_m
    
    def plot_learning_success_vs_training_steps(self, moving_number, directory, filename):
        y = self._step_data["success"]
        x = self._step_data['training_steps']
        # x_m, y_m = self.get_moving_average_x_y(x, y, moving_number)
        plt.figure(1)
        plt.plot (x,y)
        plt.ylabel("Cumulative Success")
        plt.xlabel("Training steps")
        plt.savefig(directory+"/"+filename+"_success_vs_steps.png")

    def plot_learning_reward_vs_training_steps(self, moving_number, directory, filename):
        y = self._step_data["reward"]
        x = self._step_data['training_steps']
        x_m, y_m = self.get_moving_average_x_y(x, y, moving_number)
        plt.figure(2)
        plt.plot (x_m,y_m)
        plt.ylabel("Cumulative Reward")
        plt.xlabel("Training steps")
        plt.savefig(directory+"/"+filename+"_reward_vs_steps.png")

    def plot_learning_success_vs_episodes(self, moving_number, directory, filename):
        y = self._episode_data["success"]
        x = self._episode_data['steps']
        x_m, y_m = self.get_moving_average_x_y(x, y, moving_number)
        plt.figure(3)
        plt.plot (x_m,y_m)
        plt.ylabel("Success rate")
        plt.xlabel("Episodes")
        plt.savefig(directory+"/"+filename+"_success.png")

    def plot_learning_reward_vs_episodes(self, moving_number, directory, filename):
        y = self._episode_data["reward"]
        x = self._episode_data['episode']
        x_m, y_m = self.get_moving_average_x_y(x, y, moving_number)
        plt.figure(4)
        plt.plot (x_m,y_m)
        plt.ylabel("Reward")
        plt.xlabel("Episodes")
        plt.savefig(directory+"/"+filename+"_reward.png")

    def plot_learning_steps_vs_episodes(self, moving_number, directory, filename):
        y = self._episode_data["steps"]
        x = self._episode_data['episode']
        x_m, y_m = self.get_moving_average_x_y(x, y, moving_number)
        plt.figure(5)
        plt.plot (x_m,y_m)
        plt.ylabel("Steps")
        plt.xlabel("Episodes")
        plt.savefig(directory+"/"+filename+"_steps.png")

    def plot_evaluation_success_vs_episodes(self, directory, filename):
        y = self._eval_data["success_list"]
        y = [np.mean(np.array(list_, dtype=bool)) for list_ in y]
        x = self._eval_data['episode']
        plt.figure(6)
        plt.plot (x,y)
        # plt.show()
        plt.ylabel("Success rate")
        plt.xlabel("Episodes")
        plt.savefig(directory+"/"+filename+"_eval_success.png")

    def plot_evaluation_reward_vs_episodes(self, directory, filename):
        y = self._eval_data["reward_list"]
        y = [np.mean(list_) for list_ in y]
        x = self._eval_data['episode']
        plt.figure(7)
        plt.plot (x,y)
        # plt.show()
        plt.ylabel("Reward")
        plt.xlabel("Episodes")
        plt.savefig(directory+"/"+filename+"_eval_reward.png")

    def plot_evaluation_steps_vs_episodes(self, directory, filename):
        y = self._eval_data["steps_list"]
        y = [np.mean(list_) for list_ in y]
        x = self._eval_data['episode']
        plt.figure(8)
        plt.plot (x,y)
        # plt.show()
        plt.ylabel("Steps")
        plt.xlabel("Episodes")
        plt.savefig(directory+"/"+filename+"_eval_steps.png")

    def plot_abs_size_vs_episodes(self, directory, filename):
        y = self._abs_size_data["abs_size"]
        x = self._abs_size_data['episode']
        plt.figure(9)
        plt.plot (x,y)
        # plt.show()
        plt.ylabel("Abstraction size")
        plt.xlabel("Episodes")
        plt.savefig(directory+"/"+filename+"_abs_size.png")
       
    def save_performance(self, training_steps, directory, filename):
        training_data = {}
        training_data["abs_size"] = self._abs_size_data
        training_data["episode_data"] = self._episode_data
        training_data["steps_data"] = self._step_data
        training_data["training_steps"] = training_steps
        path = directory+"/"+filename+"_learning.pickle"
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(path, "wb") as output_file:
            pickle.dump(training_data, output_file)
        output_file.close()

        path = directory+"/"+filename+"_evaluation.pickle"
        with open(path, "wb") as output_file:
            pickle.dump(self._eval_data, output_file)
        output_file.close()

    def plot_save_performance(self, training_steps, directory, filename, optionid=None):
        if optionid is None:
            filename = "option"+str(optionid)+"_"+filename
            self.plot_learning_success_vs_training_steps(self._steps_moving_number, directory, filename)
            self.plot_learning_reward_vs_training_steps(self._steps_moving_number, directory, filename)
            self.plot_learning_success_vs_episodes(self._epi_moving_number, directory, filename)
            self.plot_learning_reward_vs_episodes(self._epi_moving_number, directory, filename)
            self.plot_learning_steps_vs_episodes(self._epi_moving_number, directory, filename)
            self.plot_evaluation_success_vs_episodes(directory, filename)
            self.plot_evaluation_reward_vs_episodes(directory, filename)
            self.plot_evaluation_steps_vs_episodes(directory, filename)
            self.plot_abs_size_vs_episodes(directory, filename)
            self.save_performance(training_steps, directory, filename)

    def save_qtable(self, qtable, directory, optionid=None):
        filename = "qtable"
        if not os.path.exists(directory):
            os.makedirs(directory)
        if optionid is None:
            # filename = "option"+str(optionid)+"_"+filename
            with open(directory+"/"+filename+".pickle", 'wb') as f:     
                pickle.dump(qtable, f)

    def save_trajectories(self, trajectories_data, directory, optionid=None):
        filename = "trajectories"
        if not os.path.exists(directory):
            os.makedirs(directory)
        if optionid is None:
            # filename = "option"+str(optionid)+"_"+filename
            with open(directory+"/"+filename+".pickle", 'wb') as f:     
                pickle.dump(trajectories_data, f)
        return trajectories_data
    
    def log_concrete_transition(self, con_state, con_next_state, action):
        if con_state not in self.concrete_transition_dict:
            self.concrete_transition_dict[con_state] = {}
        if con_next_state not in self.concrete_transition_dict[con_state]:
            self.concrete_transition_dict[con_state][con_next_state] = {'label':[]}
        self.concrete_transition_dict[con_state][con_next_state]['label'].append(action)