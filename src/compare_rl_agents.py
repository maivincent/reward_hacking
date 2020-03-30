import os
import csv
import utils as ut
import argparse
import numpy as np


class Comparator(object):
	# Compares the performances of agents trained with the ground truth reward and an observed reward, by computing the regret on the ground truth score
	def __init__(self, root_path):
		# root_path: root_path for the RL results ('...results/rl')
		self.root_path = root_path
		self.test_results = []
		self.train_results = []
		self.drawer = ut.Drawer(os.path.join(self.root_path, 'graphs'))

		self.nb_best = 5 # Number of "best runs" for comparisons

	def avg_std_for_env(self, path, mode):
		path = os.path.join(path, mode)
		list_runs = [os.path.join(path, file) for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
		rewards = np.array([])
		success_rewards = np.array([])
		episode_ids = np.array([])
		number_runs = len(list_runs)

		nb_failed = 0
		nb_success = 0
		nb_unstable = 0

        if len(list_runs) == 1:


		for run in list_runs:
			data = np.load(run)
			# If it is the first time
			if episode_ids.size == 0:
				episode_ids = data[:, 1]
			# Double check that the episode ids are identical through the whole set of results
			elif not np.array_equal(episode_ids, data[:, 1]):
				raise ValueError("Different episode id among the different files. Before, we had until {}, now we go until {} with file {}".format(episode_ids[-1], data[-1, 1], run))

			if rewards.size == 0:
				rewards = np.array([data[:,0]])
			else:
				rewards = np.vstack((rewards, data[:, 0]))
			#print(run)

		# Testing and categorizing
		max_reward = np.max(rewards)
		rew_thresh = 0.7*max_reward			# Threshold of success is 70% of max reward on the environment
		high_avg_thresh = 0.85*max_reward		# High average (needed to be a success)
		low_avg_thresh = 0.5*max_reward		# Low average (needed to be "unstable")
		print(rewards.shape)
		nb_episodes = rewards.shape[1]
		start_period = int(nb_episodes*0.90) 	# Start of period of interest

		# Categorizing
		for run in rewards:
			if np.max(run) < rew_thresh or np.mean(run[start_period:-1]) < low_avg_thresh:   # Fail if never reached 70% of the max score
				cat = 'fail'
				nb_failed += 1
			elif np.min(run[start_period:-1]) < rew_thresh and np.mean(run[start_period:-1]) < high_avg_thresh:
				cat = 'unstable'			# Unstable if, in the last 15% of episodes, at least one is lower than 70% of max score and average is lower than 90% of max score (to prevent labeled as unstable in case there is only one drop)
				nb_unstable += 1
			else:
				cat = 'success'				# Stable otherwise (no lower than 70% of max score or average higher than 85% of max score on last 85%)
				nb_success += 1
				if success_rewards.size == 0:
					success_rewards = run.reshape((1,-1))
				else:
					success_rewards = np.vstack((success_rewards, run))
		status = [nb_success, nb_unstable, nb_failed]

		if len(success_rewards) != 0:
			# Successful ones average and std deviation
			success_average = np.mean(success_rewards, 0)
			success_std = np.std(success_rewards, 0)/np.sqrt(nb_success)

			# Average and std deviation for best runs (best average in the last 50%)
			period_of_interest = int(len(success_rewards[0])/2)
			each_average = np.mean(success_rewards[:, period_of_interest:], 1)
			best_indexes = each_average.argsort()[-self.nb_best:][::1]
			best_success = success_rewards[best_indexes]
			best_success_average = np.mean(best_success, 0)
			best_success_std = np.std(best_success, 0)/np.sqrt(min(self.nb_best, nb_success))
		else:
			success_average = None
			success_std = None
			best_success_average = None
			best_success_std = None

		# All average and std deviation
		all_average = np.mean(rewards, 0)
		all_std = np.std(rewards, 0)/np.sqrt(number_runs)

		return [number_runs, success_average, success_std, episode_ids, status, best_success_average, best_success_std, all_average, all_std]

	def plot(self, envs_list):
		# Test results	

		# All test results
		x_list = []
		y_list = []
		std_list = []
		legend = []
		x_label = 'Training episodes'
		y_label = 'Average score on 10 test episodes'
		for test_result in self.test_results:
			x_list.append(test_result[1][3])
			y_list.append(test_result[1][7])
			std_list.append(test_result[1][8])
			result_leg = '{} - averaged on {} runs'.format(test_result[0], test_result[1][0])
			legend.append(result_leg)
		graph_title = 'SAC test performance with different sources of reward for all runs'
		graph_save_name = 'SAC_test_all_' + envs_list
		self.drawer.saveMultiXPlotWithStdPNG(x_list, y_list, std_list, x_label, y_label, graph_title, graph_save_name, legend)


		# Success test results
		x_list = []
		y_list = []
		std_list = []
		legend = []
		x_label = 'Training episodes'
		y_label = 'Average score on 10 test episodes'
		for test_result in self.test_results:
			x_list.append(test_result[1][3])
			y_list.append(test_result[1][1])
			std_list.append(test_result[1][2])
			result_leg = '{} - averaged on {} runs'.format(test_result[0], test_result[1][4][0])
			legend.append(result_leg)
		graph_title = 'SAC test performance with different sources of reward for successful runs'
		graph_save_name = 'SAC_test_success_' + envs_list
		self.drawer.saveMultiXPlotWithStdPNG(x_list, y_list, std_list, x_label, y_label, graph_title, graph_save_name, legend)

		# Best test results
		x_list = []
		y_list = []
		std_list = []
		legend = []
		for test_result in self.test_results:
			x_list.append(test_result[1][3])
			y_list.append(test_result[1][5])
			std_list.append(test_result[1][6])
			result_leg = '{} - averaged on best {} runs'.format(test_result[0], min(self.nb_best, test_result[1][4][0]))
			legend.append(result_leg)
		graph_title = 'SAC test performance with different sources of reward for top runs'
		graph_save_name = 'SAC_test_best_' + envs_list
		self.drawer.saveMultiXPlotWithStdPNG(x_list, y_list, std_list, x_label, y_label, graph_title, graph_save_name, legend)


		## Histograms of success - fail - unstable
		for test_result in self.test_results:
			status_nb = test_result[1][4]
			env_name = test_result[0]
			graph_title = 'Success statistics for {}'.format(env_name)
			barplot_save_path = os.path.join(self.root_path, env_name, 'StabilityPlot.png')
			self.drawer.saveBarPlot(status_nb, ['Success', 'Unstable', 'Fails'], graph_title, barplot_save_path)

		# All train results	
		x_list = []
		y_list = []
		std_list = []
		legend = []
		x_label = 'Training episodes'
		y_label = 'Average score on training episodes'
		for train_result in self.train_results:
			x_list.append(train_result[1][3])
			y_list.append(train_result[1][7])
			std_list.append(train_result[1][8])
			result_leg = '{} - averaged on {} runs'.format(train_result[0], train_result[1][0])
			legend.append(result_leg)
		graph_title = 'SAC train performance with different sources of reward for all runs'
		graph_save_name = 'SAC_train_all_' + envs_list

		# Succesful train results	
		x_list = []
		y_list = []
		std_list = []
		legend = []
		x_label = 'Training episodes'
		y_label = 'Average score on training episodes'
		for train_result in self.train_results:
			x_list.append(train_result[1][3])
			y_list.append(train_result[1][1])
			std_list.append(train_result[1][2])
			result_leg = '{} - averaged on {} runs'.format(train_result[0], train_result[1][4][0])
			legend.append(result_leg)
		graph_title = 'SAC train performance with different sources of reward for successful runs'
		graph_save_name = 'SAC_train_success_' + envs_list

		self.drawer.saveMultiXPlotWithStdPNG(x_list, y_list, std_list, x_label, y_label, graph_title, graph_save_name, legend)

	def compare(self, folder_list):
		envs = [folder for folder in os.listdir(self.root_path) if (folder in folder_list and os.path.isdir(os.path.join(self.root_path, folder)))]
		for env in envs:
			path_to_runs = os.path.join(self.root_path, env, 'training_data', 'runs')
			self.test_results.append([env, self.avg_std_for_env(path_to_runs, 'test')])
			self.train_results.append([env, self.avg_std_for_env(path_to_runs, 'train')])

		

#################################################
# 					Main 						#
#################################################		

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='This script generates images from a given environment, with the associated label.')
	parser.add_argument('-c','--computer', help='Computer on which the script is launched.',required=True)
	parser.add_argument('-f','--folders', help='List of authorized folders for the graphs.',required=True)
	args = parser.parse_args()
	
	computer = ut.getComputer(args.computer)
	folder_list = ut.getFolderList(args.folders)

	if computer == 'mila':
		root_path = '../cluster_results/rl'
	elif computer == 'local':
		root_path = '../local_results/rl'

	comparator = Comparator(root_path)
	comparator.compare(folder_list)
	comparator.plot(''.join(folder_list))
