import argparse
import datetime
import numpy as np
import utils as ut
import sys
import time
import os
import csv
import torch




class RLKitDataParser():
	def __init__(self):
		pass


	def import_data(self, path_to_load):

		## Progress of the learning process
		path_to_CSV = os.path.join(path_to_load, 'progress.csv')

		mean_ev_returns = []
		std_ev_returns = []
		max_ev_returns = []
		min_ev_returns = []
		mean_ev_nb_steps_per_episode = []
		std_ev_nb_steps_per_episode = []
		max_ev_nb_steps_per_episode = []
		min_ev_nb_steps_per_episode = []
		mean_ev_reward = []
		nb_ev_paths = []

		nb_exp_steps = []
		nb_exp_episodes = []
		mean_exp_returns = []


		with open(path_to_CSV) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			line_count = 0
			for row in csv_reader:
				if line_count == 0: # Column names
					pass
					line_count += 1
				else:
					mean_ev_returns.append(float(row[78]))
					std_ev_returns.append(float(row[79]))
					max_ev_returns.append(float(row[80]))
					min_ev_returns.append(float(row[81]))
					mean_ev_reward.append(float(row[74]))
					mean_ev_nb_steps_per_episode.append(float(row[70]))
					std_ev_nb_steps_per_episode.append(float(row[71]))
					max_ev_nb_steps_per_episode.append(float(row[72]))
					min_ev_nb_steps_per_episode.append(float(row[73]))
					nb_ev_paths.append(float(row[86]))

					nb_exp_steps.append(float(row[30]))
					nb_exp_episodes.append(float(row[31]))
					mean_exp_returns.append(float(row[40]))

					line_count += 1

		data = {}
		data['mean_ev_returns'] =  mean_ev_returns
		data['std_ev_returns'] =  std_ev_returns
		data['max_ev_returns'] =  max_ev_returns
		data['min_ev_returns'] =  min_ev_returns
		data['mean_ev_reward'] =  mean_ev_reward
		data['mean_ev_nb_steps_per_episode'] =  mean_ev_nb_steps_per_episode
		data['std_ev_nb_steps_per_episode'] =  std_ev_nb_steps_per_episode
		data['max_ev_nb_steps_per_episode'] =  max_ev_nb_steps_per_episode
		data['min_ev_nb_steps_per_episode'] =  min_ev_nb_steps_per_episode
		data['nb_ev_paths'] =  nb_ev_paths
		data['nb_exp_steps'] =  nb_exp_steps
		data['nb_exp_episodes'] =  nb_exp_episodes
		data['mean_exp_returns'] =  mean_exp_returns

		## Finally trained networks
		path_to_pkl = os.path.join(path_to_load, 'params.pkl')
		SAC_nets = torch.load(path_to_pkl)
		policy = SAC_nets['trainer/policy']

		return data, policy


	def save_data(self, data, date, path_to_save):

		nb_exp_steps = data['nb_exp_steps']
		mean_ev_returns = data['mean_ev_returns']
		mean_exp_returns = data['mean_exp_returns']

		mat_ev_ret = [list (a) for a in zip(mean_ev_returns, nb_exp_steps)]
		mat_exp_ret	= [list (a) for a in zip(mean_exp_returns, nb_exp_steps)]

		print(tuple(mat_ev_ret))

		test_path = os.path.join(path_to_save, 'training_data', 'runs', 'test')
		train_path = os.path.join(path_to_save, 'training_data', 'runs','train')

		ut.makeDir(test_path)
		ut.makeDir(train_path)

		path_ev_ret = os.path.join(test_path, '{}_avg_returns_test.npy'.format(date.strftime("%Y-%m-%d_%H-%M-%S")))
		path_exp_ret = os.path.join(train_path, '{}_avg_returns_train.npy'.format(date.strftime("%Y-%m-%d_%H-%M-%S")))



		np.save(path_ev_ret, mat_ev_ret)
		np.save(path_exp_ret, mat_exp_ret)


	def save_policy(self, policy, path_to_save):
		dict_to_save = policy.state_dict()
		path = os.path.join(path_to_save, 'actor_model.pth')
		print('Saving model to {}'.format(path))
		torch.save(dict_to_save, path)

	def make_path(self, name_of_exp, date, computer):

		if computer == 'local':
			results_folder = 'local_results'
		elif computer == 'mila' or computer == 'transfer':
			results_folder = 'cluster_results'

		end_path = '{}_{}_0000--s-0'.format(name_of_exp, date.strftime("%Y_%m_%d_%H_%M_%S"))
		path_to_load = os.path.join('..', 'rlkit', 'data', name_of_exp, end_path)
		path_to_save = os.path.join('..', results_folder, 'rl', name_of_exp)

		return path_to_load, path_to_save






if __name__ == '__main__':
	date = datetime.datetime(2020,6,1,14,37,9)

	data_processor = RLKitDataParser()

	path_to_load, path_to_save = data_processor.make_path('name-of-experiment', date, 'local')
	
	print('Path to load: {}'.format(path_to_load))
	print('Path to save: {}'.format(path_to_save))

	data, policy = data_processor.import_data(path_to_load)
	data_processor.save_data(data, date, path_to_save)
	data_processor.save_policy(policy, path_to_save)
