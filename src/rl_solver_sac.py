from pytorch_soft_actor_critic.sac import *
import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from typing import NamedTuple
from pytorch_soft_actor_critic.sac import SAC
from tensorboardX import SummaryWriter
from pytorch_soft_actor_critic.replay_memory import ReplayMemory
import utils as ut
import sys


class SAC_args(NamedTuple):
	gamma: float
	tau: float
	lr: float
	alpha: float
	automatic_entropy_tuning: bool
	policy: str
	target_update_interval: int	
	cuda : bool
	hidden_size : int


class SAC_Solver():
	def __init__(
		self, 
		env,
		test_env,
		config_path,
		config_rl
		):
		
		self.env = env
		
		### RL parameters
		self.env_name = config_rl['env_name']
		self.test_env_name = config_rl['test_env_name']
		self.seed = config_rl['seed']
		self.policy = config_rl['policy']
		self.eval = config_rl['eval_']
		self.gamma = config_rl['gamma']
		self.tau = config_rl['tau']
		self.lr = config_rl['lr']
		self.alpha = config_rl['alpha']
		self.automatic_entropy_tuning = config_rl['automatic_entropy_tuning']
		self.batch_size = config_rl['batch_size']
		self.num_episodes = config_rl['num_episodes']
		self.max_steps_episode = config_rl['max_steps_episode']
		self.hidden_size = config_rl['hidden_size']
		self.updates_per_step = config_rl['updates_per_step']
		self.start_steps = config_rl['start_steps']
		self.target_update_interval = config_rl['target_update_interval']
		self.replay_size = config_rl['replay_size']
		self.cuda = config_rl['cuda']

		self.sac_args = SAC_args(self.gamma, self.tau, self.lr, self.alpha, self.automatic_entropy_tuning, self.policy, self.target_update_interval, self.cuda, self.hidden_size)

		self.env = env
		self.test_env = test_env
		#self.env.env.reset()
		torch.manual_seed(self.seed)
		np.random.seed(self.seed)
		self.env.seed(self.seed)
		print(self.env.observation_space.shape)
		print(self.env.observation_space.shape[0])
		self.agent = SAC(self.env.observation_space.shape[0], self.env.action_space, self.sac_args)

		### Path parameters
		self.training_data_path = config_path['training_data_path']
		ut.makeDir(self.training_data_path + '/runs')
		ut.makeDir(self.training_data_path + '/runs/train')
		ut.makeDir(self.training_data_path + '/runs/test')
		self.vis_path = self.training_data_path + '/runs/vis/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), self.env_name, self.policy, "autotune" if self.automatic_entropy_tuning else "")
		self.train_res_path = self.training_data_path + '/runs/train/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), self.env_name, self.policy, "autotune" if self.automatic_entropy_tuning else "")
		self.test_res_path = self.training_data_path + '/runs/test/{}_SAC_{}_{}_{}_test'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), self.env_name, self.policy, "autotune" if self.automatic_entropy_tuning else "")
		self.critic_model_path = config_path['critic_model_path']
		self.actor_model_path = config_path['actor_model_path']
		self.writer = SummaryWriter(logdir=self.vis_path)
		self.memory = ReplayMemory(self.replay_size)

		self.reward_mem = []
		self.test_mem = []



	def test(self, i_episode):
		avg_reward = 0.
		episodes = 10
		for _  in range(episodes):
			state = self.test_env.reset()
			episode_reward = 0
			episode_steps = 0
			done = False
			while not done:
				action = self.agent.select_action(state, eval=True)
				if self.test_env_name == 'cartpole':
					next_state, reward, done, _ = self.test_env.step(action)
				elif self.test_env_name == 'duckietown':
					next_state, reward, done, _ = self.test_env.step(action)
				episode_reward += reward
				state = next_state

				episode_steps += 1
				if episode_steps%1000 == 0:
					print("Episode step: {}".format(episode_steps))
				if episode_steps > self.max_steps_episode:
					print('Ending episode because more than {} steps.'.format(self.max_steps_episode))
					done = True
					self.env.reset()

			avg_reward += episode_reward
		avg_reward /= episodes


		self.writer.add_scalar('avg_reward/test', avg_reward, i_episode)
		self.test_mem.append((avg_reward, i_episode))

		print("----------------------------------------")
		print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
		print("----------------------------------------")		


	def train(self):

		# Training Loop
		total_numsteps = 0
		updates = 0

		# Episodes
		for i_episode in range(1, self.num_episodes + 1):
			episode_reward = 0
			episode_steps = 0
			done = False
			state = self.env.reset()

			# Steps
			while not done:
				if self.start_steps > total_numsteps:
					action = self.env.action_space.sample()  # Sample random action
				else:
					action = self.agent.select_action(state)  # Sample action from policy

				if len(self.memory) > self.batch_size:
					# Number of updates per step in environment
					for i in range(self.updates_per_step):
						# Update parameters of all the networks
						critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self.agent.update_parameters(self.memory, self.batch_size, updates)
						#self.writer.add_scalar('loss/critic_1', critic_1_loss, updates)
						#self.writer.add_scalar('loss/critic_2', critic_2_loss, updates)
						#self.writer.add_scalar('loss/policy', policy_loss, updates)
						#self.writer.add_scalar('loss/entropy_loss', ent_loss, updates)
						#self.writer.add_scalar('entropy_temprature/alpha', alpha, updates)
						updates += 1

				next_state, reward, done, _ = self.env.step(action) # Step
				episode_steps += 1
				if episode_steps%1000 == 0:
					print("Episode step: {}".format(episode_steps))
				if episode_steps > self.max_steps_episode:
					print('Ending episode because more than 3000 steps.')
					done = True
					self.env.reset()
				total_numsteps += 1
				episode_reward += reward

				# Ignore the "done" signal if it comes from hitting the time horizon.
				# (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
				mask = 1 if episode_steps == self.max_steps_episode else float(not done)

				self.memory.push(state, action, reward, next_state, mask) # Append transition to self.memory

				state = next_state

			self.writer.add_scalar('reward/train', episode_reward, i_episode)
			self.reward_mem.append((episode_reward, i_episode))

			print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

			if i_episode % 10 == 0 and self.eval == True:
				self.test(i_episode)
				self.save()
				self.save_results()

		self.save_results()
		self.env.close()

	def save(self):
		self.agent.save_model(self.actor_model_path, self.critic_model_path)

	def save_results(self):
		np.save(self.test_res_path,self.test_mem)
		np.save(self.train_res_path, self.reward_mem)
		

if __name__ == '__main__':
	# Loading general config file

	parser = argparse.ArgumentParser(description='This script trains a CNN on the images and label created from generate_images.py.')
	parser.add_argument('-c','--computer', help='Computer on which the script is launched.',required=True)
	parser.add_argument('-t','--train_environment',help='Training environment', required=True)
	parser.add_argument('-e','--environment',help='Testing environment', required=True)
	parser.add_argument('-g','--generation_mode', help='Mode of image generation. Required for any training environment with "CNN" inside their name!', required = False, default = None)
	parser.add_argument('-n','--model_name', help='CNN model name from which the environment predicts the reward. Required for any training environment with "CNN" inside their name!', required = False, default = None)
	args = parser.parse_args()
	
	computer = ut.getComputer(args.computer)
	test_env_name = ut.getImEnv(args.environment)
	env_name = ut.getTrainEnv(args.train_environment)
	gen_mode = ut.getGenMode(args.generation_mode)
	model_name = ut.getModelName(args.model_name)

	config = ut.loadYAMLFromFile('config_' + test_env_name + '.yaml')

	# Building config_path
	config_path = {}

	### Environments
	config['rl']['env_name'] = env_name
	config['rl']['test_env_name'] = test_env_name
	### Changing temp_root if necessary
	temp_root = ''
	if computer == 'mila':
		# Getting local disk info
		temp_root = os.environ['SLURM_TMPDIR'] + '/'

	### RL saving paths during usage (data training and model)
	use_rl_path = os.path.join(temp_root, config['paths'][computer]['rl'])
	if model_name:
		exp_name = env_name + '_' + model_name
	else:
		exp_name = env_name
	use_rl_env_path = os.path.join(use_rl_path, exp_name)

	print('use_rl_env_path: {}'.format(use_rl_env_path))

	rl_training_data_path = os.path.join(use_rl_env_path, 'training_data')
	ut.makeDir(rl_training_data_path)
	config_path['training_data_path'] = rl_training_data_path
	config_path['critic_model_path'] = os.path.join(use_rl_env_path, 'critic_model.pth')
	config_path['actor_model_path'] =os.path.join(use_rl_env_path, 'actor_model.pth')

	### Training environment

	#### Importing library
	if test_env_name == 'cartpole':
		from cartpole_mod_env import *
	elif test_env_name == 'duckietown' or test_env_name == 'duckietown_cam':
		from duckietown_mod_env import *

	#### Loading environment
	##### Cartpole
	if env_name == 'CP_CNN_State':
		if not model_name or not gen_mode:
			raise ValueError("model_name and gen_mode is needed under options -n and -g for {}".format(env_name))
		cnn_use_path = os.path.join(temp_root, config['paths'][computer]['cnn'], test_env_name, 'State', gen_mode, model_name, 'model.pth')
		# If Mila, copy model from save folder to local disk
		if computer == 'mila':
			cnn_save_path =  os.path.join(config['paths'][computer]['save_cnn'], test_env_name, 'State', gen_mode, model_name, 'model.pth')
			ut.copyAndOverwriteFile(cnn_save_path, cnn_use_path)
		env = S_CNNDenseRewardWrapperCartpole(ContinuousActionWrapperCartpole(gym.make('CartPole-v0')), model_path = cnn_use_path)
	elif env_name == 'CP_CNN_Reward':
		if not model_name or not gen_mode:
			raise ValueError("model_name and gen_mode is needed under options -n and -g for {}".format(env_name))
		cnn_use_path = os.path.join(temp_root, config['paths'][computer]['cnn'], test_env_name, 'Reward', gen_mode, model_name, 'model.pth')
		# If Mila, copy model from save folder to local disk
		if computer == 'mila':
			cnn_save_path =  os.path.join(config['paths'][computer]['save_cnn'], test_env_name, 'Reward', gen_mode, model_name, 'model.pth')
			ut.copyAndOverwriteFile(cnn_save_path, cnn_use_path)
		env = R_CNNDenseRewardWrapperCartpole(ContinuousActionWrapperCartpole(gym.make('CartPole-v0')), model_path = cnn_use_path)
	elif env_name == 'CP_Noisy_Reward_10':
		env = NoisyDenseRewardWrapperCartpole(ContinuousActionWrapperCartpole(gym.make('CartPole-v0')), std_dev = 0.1)
	elif env_name == 'CP_Noisy_Reward_20':
		env = NoisyDenseRewardWrapperCartpole(ContinuousActionWrapperCartpole(gym.make('CartPole-v0')), std_dev = 0.2)
	elif env_name == 'CP_Noisy_Reward_30':
		env = NoisyDenseRewardWrapperCartpole(ContinuousActionWrapperCartpole(gym.make('CartPole-v0')), std_dev = 0.3)
	elif env_name == 'CP_GT_Reward':
		env = DenseRewardWrapperCartpole(ContinuousActionWrapperCartpole(gym.make('CartPole-v0')))
	elif env_name == 'CP_Original_Reward':
		env = ContinuousActionWrapperCartpole(gym.make('CartPole-v0'))
	
	##### Duckietown
	elif env_name == 'DT_GT_Reward':
		env = DuckietownEnv()
		env = DTDroneImageGenerator(env)
		env = DTDistAngleObsWrapper(env)
		env = DTLaneFollowingRewardWrapper(env)
		env = DTConstantVelWrapper(env)
	elif env_name == 'DT_Noisy_Reward_10':
		env = DuckietownEnv()
		env = DTDroneImageGenerator(env)
		env = DTDistAngleObsWrapper(env)
		env = DTLaneFollowingRewardWrapper(env)
		env = DTConstantVelWrapper(env)
		env = DTNoisyRewardWrapper(env, 0.10)
	elif env_name == 'DT_Noisy_Reward_20':
		env = DuckietownEnv()
		env = DTDroneImageGenerator(env)
		env = DTDistAngleObsWrapper(env)
		env = DTLaneFollowingRewardWrapper(env)
		env = DTConstantVelWrapper(env)
		env = DTNoisyRewardWrapper(env, 0.20)
	elif env_name == 'DT_Noisy_Reward_30':
		env = DuckietownEnv()
		env = DTDroneImageGenerator(env)
		env = DTDistAngleObsWrapper(env)
		env = DTLaneFollowingRewardWrapper(env)
		env = DTConstantVelWrapper(env)
		env = DTNoisyRewardWrapper(env, 0.30)
	elif env_name == 'DT_Noisy_Reward_40':
		env = DuckietownEnv()
		env = DTDroneImageGenerator(env)
		env = DTDistAngleObsWrapper(env)
		env = DTLaneFollowingRewardWrapper(env)
		env = DTConstantVelWrapper(env)
		env = DTNoisyRewardWrapper(env, 0.40)
	elif env_name == 'DT_Noisy_Reward_50':
		env = DuckietownEnv()
		env = DTDroneImageGenerator(env)
		env = DTDistAngleObsWrapper(env)
		env = DTLaneFollowingRewardWrapper(env)
		env = DTConstantVelWrapper(env)
		env = DTNoisyRewardWrapper(env, 0.50)	
	elif env_name == 'DT_Noisy_Reward_100':
		env = DuckietownEnv()
		env = DTDroneImageGenerator(env)
		env = DTDistAngleObsWrapper(env)
		env = DTLaneFollowingRewardWrapper(env)
		env = DTConstantVelWrapper(env)
		env = DTNoisyRewardWrapper(env, 1)	
	elif env_name == 'DT_Noisy_Reward_500':
		env = DuckietownEnv()
		env = DTDroneImageGenerator(env)
		env = DTDistAngleObsWrapper(env)
		env = DTLaneFollowingRewardWrapper(env)
		env = DTConstantVelWrapper(env)
		env = DTNoisyRewardWrapper(env, 5)

	elif env_name == 'DT_R_CNN_Reward':
		env = DuckietownEnv()
		env = DTDroneImageGenerator(env)
		env = DTDistAngleObsWrapper(env)
		env = DTConstantVelWrapper(env)
		cnn_folder = os.path.join(temp_root, config['paths'][computer]['cnn'], 'duckietown', 'Reward', gen_mode, model_name)
		# If Mila, copy model from save folder to local disk
		if computer == 'mila':
			cnn_save_folder =  os.path.join(config['paths'][computer]['save_cnn'], 'duckietown', 'Reward', gen_mode, model_name)
			cnn_use_file = os.path.join(cnn_folder, 'latest_model.pth')
			cnn_save_file = os.path.join(cnn_save_folder, 'latest_model.pth')
			cnn_use_params = os.path.join(cnn_folder, 'cnn_params.yaml')
			cnn_save_params = os.path.join(cnn_save_folder, 'cnn_params.yaml')
			ut.copyAndOverwriteFile(cnn_save_file, cnn_use_file)
			ut.copyAndOverwriteFile(cnn_save_params, cnn_use_params)
		env = DT_R_CNN_RewardWrapper(env, cnn_folder, 'resnet18')

	elif env_name == 'DT_S_CNN_Reward':
		env = DuckietownEnv()
		env = DTDroneImageGenerator(env)
		env = DTDistAngleObsWrapper(env)
		env = DTConstantVelWrapper(env)
		cnn_folder = os.path.join(temp_root, config['paths'][computer]['cnn'], 'duckietown', 'State', gen_mode, model_name)
		# If Mila, copy model from save folder to local disk
		if computer == 'mila':
			cnn_save_folder =  os.path.join(config['paths'][computer]['save_cnn'], 'duckietown', 'State', gen_mode, model_name)
			cnn_use_file = os.path.join(cnn_folder, 'latest_model.pth')
			cnn_save_file = os.path.join(cnn_save_folder, 'latest_model.pth')
			cnn_use_params = os.path.join(cnn_folder, 'cnn_params.yaml')
			cnn_save_params = os.path.join(cnn_save_folder, 'cnn_params.yaml')
			ut.copyAndOverwriteFile(cnn_save_file, cnn_use_file)
			ut.copyAndOverwriteFile(cnn_save_params, cnn_use_params)
		env = DT_S_CNN_RewardWrapper(env, cnn_folder, 'resnet18')

	elif env_name == 'DT_Ssplit_CNN_Reward':
		raise NotImplementedError('Environment DT_Ssplit_CNN_Reward is not completely implemented yet - missing a solution for model_name')
		env = DuckietownEnv()
		env = DTDroneImageGenerator(env)
		env = DTDistAngleObsWrapper(env)
		env = DTConstantVelWrapper(env)
		### Error here; there is supposed to be two model names - but the current implementation does not allow this
		cnn_folder_d = os.path.join(temp_root, config['paths'][computer]['cnn'], 'duckietown', 'Distance', gen_mode, model_name)
		cnn_folder_a = os.path.join(temp_root, config['paths'][computer]['cnn'], 'duckietown', 'Angle', gen_mode, model_name)
		
		if computer == 'mila': # To adapt!
			cnn_save_folder =  os.path.join(config['paths'][computer]['save_cnn'], 'duckietown', 'Distance', gen_mode, model_name)
			cnn_use_file = os.path.join(cnn_folder, 'latest_model.pth')
			cnn_save_file = os.path.join(cnn_save_folder, 'latest_model.pth')
			cnn_use_params = os.path.join(cnn_folder, 'cnn_params.yaml')
			cnn_save_params = os.path.join(cnn_save_folder, 'cnn_params.yaml')
			ut.copyAndOverwriteFile(cnn_save_file, cnn_use_file)
			ut.copyAndOverwriteFile(cnn_save_params, cnn_use_params)
		env = DT_Ssplit_CNN_RewardWrapper(env, cnn_folder_d, 'resnet18', cnn_folder_a, 'resnet18')

	
	##### Duckietown Cam
	elif env_name == 'DTC_GT_Reward':
		env = DTConstantVelWrapper(DTLaneFollowingRewardWrapper(DuckietownEnv()))
		env = ResizeWrapper(env)
		env = NormalizeWrapper(env)
		env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
		print(env.observation_space)
	##### Others
	elif env_name == 'HalfCheetah-v2':
		env = env = gym.make('HalfCheetah-v2')

	else:
		raise ValueError('Environment name {} as written in config.yaml is unknown.'.format(env_name))
	print("Training environment initialized.")

	### Testing environment
	if test_env_name == "cartpole":
		test_env = DenseRewardWrapperCartpole(ContinuousActionWrapperCartpole(gym.make('CartPole-v0')))
	elif test_env_name == 'duckietown':
		test_env = DTConstantVelWrapper(DTLaneFollowingRewardWrapper(DTDistAngleObsWrapper(DTDroneImageGenerator(DuckietownEnv()))))#DTLaneFollowingRewardWrapper(DuckietownEnv())#
	elif test_env_name == 'duckietown_cam':
		test_env = DTConstantVelWrapper(DTLaneFollowingRewardWrapper(DTDroneImageGenerator(DuckietownEnv())))#DTLaneFollowingRewardWrapper(DuckietownEnv())#
	else:
		raise ValueError('Test Environment name {} as written in config.yaml is unknown or unexpected (should be CP_GT_Reward).'.format(env_name))

	trainer = SAC_Solver(env, test_env, config_path, config['rl'])#env_name = 'CartPole-v0')
	print("Initialized trainer")

	trainer.train()
	trainer.save()

	### Copying results to save repo
	if computer == 'mila':
		# Copying RL model and data from local disk to save repo (tmp1/maivincent)
		save_rl_path = config['paths'][computer]['save_rl']
		ut.copyAndOverwrite(use_rl_path, save_rl_path)
		print("Copying results: done!")