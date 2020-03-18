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
import time

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


class SAC_Viewer():
	def __init__(
		self, 
		env,
		config_path,
		config_rl
		):
		
		self.env = env
		
		### RL parameters
		self.env_name = config_rl['env_name']
		self.seed = config_rl['seed']
		self.policy = config_rl['policy']
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

		#self.env.env.reset()
		torch.manual_seed(self.seed)
		np.random.seed(self.seed)
		self.env.seed(self.seed)
		self.agent = SAC(self.env.observation_space.shape[0], self.env.action_space, self.sac_args)



		self.critic_model_path = config_path['critic_model_path']
		self.actor_model_path = config_path['actor_model_path']


		self.agent.load_model(self.actor_model_path, self.critic_model_path)

	def view_policy(self, nb_episodes):
		for ep_id  in range(nb_episodes):
			state = self.env.reset()
			print(state)
			episode_reward = 0
			episode_steps = 0
			done = False
			if self.env_name == 'duckietown':
				self.env.render(mode = 'drone')
			else:
				self.env.render()

			while not done:
				action = self.agent.select_action(state, eval=True)
				if self.env_name == 'cartpole':
					next_state, reward, done, _ = self.env.step(action)
				elif self.env_name == 'duckietown':
					next_state, reward, done, _ = self.env.step(action)
				episode_reward += reward
				state = next_state
				#print("{} : {} : {} : {}".format(episode_steps, state, action, reward))
				if self.env_name == 'duckietown':
					self.env.render(mode = 'drone')
				else:
					self.env.render()
				time.sleep(0.01)
				episode_steps += 1
				if episode_steps%1000 == 0:
					print("Episode step: {}".format(episode_steps))
				if episode_steps > self.max_steps_episode:
					print('Ending episode because more than {} steps.'.format(self.max_steps_episode))
					done = True
					self.env.reset()
			print("----------------------------------------")
			print("Episode: {}/{}, Final reward: {}".format(ep_id, nb_episodes, episode_reward))
			print("----------------------------------------")		







if __name__ == '__main__':
	# Loading config
	
	parser = argparse.ArgumentParser(description='This script trains a CNN on the images and label created from generate_images.py.')
	parser.add_argument('-c','--computer', help='Computer on which the script is launched.',required=True)
	parser.add_argument('-e','--environment',help='Testing environment', required=True)
	parser.add_argument('-g','--generation_mode', help='Mode of image generation. Required for any training environment with "CNN" inside their name!', required = False, default = None)
	parser.add_argument('-n','--model_name', help='Name of the RL model', required = True)
	
	args = parser.parse_args()
	
	computer = ut.getComputer(args.computer)
	env_name = ut.getImEnv(args.environment)
	gen_mode = ut.getGenMode(args.generation_mode)
	model_name = ut.getModelName(args.model_name)

	config = ut.loadYAMLFromFile('config_' + env_name + '.yaml')


	### RL paths (model)
	use_rl_env_path = os.path.join(config['paths'][computer]['rl'], model_name)

	print('use_rl_env_path: {}'.format(use_rl_env_path))

	# Building config_path
	config_path = {}
	config_path['critic_model_path'] = os.path.join(use_rl_env_path, 'critic_model.pth')
	config_path['actor_model_path'] =os.path.join(use_rl_env_path, 'actor_model.pth')

	config['rl']['env_name'] = env_name

	if env_name == 'cartpole':
		from cartpole_mod_env import *
		env = ContinuousActionWrapperCartpole(gym.make('CartPole-v0'))
		env = DenseRewardWrapperCartpole(ContinuousActionWrapperCartpole(gym.make('CartPole-v0')))
	elif env_name == 'duckietown' or env_name == 'duckietown_cam':
		from duckietown_mod_env import *
		env = DTConstantVelWrapper(DTLaneFollowingRewardWrapper(DTDistAngleObsWrapper(DTDroneImageGenerator(DuckietownEnv()))))#

	viewer = SAC_Viewer(env, config_path, config['rl'])
	viewer.view_policy(100)