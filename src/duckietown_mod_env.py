from gym_duckietown.simulator import Simulator
from gym_duckietown.wrappers import *
from gym_duckietown.envs.duckietown_env import DuckietownEnv

import gym
import numpy as np 

from gym import spaces
import numpy as np 
from train_model import Rescale, NormalizeImage, UnnormalizeLabel
import torch
import torch.nn as nn
import torchvision.models as models

import logging
from utils import *
import sys
import os
import cv2


logger = logging.getLogger("gym-duckietown")
logger.disabled = True



def state2reward(dist, angle):
	reward = 1 - 1/2*np.abs(angle) - 1/2*np.abs(dist/0.15)**2
	return reward

class DTDistAngleObsWrapper(gym.Wrapper):
	def __init__(self, env):
		gym.Wrapper.__init__(self, env)
		self.observation_space = spaces.Box(low=-4, high=4, shape=(2,)) 

	def reset(self):
		self.env.reset()
		lane_pose = self.env.get_lane_pos2(self.env.cur_pos, self.env.cur_angle)
		dist = lane_pose.dist        # Distance to lane center. Left is negative, right is positive.
		angle = angleLimit(lane_pose.angle_rad)  # Angle from straight, in radians. Left is negative, right is positive.
		observation = np.array([dist, angle])
		return observation

	def step(self, action):
		_, reward, done, info = self.env.step(action)
		 # Getting the pose
		lane_pose = self.env.get_lane_pos2(self.env.cur_pos, self.env.cur_angle)
		dist = lane_pose.dist        # Distance to lane center. Left is negative, right is positive.
		angle = angleLimit(lane_pose.angle_rad)  # Angle from straight, in radians. Left is negative, right is positive.
		observation = np.array([dist, angle])
		return observation, reward, done, info

	def render(self, mode):
		return self.env.render(mode)

class DTLaneFollowingRewardWrapper(gym.Wrapper):
	def step(self, action):
		observation, reward, done, info = self.env.step(action)
		 # Getting the pose
		lane_pose = self.env.get_lane_pos2(self.env.cur_pos, self.env.cur_angle)
		dist = lane_pose.dist        # Distance to lane center. Left is negative, right is positive.
		angle = angleLimit(lane_pose.angle_rad)  # Angle from straight, in radians. Left is negative, right is positive.
		reward = self.reward([dist, angle])
		return observation, reward, done, info

	def reward(self, state):
		dist = state[0]
		angle = angleLimit(state[1])
		return state2reward(dist, angle)

	def render(self, mode):
		return self.env.render(mode)

class DTConstantVelWrapper(gym.Wrapper):
	def __init__(self, env):
		gym.Wrapper.__init__(self, env)
		self.action_space = spaces.Box(low=-2, high=2, shape=(1,)) 

	def step(self, action):
		velocity = 0.2
		action = [velocity, action[0]]
		observation, reward, done, info = self.env.step(action)
		return observation, reward, done, info

	def render(self, mode):
		return self.env.render(mode)

class DTNoisyRewardWrapper(gym.Wrapper):
	def __init__(self, env, std_dev):
		super().__init__(env)
		self.std_dev = std_dev

	def step(self, action):
		observation, true_reward, done, info = self.env.step(action)
		noise = np.random.normal(loc=0.0, scale=self.std_dev)
		reward = true_reward + noise
		return observation, reward, done, info

class DTDroneImageGenerator(gym.Wrapper):
	def __init__(self, env, drone_sim = 'random', drone_angle_follow = False):		# drone_sim: 'random' simulates the dynamics of the drone, 'random_0' has no drone dynamics (angle is always same and bot in the center), 'random_1' has bot in center but changing angle, 'random_2' has same angle but bot not in center  
		gym.Wrapper.__init__(self, env)
		self.drone_sim = drone_sim
		self.x_lim = 0.5
		self.y_lim = 0.5
		self.z_lim = 0.5

		self.x_speed_lim = 0.005
		self.y_speed_lim = 0.005
		self.z_speed_lim = 0.005
		self.ang_speed_lim = 0.002

		self.drone_angle_follow = drone_angle_follow

		self.init_drone_state()

	def reset(self):
		obs = self.env.reset()
		self.init_drone_state()
		return obs

	def init_drone_state(self):
		# Position
		if self.drone_sim == 'random' or self.drone_sim == 'random_2':
			self.zoom_speed = 2 *self.z_speed_lim * np.random.random_sample() - self.z_speed_lim
			self.x_speed = 2 *self.x_speed_lim * np.random.random_sample() - self.x_speed_lim
			self.y_speed = 2 *self.y_speed_lim * np.random.random_sample() - self.y_speed_lim

			self.zoom = 2 *self.z_lim * np.random.random_sample() - self.z_lim
			self.x_push = 2 *self.x_lim * np.random.random_sample() - self.x_lim
			self.y_push = 2 *self.y_lim * np.random.random_sample() - self.y_lim

		elif self.drone_sim == 'random_0' or self.drone_sim == 'random_1':
			self.zoom_speed, self.x_speed, self.y_speed = 0.0, 0.0, 0.0
			self.zoom, self.x_push, self.y_push = 0.0, 0.0, 0.0	

		# Angle
		if self.drone_sim == 'random' or self.drone_sim == 'random_1':
			self.ang_speed = 2 *self.ang_speed_lim * np.random.random_sample() - self.ang_speed_lim

			self.angle = angleLimit(2* np.pi* np.random.random_sample())

		elif self.drone_sim == 'random_0' or self.drone_sim == 'random_2':
			self.ang_speed = 0.0
			self.angle = angleLimit(0.0)

		elif self.drone_sim == 'random_3':
			self.ang_speed = 0.0
			self.angle = angleLimit(self.env.cur_angle)

	def propagate_drone_dynamics(self):
		# Position
		if self.drone_sim == 'random' or self.drone_sim == 'random_2':
			x_acc = np.random.normal(loc= - 0.001*self.x_push, scale=0.0003)
			y_acc = np.random.normal(loc= - 0.001*self.y_push, scale=0.0003)
			z_acc = np.random.normal(loc= - 0.001*self.zoom, scale=0.0003)

			self.zoom_speed = limitAbsoluteVal(self.zoom_speed + z_acc, self.z_speed_lim)		
			self.x_speed = limitAbsoluteVal(self.x_speed + x_acc, self.x_speed_lim)		
			self.y_speed = limitAbsoluteVal(self.y_speed + y_acc, self.y_speed_lim)	

			self.zoom = limitAbsoluteVal(self.zoom + self.zoom_speed, self.z_lim)
			self.x_push = limitAbsoluteVal(self.x_push + self.x_speed, self.x_lim)
			self.y_push = limitAbsoluteVal(self.y_push + self.y_speed, self.y_lim)

		elif self.drone_sim == 'random_0' or self.drone_sim == 'random_1' or self.drone_sim == 'random_3':
			self.zoom_speed, self.x_speed, self.y_speed = 0.0, 0.0, 0.0
			self.zoom, self.x_push, self.y_push = 0.0, 0.0, 0.0	

		# Angle
		if self.drone_sim == 'random' or self.drone_sim == 'random_1':
			ang_acc = np.random.normal(loc= - 0.001*self.ang_speed, scale=0.0003)
			self.ang_speed = limitAbsoluteVal(self.ang_speed + ang_acc, self.ang_speed_lim)
			self.angle = angleLimit(self.angle + self.ang_speed)
		elif self.drone_sim == 'random_0' or self.drone_sim == 'random_2':
			self.ang_speed = 0.0
			self.angle = angleLimit(0.0)
		elif self.drone_sim == 'random_3':
			self.ang_speed = 0.0
			self.angle = angleLimit(self.env.cur_angle)


	def set_state(self, state):
		raise NotImplementedError("DuckieTown environment set_state(self, state) not implemented yet.") 

	def render(self, mode = 'drone'):
		mode = 'drone'
		self.propagate_drone_dynamics()
		image = self.env.render(mode = mode, drone_params = [self.x_push, self.zoom, self.y_push, self.angle], drone_angle_follow = self.drone_angle_follow)
		return image

	def get_drone_params(self):
		return [self.x_push, self.zoom, self.y_push, self.angle]


class DT_R_CNN_RewardWrapper(gym.Wrapper):
	def __init__(self, env, cnn_folder_path, cnn_type):
		super().__init__(env)
		params_file = os.path.join(cnn_folder_path, 'cnn_params.yaml')
		self.model_params = loadYAMLFromFile(params_file)
		self.model = self.loadModel(cnn_folder_path, cnn_type)
		self.config = loadYAMLFromFile('config_duckietown.yaml')

	def loadModel(self, cnn_folder_path, cnn_type):
		if cnn_type == 'resnet18':
			model = models.resnet18(pretrained=False, num_classes=1)
		else:
			raise NotImplementedError("DTRCNNRewardWrapper not implemented for another model type than resnet18.")
		model_path = os.path.join(cnn_folder_path, 'latest_model.pth')
		model.load_state_dict(torch.load(model_path))
		print('Loaded model ' + model_path)
		model.eval()
		return model

	def step(self, action):
		observation, reward, done, info = self.env.step(action)
		reward = self.reward()
		return observation, reward, done, info

	def reward(self):
		image = self.render()
		image = self.transformImage(image)
		reward = self.model.forward(image.float())
		reward = reward.detach().numpy()[0][0] #If bug here, remove one [0]
		reward = UnnormalizeLabel('duckietown', 'Reward', reward)
		return reward

	def transformImage(self, image):
		image = image.copy()
		image = NormalizeImage(image, self.model_params['dataset_stats'])
		new_size = (self.config['cnn']['rescale_size'][0], self.config['cnn']['rescale_size'][1])
		image = rescaleImage(image, new_size)
		image = image.transpose((2, 0, 1))
		image = torch.from_numpy(image)
		image = image.unsqueeze(0)
		return image

class DT_S_CNN_RewardWrapper(gym.Wrapper):
	def __init__(self, env, cnn_folder_path, cnn_type):
		super().__init__(env)
		params_file = os.path.join(cnn_folder_path, 'cnn_params.yaml')
		self.model_params = loadYAMLFromFile(params_file)
		self.model = self.loadModel(cnn_folder_path, cnn_type)
		self.config = loadYAMLFromFile('config_duckietown.yaml')

	def loadModel(self, cnn_folder_path, cnn_type):
		if cnn_type == 'resnet18':
			model = models.resnet18(pretrained=False, num_classes=3)
		else:
			raise NotImplementedError("DTRCNNRewardWrapper not implemented for another model type than resnet18.")
		model_path = os.path.join(cnn_folder_path, 'latest_model.pth')
		model.load_state_dict(torch.load(model_path))
		print('Loaded model ' + model_path)
		model.eval()
		return model

	def step(self, action):
		observation, reward, done, info = self.env.step(action)
		reward = self.reward()
		return observation, reward, done, info

	def reward(self):
		image = self.render()
		image = self.transformImage(image)
		state_pred = self.model.forward(image.float())
		state_pred = state_pred.detach().numpy()[0]
		state = UnnormalizeLabel('duckietown', 'State', state_pred)
		reward = state2reward(state[0], state[1])
		return reward

	def transformImage(self, image):
		image = image.copy()
		image = NormalizeImage(image, self.model_params['dataset_stats'])
		new_size = (self.config['cnn']['rescale_size'][0], self.config['cnn']['rescale_size'][1])
		image = rescaleImage(image, new_size)
		image = image.transpose((2, 0, 1))
		image = torch.from_numpy(image)
		image = image.unsqueeze(0)
		return image

class DT_Ssplit_CNN_RewardWrapper(gym.Wrapper):
	def __init__(self, env, cnn_folder_path_d, cnn_type_d, cnn_folder_path_a, cnn_type_a):
		super().__init__(env)
		# Distance CNN
		params_file_d = os.path.join(cnn_folder_path_d, 'cnn_params.yaml')
		self.model_params_d = loadYAMLFromFile(params_file_d)
		self.model_d = self.loadModel(cnn_folder_path_d, cnn_type_d, 1)

		# Angle CNN
		params_file_a = os.path.join(cnn_folder_path_a, 'cnn_params.yaml')
		self.model_params_a = loadYAMLFromFile(params_file_a)
		self.model_a = self.loadModel(cnn_folder_path_a, cnn_type_a, 2)
		
		self.config = loadYAMLFromFile('config_duckietown.yaml')

	def loadModel(self, cnn_folder_path, cnn_type, num_classes):
		if cnn_type == 'resnet18':
			model = models.resnet18(pretrained=False, num_classes=num_classes)
		else:
			raise NotImplementedError("DTRCNNRewardWrapper not implemented for another model type than resnet18.")
		model_path = os.path.join(cnn_folder_path, 'latest_model.pth')
		model.load_state_dict(torch.load(model_path))
		print('Loaded model ' + model_path)
		model.eval()
		return model

	def step(self, action):
		observation, reward, done, info = self.env.step(action)
		reward = self.reward()
		return observation, reward, done, info

	def reward(self):
		image = self.render()
		image = self.transformImage(image)

		# Distance
		dist_pred = self.model_d.forward(image.float())
		dist_pred = dist_pred.detach().numpy()[0]
		distance = UnnormalizeLabel('duckietown', 'Distance', dist_pred)

		# Angle
		angle_pred = self.model_a.forward(image.float())
		angle_pred = angle_pred.detach().numpy()[0]
		angle = UnnormalizeLabel('duckietown', 'Angle', angle_pred)		

		reward = state2reward(distance, angle)

		return reward

	def transformImage(self, image):
		image = image.copy()
		image = NormalizeImage(image, self.model_params['dataset_stats'])
		new_size = (self.config['cnn']['rescale_size'][0], self.config['cnn']['rescale_size'][1])
		image = rescaleImage(image, new_size)
		image = image.transpose((2, 0, 1))
		image = torch.from_numpy(image)
		image = image.unsqueeze(0)
		return image



class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(120, 160, 3)):
        super(ResizeWrapper, self).__init__(env)
        self.observation_space.shape = shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            shape,
            dtype=self.observation_space.dtype)
        self.shape = shape

    def observation(self, observation):
        from PIL import Image
        return np.array(Image.fromarray(observation).resize(self.shape[0:2]))


class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)


class ImgWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ImgWrapper, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)

class GTDenseRewardInfoWrapperDT(gym.Wrapper):


	def gt_reward(self):
		lane_pose = self.env.get_lane_pos2(self.env.cur_pos, self.env.cur_angle)
		dist = lane_pose.dist        # Distance to lane center. Left is negative, right is positive.
		angle = angleLimit(lane_pose.angle_rad)
		gt_reward = state2reward(dist, angle)
		return gt_reward

	def step(self, action):
		observation, reward, done, info = self.env.step(action)
		info['GT_reward'] = self.gt_reward()
		return observation, reward, done, info

#### Action wrappers
if __name__ == '__main__':


	sim = DTNoisyLaneFollowingRewardWrapper(DuckietownEnv(), 10)
	print('Environment loaded!')
	image = sim.render(mode = 'drone')

	input()