import numpy as np
import cv2
import os
import csv
import gym
import argparse
import utils
import time
import sys
from cartpole_mod_env import *
from duckietown_mod_env import *

class image_generator():
	# Calling create_images generates images from the OpenAI Gym Cartpole environment specified in self.init

	def __init__(self, config, environment, gen_mode):
		self.image_height = config['image']['height']
		self.image_width = config['image']['width']  # From render function of Cartpole env
		self.env_name = environment
		if self.env_name == 'cartpole':
			if gen_mode == 'random' or gen_mode == 'incremental':
				self.env = DenseRewardWrapperCartpole(ImGenDenseRewardWrapperCartpole(gym.make("CartPole-v1")))    # This wrapper allows saving the image and the associated reward
			elif gen_mode == 'random_weird' or gen_mode == 'incremental_weird':
				self.env = WeirdDenseRewardWrapperCartpole(ImGenDenseRewardWrapperCartpole(gym.make("CartPole-v1")))    # This wrapper allows saving the image and the associated reward
		elif self.env_name == 'duckietown':
			if gen_mode == 'random':
				self.env = DTDistAngleObsWrapper(DTConstantVelWrapper(DTLaneFollowingRewardWrapper(DTDroneImageGenerator(DuckietownEnv(accept_start_angle_deg = 90), drone_sim = 'random', drone_angle_follow = True))))
			elif gen_mode == 'random_0':
				self.env = DTDistAngleObsWrapper(DTConstantVelWrapper(DTLaneFollowingRewardWrapper(DTDroneImageGenerator(DuckietownEnv(accept_start_angle_deg = 90), drone_sim = 'random_0', drone_angle_follow = True))))
			elif gen_mode == 'random_1':
				self.env = DTDistAngleObsWrapper(DTConstantVelWrapper(DTLaneFollowingRewardWrapper(DTDroneImageGenerator(DuckietownEnv(accept_start_angle_deg = 90), drone_sim = 'random_1', drone_angle_follow = True))))				
			elif gen_mode == 'random_2':
				self.env = DTDistAngleObsWrapper(DTConstantVelWrapper(DTLaneFollowingRewardWrapper(DTDroneImageGenerator(DuckietownEnv(accept_start_angle_deg = 90), drone_sim = 'random_2', drone_angle_follow = True))))								
			elif gen_mode == 'random_3':
				self.env = DTDistAngleObsWrapper(DTConstantVelWrapper(DTLaneFollowingRewardWrapper(DTDroneImageGenerator(DuckietownEnv(accept_start_angle_deg = 90), drone_sim = 'random_3', drone_angle_follow = True))))								
			elif gen_mode == 'random_straight':
				self.env = DTDistAngleObsWrapper(DTConstantVelWrapper(DTLaneFollowingRewardWrapper(DTDroneImageGenerator(DuckietownEnv(accept_start_angle_deg = 90, map_name = 'straight_road_mid'), drone_sim = 'random', drone_angle_follow = True))))
			elif gen_mode == 'random_0_straight':
				self.env = DTDistAngleObsWrapper(DTConstantVelWrapper(DTLaneFollowingRewardWrapper(DTDroneImageGenerator(DuckietownEnv(accept_start_angle_deg = 90, map_name = 'straight_road_mid'), drone_sim = 'random_0', drone_angle_follow = True))))				
			elif gen_mode == 'random_1_straight':
				self.env = DTDistAngleObsWrapper(DTConstantVelWrapper(DTLaneFollowingRewardWrapper(DTDroneImageGenerator(DuckietownEnv(accept_start_angle_deg = 90, map_name = 'straight_road_mid'), drone_sim = 'random_1', drone_angle_follow = True))))				
			elif gen_mode == 'random_2_straight':
				self.env = DTDistAngleObsWrapper(DTConstantVelWrapper(DTLaneFollowingRewardWrapper(DTDroneImageGenerator(DuckietownEnv(accept_start_angle_deg = 90, map_name = 'straight_road_mid'), drone_sim = 'random_2', drone_angle_follow = True))))				
			elif gen_mode == 'random_3_straight':
				self.env = DTDistAngleObsWrapper(DTConstantVelWrapper(DTLaneFollowingRewardWrapper(DTDroneImageGenerator(DuckietownEnv(accept_start_angle_deg = 90, map_name = 'straight_road_mid'), drone_sim = 'random_3', drone_angle_follow = True))))				
			else:
				raise ValueError('{} as a image generation mode does not exist for Duckietown'.format(gen_mode))				
		self.env.reset()   # Reset needed for the environment to start
		self.env.render(mode='human')
		print("Initialized image generator")

	def generate_pair(self, state = None):
		# Generate an image and its associated reward given a state.
		# If state = None, it will be randomly generated according to self.env.reset()
		if state:
			self.env.reset()
			self.env.set_state(state)
		else:
			state = self.env.reset()
			if self.env_name == 'cartpole':  			# State from cartpole env is [x, xdot, theta, thetadot]
				state = [state[0], state[2]]
		reward = self.env.reward(state)
		image = self.env.render(mode='rgb_array')
		if self.env_name == 'duckietown':
			drone_params = self.env.get_drone_params()
			state += drone_params
		else:
			drone_params = [0, 0, 0, 0]
		self.env.close()
		return reward, image, state

	def add_pair_to_CSV(self, csv_path, pair):
		# Save a pair of strings (here, image path and its reward/state)
		with open(csv_path, "a") as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(pair)

	def one_image(self, set_name, csv_path, image_path, state = None):
		# Generate and save one image. If state is None, a random one is drawn.
		reward, image, state = self.get_image(state, image_path)

		self.env.close()
		if self.env_name == 'duckietown':
			self.add_pair_to_CSV(csv_path, (state[0], state[1], str(reward), image_path, state[5])) #state[5] is drone angle
		elif self.env_name == 'cartpole':
			self.add_pair_to_CSV(csv_path, (state[0], state[1], str(reward), image_path))


	def get_image(self, des_state, image_path):
		reward, image, state = self.generate_pair(des_state)
		cv2.imwrite(image_path, image)

		# Double check that the image was correctly saved, otherwise, do it again
		try:
			image_2 = cv2.imread(image_path)
		except:
			print("Resaving image {}".format(image_path))
			reward, image, state = self.get_image(self.env_name, des_state, image_path)
		return reward, image, state

	def create_images(self, number_images, set_name, computer, config, gen_mode):
		# Generates images from the OpenAI Gym Cartpole environment specified in self.init

		root_path = config['paths'][computer]['save_images']
		images_path = os.path.join(root_path, self.env_name, gen_mode, set_name)
		utils.makeDir(images_path)

		# Init csv file
		csv_path = os.path.join(images_path, 'data_pairs.csv')
		with open(csv_path, "w") as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(['X', 'Theta','Reward', 'Image', 'Drone angle'])

		# Create images
		if gen_mode in ['random', 'random_weird', 'random_0', 'random_1', 'random_2', 'random_3', 'random_straight', 'random_0_straight', 'random_1_straight', 'random_2_straight', 'random_3_straight']:
			for i in range(number_images):
				if i%100 == 99:
					print(i)
				image_name = str(i)
				image_path = os.path.join(images_path, image_name+'.png')
				self.one_image(set_name, csv_path, image_path)
			self.env.close()

		elif gen_mode == 'incremental' or gen_mode == 'incremental_weird':
			if self.env_name == 'duckietown':
				raise ValueError("It is not possible to run an incremental image generation on the Duckietown environment.")
			if set_name == 'train':
				pass
			square_side = int(np.floor(np.sqrt(number_images)))
			x_range = self.env.x_range
			theta_range = self.env.theta_range
			i = 0
			print('x_range[0]: {}, x_range[1]: ­{}, num = square_side: ­{}'.format(x_range[0], x_range[1],square_side))
			for x in np.linspace(x_range[0], x_range[1], num = square_side):
				for theta in np.linspace(theta_range[0], theta_range[1], num = square_side):
					state = (x, theta)
					image_name = str(i)
					image_path = os.path.join(images_path, image_name+'.png')
					self.one_image(set_name, csv_path, image_path, state)
					i += 1
			self.env.close()
		
		elif gen_mode == 'training':
			raise NotImplementedError("Generating images during training has not been implemented yet.")


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='This script generates images from a given environment.')
	parser.add_argument('-c','--computer', help='Computer on which the script is launched.',required=True)
	parser.add_argument('-e','--environment',help='Environment from which the images are created', required=True)
	parser.add_argument('-g', '--generation_mode', help='Mode of generation', required = True)
	parser.add_argument('-t', '--test_only', help='If only test', action = 'store_true')

	args = parser.parse_args()
	
	computer = utils.getComputer(args.computer)
	environment = utils.getImEnv(args.environment)
	gen_mode = utils.getGenMode(args.generation_mode)

	# Loading config
	config = utils.loadYAMLFromFile('config_' + environment + '.yaml')
	img_gen = image_generator(config, environment, gen_mode)
	nb_train_images = config['exp']['nb_train_im']
	nb_test_images = config['exp']['nb_test_im']

	# Generating images
	if not args.test_only:
		img_gen.create_images(nb_train_images, 'train', computer, config, gen_mode)
	img_gen.create_images(nb_test_images, 'test', computer, config, gen_mode)