import gym
import numpy as np 


from gym import spaces
import numpy as np 
from train_model import Net, Rescale, NormalizeImage, UnnormalizeLabel
import torch
import torch.nn as nn
import torchvision.models as models

from utils import *

##### CARTPOLE WRAPPERS ######
# Here are several wrappers on the OpenAI Gym Cartpole environment.

## Action wrappers ##
class ContinuousActionWrapperCartpole(gym.Wrapper):

	def __init__(self, env):
		gym.Wrapper.__init__(self, env)
		self.action_space = spaces.Box(low=-1, high=1, shape=(1,)) 
		self._max_episode_steps = 1000 

	def reset(self):
		self.steps_beyond_done = None
		self.env.reset()
		print('---')
		print(self.env.state)
		new_state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
		self.env.state = new_state
		print(self.env.state)
		print('---')
		return np.transpose(np.array(self.env.state))

	def step(self, action):
		assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
		action = action[0]
		state = self.env.state
		x, x_dot, theta, theta_dot = state
		force = self.force_mag*action
		costheta = math.cos(theta)
		sintheta = math.sin(theta)
		temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
		thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
		xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
		if self.kinematics_integrator == 'euler':
			x  = x + self.tau * x_dot
			x_dot = x_dot + self.tau * xacc
			theta = theta + self.tau * theta_dot
			theta_dot = theta_dot + self.tau * thetaacc
		else: # semi-implicit euler
			x_dot = x_dot + self.tau * xacc
			x  = x + self.tau * x_dot
			theta_dot = theta_dot + self.tau * thetaacc
			theta = theta + self.tau * theta_dot
		self.env.state = (x, x_dot, theta, theta_dot)
		done =  x < -self.x_threshold \
				or x > self.x_threshold \
				or theta < -self.theta_threshold_radians \
				or theta > self.theta_threshold_radians
		done = bool(done)

		if not done:
			reward = 1.0
		elif self.steps_beyond_done is None:
			# Pole just fell!
			self.steps_beyond_done = 0
			reward = 1.0
		else:
			if self.steps_beyond_done == 0:
				logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
			self.steps_beyond_done += 1
			reward = 0.0

		return np.array(self.env.state), reward, done, {}

	def render(self, mode='human'):
		screen_width = 600
		screen_height = 400

		world_width = self.x_threshold*2
		scale = screen_width/world_width
		carty = 100 # TOP OF CART
		polewidth = 10.0
		polelen = scale * (2 * self.length)
		cartwidth = 50.0
		cartheight = 30.0

		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width, screen_height)
			l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
			axleoffset =cartheight/4.0
			cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			self.carttrans = rendering.Transform()
			cart.add_attr(self.carttrans)
			self.viewer.add_geom(cart)
			l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
			pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			pole.set_color(.8,.6,.4)
			self.poletrans = rendering.Transform(translation=(0, axleoffset))
			pole.add_attr(self.poletrans)
			pole.add_attr(self.carttrans)
			self.viewer.add_geom(pole)
			self.axle = rendering.make_circle(polewidth/2)
			self.axle.add_attr(self.poletrans)
			self.axle.add_attr(self.carttrans)
			self.axle.set_color(.5,.5,.8)
			self.viewer.add_geom(self.axle)
			self.track = rendering.Line((0,carty), (screen_width,carty))
			self.track.set_color(0,0,0)
			self.viewer.add_geom(self.track)

			self._pole_geom = pole

		if self.state is None: return None

		# Edit the pole polygon vertex
		pole = self._pole_geom
		l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
		pole.v = [(l,b), (l,t), (r,t), (r,b)]

		x = self.state
		cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
		self.carttrans.set_translation(cartx, carty)
		self.poletrans.set_rotation(-x[2])

		return self.viewer.render(return_rgb_array = mode=='rgb_array')

## Reward wrappers ##
# DenseRewardWrapperCartpole generates the reward based directly on the state. It can be seen as Ground Truth reward environment. 
# ImGenDenseRewardWrapperCartpole does the same than DenseRewardWrapperCartpole but is made to return a randomly generated state and return the associated image and reward.
# R_CNNDenseRewardWrapperCartpole generates a reward by rendering an image and passing it through a previously trained model.


def state2reward(x, theta):
	reward = 1 - (3*np.abs(theta/.5)/4) - (np.abs(x/2.5)/2)**2
	return reward


class DenseRewardWrapperCartpole(gym.Wrapper):
	# Generating dense reward based on the state (different from original problem)

	def step(self, action):
		observation, reward, done, info = self.env.step(action)
		return observation, self.reward(observation), done, info

	def reward(self, obs):
		x = self.state[0]
		theta = self.state[2]
		return state2reward(x, theta)

class NoisyDenseRewardWrapperCartpole(gym.Wrapper):
	# Generating dense reward based on the state (different from original problem)
	def __init__(self, env, std_dev = 0.2):
		gym.Wrapper.__init__(self, env)
		self.std_dev = std_dev
		env = DenseRewardWrapperCartpole(ContinuousActionWrapperCartpole(gym.make('CartPole-v0')))

	def step(self, action):
		observation, reward, done, info = self.env.step(action)
		noisy_reward = self.reward(observation)
		return observation, self.reward(observation), done, info

	def reward(self, obs):
		reward = state2reward(self.state[0], self.state[2]) + np.random.normal(loc=0.0, scale=self.std_dev)
		return reward

class ImGenDenseRewardWrapperCartpole(DenseRewardWrapperCartpole):
	# Generating reward based on the state as its parent class, plus has a function to generate a random state, and allows to render and save an image

	def __init__(self, env):
		super().__init__(env)
		self.x_range = [-2.5, 2.5]
		self.theta_range = [-0.5, 0.5]

	def reset(self):
		self.state = self.generate_state()
		self.steps_beyond_done = None
		return np.array(self.state)

	def set_state(self, state):
		if len(state) == 2:
			x = state[0]
			xdot = 0
			theta = state[1]
			thetadot = 0
		elif len(state) == 4:
			x = state[0]
			xdot = state[1]
			theta = state[2]
			thetadot = state[3]
		self.state = (x, xdot, theta, thetadot)

	def generate_state(self):
		x = np.random.uniform(low=self.x_range[0], high=self.x_range[1])
		xdot = 0			# Does not matter for reward
		theta = np.random.uniform(low=self.theta_range[0], high=self.theta_range[1])
		thetadot = 0		# Does not matter for reward
		return (x, xdot, theta, thetadot)

	def render(self, mode='human'):
		screen_width = 600
		screen_height = 400

		world_width = self.x_threshold*2
		scale = screen_width/world_width
		carty = 100 # TOP OF CART
		polewidth = 10.0
		polelen = scale * (2 * self.length)
		cartwidth = 50.0
		cartheight = 30.0

		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width, screen_height)
			l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
			axleoffset =cartheight/4.0
			cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			self.carttrans = rendering.Transform()
			cart.add_attr(self.carttrans)
			self.viewer.add_geom(cart)
			l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
			pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			pole.set_color(.8,.6,.4)
			self.poletrans = rendering.Transform(translation=(0, axleoffset))
			pole.add_attr(self.poletrans)
			pole.add_attr(self.carttrans)
			self.viewer.add_geom(pole)
			self.axle = rendering.make_circle(polewidth/2)
			self.axle.add_attr(self.poletrans)
			self.axle.add_attr(self.carttrans)
			self.axle.set_color(.5,.5,.8)
			self.viewer.add_geom(self.axle)
			self.track = rendering.Line((0,carty), (screen_width,carty))
			self.track.set_color(0,0,0)
			self.viewer.add_geom(self.track)

			self._pole_geom = pole

		if self.state is None: return None

		# Edit the pole polygon vertex
		pole = self._pole_geom
		l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
		pole.v = [(l,b), (l,t), (r,t), (r,b)]

		x = self.state
		cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
		self.carttrans.set_translation(cartx, carty)
		self.poletrans.set_rotation(-x[2])

		return self.viewer.render(return_rgb_array = mode=='rgb_array')


class R_CNNDenseRewardWrapperCartpole(gym.Wrapper):
	# Generates reward through rendering the environment, and passing the image through a loaded, pre-trained model 

	def __init__(self, env, cnn_folder_path, cnn_type):
		super().__init__(env)
		params_file = os.path.join(cnn_folder_path, 'cnn_params.yaml')
		self.model_params = loadYAMLFromFile(params_file)
		self.model = self.loadModel(cnn_folder_path, cnn_type)
		self.config = loadYAMLFromFile('config_cartpole.yaml')

	def loadModel(self, cnn_folder_path, cnn_type="resnet18"):
		model = self.initialize_net(cnn_type)
		model_path = os.path.join(cnn_folder_path, 'latest_model.pth')
		model.load_state_dict(torch.load(model_path))
		print('Loaded model ' + model_path)
		model.eval()
		return model

	def initialize_net(self, cnn_type):
		if cnn_type == 'resnet18':
			model = models.resnet18(pretrained=False, num_classes=1)
		else:
			raise NotImplementedError("CP_R_CNN_State_Wrapper not implemented for another model type than resnet18.")
		return model

	def step(self, action):
		observation, reward, done, info = self.env.step(action)
		return observation, self.reward(observation), done, info

	def reward(self, state):
		image = self.render(mode='rgb_array')
		image = self.transformImage(image)
		reward = self.model.forward(image.float())
		reward = reward.detach().numpy()[0][0]
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

	def render(self, mode='human'):
		screen_width = 600
		screen_height = 400

		world_width = self.x_threshold*2
		scale = screen_width/world_width
		carty = 100 # TOP OF CART
		polewidth = 10.0
		polelen = scale * (2 * self.length)
		cartwidth = 50.0
		cartheight = 30.0

		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width, screen_height)
			l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
			axleoffset =cartheight/4.0
			cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			self.carttrans = rendering.Transform()
			cart.add_attr(self.carttrans)
			self.viewer.add_geom(cart)
			l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
			pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			pole.set_color(.8,.6,.4)
			self.poletrans = rendering.Transform(translation=(0, axleoffset))
			pole.add_attr(self.poletrans)
			pole.add_attr(self.carttrans)
			self.viewer.add_geom(pole)
			self.axle = rendering.make_circle(polewidth/2)
			self.axle.add_attr(self.poletrans)
			self.axle.add_attr(self.carttrans)
			self.axle.set_color(.5,.5,.8)
			self.viewer.add_geom(self.axle)
			self.track = rendering.Line((0,carty), (screen_width,carty))
			self.track.set_color(0,0,0)
			self.viewer.add_geom(self.track)

			self._pole_geom = pole

		if self.state is None: return None

		# Edit the pole polygon vertex
		pole = self._pole_geom
		l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
		pole.v = [(l,b), (l,t), (r,t), (r,b)]

		x = self.state
		cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
		self.carttrans.set_translation(cartx, carty)
		self.poletrans.set_rotation(-x[2])

		return self.viewer.render(return_rgb_array = mode=='rgb_array')



class S_CNNDenseRewardWrapperCartpole(R_CNNDenseRewardWrapperCartpole):
	# Generates reward through 1) rendering the environment, 2) passing the image through a loaded, pre-trained model which outputs the state, 3) convert state in reward

	def __init__(self, env, model_path='State_trained_model.pth'):
		super().__init__(env, model_path = model_path)

	def initialize_net(self, cnn_type):
		if cnn_type == 'resnet18':
			model = models.resnet18(pretrained=False, num_classes=2)
		else:
			raise NotImplementedError("CP_S_CNN_State_Wrapper not implemented for another model type than resnet18.")
		return model

	def reward(self, state):
		image = self.render(mode='rgb_array')
		image = self.transformImage(image)
		state = self.model.forward(image.float())
		state = state.detach().numpy()[0]

		x, theta = UnnormalizeLabel('cartpole', 'State', state)
		return state2reward(x, theta)
