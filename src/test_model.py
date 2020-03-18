from __future__ import print_function, division
import os
import torch
import cv2
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

import utils as ut
import argparse
from train_model import Net, ImageLabelDataset, Rescale, ToTensor, UnnormalizeLabel
import cartpole_mod_env as cp_modenv
import duckietown_mod_env as dt_modenv
import resnet


#################################################
# 					Tester 						#
#################################################	
class Tester(object):

	def __init__(self, config, computer, test_incremental): #trained_model_prefix, label_style):
		# GPU Params
		use_gpu = torch.cuda.is_available()
		self.device = torch.device("cuda:0" if use_gpu else "cpu")

		# Paths and configuration parameters
		self.environment = config['exp']['env']
		self.gen_mode = config['exp']['gen_mode']
		test_set_gen_mode = 'incremental' if test_incremental else self.gen_mode
		self.label_style = config['exp']['label_type']
		model_name = config['exp']['model_name']
		self.testing_set_path = os.path.join(config['paths'][computer]['save_images'], self.environment, test_set_gen_mode, 'test')
		self.root_path = config['paths'][computer]['save_cnn']
		self.root_label_path = os.path.join(self.root_path, self.environment, self.label_style, self.gen_mode, model_name)
		self.rescale_size = tuple(config['cnn']['rescale_size'])

		self.model_type = config['cnn']['model']

		if test_incremental:
			self.test_results_path = os.path.join(self.root_label_path, 'test_results', 'incremental')
		else:
			self.test_results_path = os.path.join(self.root_label_path, 'test_results', 'test_set')
			print(self.test_results_path)
		ut.makeDir(self.test_results_path)
		print(self.test_results_path)
		self.model_path = os.path.join(self.root_label_path, 'latest_model.pth')
		self.cnn_params_path = os.path.join(self.root_label_path, 'cnn_params.yaml')

		# Loading net
		self.net = self.load_model()
		self.criterion = nn.MSELoss()
		# Loading data
		self.test_set = self.load_data()

	def load_data(self):
		# Load data from the data set given by ` prefix + label_style + "data_pairs.csv" ` produced beforehand by generate_images.py
		# And apply transforms of rescaling and passing to pytorch tensor formatcomposed_transform = transforms.Compose([Rescale((32, 32)), ToTensor()])
		composed_transform = transforms.Compose([Rescale(self.rescale_size), ToTensor()])
		labeled_dataset = ImageLabelDataset(os.path.join(self.testing_set_path, 'data_pairs.csv'), self.cnn_params_path, self.label_style, self.environment, transform = composed_transform)
		data_loader = DataLoader(labeled_dataset, batch_size = 1, shuffle=False, num_workers=0)
		print('Data loaded')
		return data_loader
	
	def load_model(self):
		# Load the model as defined by ` self.trained_model_prefix + "latest_model.pth" ` and already trained by train_model.py
		nb_outputs = ut.nbOutputs(self.label_style, self.environment)
		
		if self.model_type == 'small':
			net = Net(input_size = self.rescale_size, nb_outputs = nb_outputs)
		elif self.model_type == 'resnet18':
			net = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], num_classes=nb_outputs)
		elif self.model_type == 'resnet34':
			net = resnet.ResNet(resnet.BasicBlock, [3, 4, 6, 3], num_classes=nb_outputs)
		elif self.model_type == 'resnet50':
			net = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=nb_outputs)
		elif self.model_type == 'resnet101':
			net = resnet.ResNet(resnet.Bottleneck, [3, 4, 23, 3], num_classes=nb_outputs)
		elif self.model_type == 'resnet152':
			net = resnet.ResNet(resnet.Bottleneck, [3, 8, 36, 3], num_classes=nb_outputs)

		net.load_state_dict(torch.load(self.model_path))
		print('Loaded model')
		net.eval()
		net = net.to(self.device)
		return net

	def test_one_image(self,image, label):
		# Returns criterion loss and actual error when applying the model to an image and comparing output to label
		outputs = self.net.forward(image.float())
		print('Outputs: {}'.format(outputs.float().detach().cpu().numpy()[0]))
		#print('Labels: {}'.format(label.float().detach().cpu().numpy()[0]))
		loss = self.criterion(outputs.float(), label.float())
		#print('Loss: {}'.format(loss.item()))
		if self.label_style == 'Reward':
			state_error = 0
			estim_reward = UnnormalizeLabel(self.environment, self.label_style, outputs.float().detach().cpu().numpy()[0][0])
			real_reward = UnnormalizeLabel(self.environment, self.label_style, label.float().detach().cpu().numpy()[0][0])
			rew_error = estim_reward - real_reward # (outputs.float() - label.float()).detach().cpu().numpy()[0][0]
			pred_state = [0, 0]
		elif self.label_style == 'State':
			state_estim = UnnormalizeLabel(self.environment, self.label_style, outputs.float().detach().cpu().numpy()[0])
			state_gt = UnnormalizeLabel(self.environment, self.label_style, label.float().detach().cpu().numpy()[0])
			hat_x = state_estim[0]
			hat_theta = state_estim[1]
			x = state_gt[0]
			pred_state = [hat_x, hat_theta]
			theta = state_gt[1]
			state_error = [hat_x - x, ut.angleDiff(hat_theta, theta)]
			if self.environment == 'cartpole':
				real_reward = cp_modenv.state2reward(x, theta)
				estim_reward = cp_modenv.state2reward(hat_x, hat_theta)
			elif self.environment == 'duckietown':
				real_reward = dt_modenv.state2reward(x, theta)
				estim_reward = dt_modenv.state2reward(hat_x, hat_theta)
			rew_error = estim_reward - real_reward # (outputs.float() - label.float()).detach().cpu().numpy()[0][0]
		elif self.label_style == 'Angle':
			angle_estim = UnnormalizeLabel(self.environment, self.label_style, outputs.float().detach().cpu().numpy()[0])
			angle_gt = UnnormalizeLabel(self.environment, self.label_style, label.float().detach().cpu().numpy()[0])
			hat_theta = angle_estim[0]
			pred_state = [0, hat_theta]
			theta = angle_gt[0]
			state_error = [0, ut.angleDiff(hat_theta, theta)]
			rew_error = 0
			real_reward = 0
			estim_reward = 0
		elif self.label_style == 'Angle_droneAngle':
			angles_estim = UnnormalizeLabel(self.environment, self.label_style, outputs.float().detach().cpu().numpy()[0])
			angles_gt = UnnormalizeLabel(self.environment, self.label_style, label.float().detach().cpu().numpy()[0])
			hat_theta = angles_estim[0]
			theta = angles_gt[0]
			hat_drone_angle = angles_estim[1]
			drone_angle_gt = angles_gt[1]
			pred_state = [0, hat_theta, hat_drone_angle]
			state_error = [0, ut.angleDiff(hat_theta, theta), ut.angleDiff(hat_drone_angle, drone_angle_gt)]
			rew_error = 0
			real_reward = 0
			estim_reward = 0			
		elif self.label_style == 'Distance':
			dist_estim = UnnormalizeLabel(self.environment, self.label_style, outputs.float().detach().cpu().numpy()[0])
			dist_gt = UnnormalizeLabel(self.environment, self.label_style, label.float().detach().cpu().numpy()[0])
			hat_x = dist_estim[0]
			pred_state = [hat_x, 0]
			x = dist_gt[0]
			state_error = [hat_x - x, 0]
			rew_error = 0
			real_reward = 0
			estim_reward = 0
		elif self.label_style == 'State_droneAngle':
			state_estim = UnnormalizeLabel(self.environment, self.label_style, outputs.float().detach().cpu().numpy()[0])
			state_gt = UnnormalizeLabel(self.environment, self.label_style, label.float().detach().cpu().numpy()[0])	

			hat_x = state_estim[0]
			hat_theta = state_estim[1]
			hat_drone_angle = state_estim[2]
			pred_state = [hat_x, hat_theta, hat_drone_angle]

			x = state_gt[0]
			theta = state_gt[1]
			drone_angle_gt = angles_gt[1]

			state_error = [hat_x - x, ut.angleDiff(hat_theta, theta), ut.angleDiff(hat_drone_angle, drone_angle_gt)]

			real_reward = dt_modenv.state2reward(x, theta)
			estim_reward = dt_modenv.state2reward(hat_x, hat_theta)
			rew_error = estim_reward - real_reward # (outputs.float() - label.float()).detach().cpu().numpy()[0][0]

		elif self.label_style == 'droneAngle':
			drone_angle_estim = UnnormalizeLabel(self.environment, self.label_style, outputs.float().detach().cpu().numpy()[0])
			drone_angle_label = UnnormalizeLabel(self.environment, self.label_style, label.float().detach().cpu().numpy()[0])
			hat_drone_angle = drone_angle_estim[0]
			pred_state = [0, 0, hat_drone_angle]
			drone_angle_gt = drone_angle_label[0]
			state_error = [0, 0, ut.angleDiff(hat_drone_angle, drone_angle_gt)]
			rew_error = 0
			real_reward = 0
			estim_reward = 0

		return loss.item(), pred_state, state_error, rew_error, real_reward, estim_reward


	def test_all_images(self):
		# Tests all the images and outputs relevant statistics
		losses = []
		pred_states = []
		state_errors = []
		rew_errors = []
		xs = []
		thetas = []
		drone_angles = []
		real_rewards = []
		estim_rewards = []


		for i, data in enumerate(self.test_set, 0):
			#print(data)
			if i%100 == 0:
				print(i)
			#if i >=5000:
				#break
			image = data['image']
			image = image.to(self.device)
			label = data['label']
			x = data['x'].detach().numpy()[0]
			theta = data['theta'].detach().numpy()[0]
			drone_angle = data['drone_angle'].detach().numpy()[0]
			label = label.to(self.device)
			loss, pred_state, state_error, rew_error, real_reward, estim_reward = self.test_one_image(image, label)	
			losses.append(loss)
			pred_states.append(pred_state)
			state_errors.append(state_error)
			rew_errors.append(rew_error)
			xs.append(x)
			thetas.append(theta)
			drone_angles.append(drone_angle)
			real_rewards.append(real_reward)
			estim_rewards.append(estim_reward)

		avg_loss = np.mean(losses)
		avg_state_error = np.mean(state_errors, 0)
		avg_abs_state_error = np.mean(np.abs(state_errors), 0)
		avg_rew_error = np.mean(rew_errors)
		avg_abs_rew_error = np.mean(np.abs(rew_errors))
		mean_square_error_reward = np.sqrt(np.mean(np.square(rew_errors)))
		rew_variance = np.var(rew_errors)

		print('Average loss on test set: ' + str(avg_loss))
		print('Average absolute error on reward: ' + str(avg_abs_rew_error))
		print('Average error on reward: ' + str(avg_rew_error))
		print('Variance on reward: ' + str(rew_variance))
		print('Mean square error on reward: ' + str(mean_square_error_reward))
		print('Average absolute error on state (0 if non applicable): ' + str(avg_abs_state_error))
		print('Average error on state (0 if non applicable): ' + str(avg_state_error))


		### Saving data
		np.save(self.test_results_path + "/losses", losses)
		np.save(self.test_results_path + "/state_errors", state_errors)
		np.save(self.test_results_path + "/rew_errors", rew_errors)
		np.save(self.test_results_path + "/xs", xs)
		np.save(self.test_results_path + "/thetas", thetas)
		np.save(self.test_results_path + "/pred_states", pred_states)

		### Making and saving histograms
		plt.figure()
		n, _, _ = plt.hist(losses, bins=50, density = True, range = (0, 0.15))
		plt.title("Loss histogram for {}-predicting model".format(self.label_style))
		plt.savefig(self.test_results_path + "/loss_histogram.png", bbox_inches="tight")

		plt.figure()
		n, _, _ = plt.hist(rew_errors, bins=100, density =True, color="blue", ec="black", range = (-0.5, 0.5))
		plt.title("Reward error histogram for {}-predicting model".format(self.label_style))
		plt.savefig(self.test_results_path + "/rew_err_histogram.png", bbox_inches="tight")

		plt.figure()
		fig, ax = plt.subplots()
		normalize = colors.DivergingNorm(0)

		plt.scatter(xs, thetas, c=rew_errors, norm=normalize, cmap ='seismic')
		ax.set_xlabel('Position X')
		ax.set_ylabel('Angle theta')
		cbar = plt.colorbar()
		cbar.set_label('Reward error')
		plt.savefig(self.test_results_path + "/rew_error_3Dplat.png", bbox_inches="tight")

		plt.figure()
		fig, ax = plt.subplots()
		plt.scatter(xs, thetas, c=np.abs(rew_errors), cmap ='Blues')
		ax.set_xlabel('Position X')
		ax.set_ylabel('Angle theta')
		cbar = plt.colorbar()
		cbar.set_label('Absolute reward error')
		plt.savefig(self.test_results_path + "/abs_rew_error_3Dplat.png", bbox_inches="tight")

		plt.figure()
		fig, ax = plt.subplots()
		plt.scatter(xs, thetas, c=estim_rewards, cmap ='Blues')
		ax.set_xlabel('Position X')
		ax.set_ylabel('Angle theta')
		cbar = plt.colorbar()
		cbar.set_label('Estimated reward')
		plt.savefig(self.test_results_path + "/estim_rew_3Dplat.png", bbox_inches="tight")

		plt.figure()
		fig, ax = plt.subplots()
		plt.scatter(xs, thetas, c=real_rewards, cmap ='Blues')
		ax.set_xlabel('Position X')
		ax.set_ylabel('Angle theta')
		cbar = plt.colorbar()
		cbar.set_label('Ground truth reward')
		plt.savefig(self.test_results_path + "/real_rew_3Dplat.png", bbox_inches="tight")


		if self.label_style == "State":
			state_errors_np = np.array(state_errors)
			x_errors = state_errors_np[:,0]
			theta_errors = state_errors_np[:,1]

			plt.figure()
			n, _, _ = plt.hist(theta_errors, bins=100, density = True, color="skyblue", ec="black")
			plt.title("Angle error histogram for {}-predicting model".format(self.label_style))
			plt.savefig(self.test_results_path + "/angle_err_histogram.png", bbox_inches="tight")

			plt.figure()
			n, _, _ = plt.hist(x_errors, bins=100, density = True, color="skyblue", ec="black")
			plt.title("Position error histogram for {}-predicting model".format(self.label_style))
			plt.savefig(self.test_results_path + "/pos_err_histogram.png", bbox_inches="tight")

			plt.figure()
			fig, ax = plt.subplots()
			plt.scatter(xs, thetas, c=theta_errors, norm=normalize, cmap ='seismic')
			ax.set_xlabel('Position X')
			ax.set_ylabel('Angle theta')
			cbar = plt.colorbar()
			cbar.set_label('Theta error')
			plt.savefig(self.test_results_path + "/theta_error_3Dplat.png", bbox_inches="tight")

			plt.figure()
			fig, ax = plt.subplots()
			plt.scatter(xs, thetas, c=x_errors, norm=normalize, cmap ='seismic')
			ax.set_xlabel('Position X')
			ax.set_ylabel('Angle theta')
			cbar = plt.colorbar()
			cbar.set_label('X error')
			plt.savefig(self.test_results_path + "/x_errors_3Dplat.png", bbox_inches="tight")

		elif self.label_style == "State_droneAngle":
			state_errors_np = np.array(state_errors)
			x_errors = state_errors_np[:,0]
			theta_errors = state_errors_np[:,1]
			drone_angle_errors = state_errors_np[:,2]

			plt.figure()
			n, _, _ = plt.hist(theta_errors, bins=100, density = True, color="skyblue", ec="black")
			plt.title("Angle error histogram for {}-predicting model".format(self.label_style))
			plt.savefig(self.test_results_path + "/angle_err_histogram.png", bbox_inches="tight")

			plt.figure()
			n, _, _ = plt.hist(x_errors, bins=100, density = True, color="skyblue", ec="black")
			plt.title("Position error histogram for {}-predicting model".format(self.label_style))
			plt.savefig(self.test_results_path + "/pos_err_histogram.png", bbox_inches="tight")

			plt.figure()
			n, _, _ = plt.hist(drone_angle_errors, bins=100, density = True, color="skyblue", ec="black")
			plt.title("Drone angle error histogram for {}-predicting model".format(self.label_style))
			plt.savefig(self.test_results_path + "/drone_angle_err_histogram.png", bbox_inches="tight")

			plt.figure()
			fig, ax = plt.subplots()
			plt.scatter(xs, thetas, c=theta_errors, norm=normalize, cmap ='seismic')
			ax.set_xlabel('Position X')
			ax.set_ylabel('Angle theta')
			cbar = plt.colorbar()
			cbar.set_label('Theta error')
			plt.savefig(self.test_results_path + "/theta_error_3Dplat.png", bbox_inches="tight")

			plt.figure()
			fig, ax = plt.subplots()
			plt.scatter(xs, thetas, c=x_errors, norm=normalize, cmap ='seismic')
			ax.set_xlabel('Position X')
			ax.set_ylabel('Angle theta')
			cbar = plt.colorbar()
			cbar.set_label('X error')
			plt.savefig(self.test_results_path + "/x_errors_3Dplat.png", bbox_inches="tight")

			plt.figure()
			fig, ax = plt.subplots()
			plt.scatter(xs, thetas, c=drone_angle_errors, norm=normalize, cmap ='seismic')
			ax.set_xlabel('Position X')
			ax.set_ylabel('Angle theta')
			cbar = plt.colorbar()
			cbar.set_label('Drone angle error')
			plt.savefig(self.test_results_path + "/drone_angle_errors_3Dplat.png", bbox_inches="tight")

		elif self.label_style == 'Distance':
			state_errors_np = np.array(state_errors)
			x_errors = state_errors_np[:,0]

			plt.figure()
			n, _, _ = plt.hist(x_errors, bins=100, density = True, color="skyblue", ec="black")
			plt.title("Position error histogram for {}-predicting model".format(self.label_style))
			plt.savefig(self.test_results_path + "/pos_err_histogram.png", bbox_inches="tight")

			plt.figure()
			fig, ax = plt.subplots()
			plt.scatter(xs, thetas, c=x_errors, norm=normalize, cmap ='seismic')
			ax.set_xlabel('Position X')
			ax.set_ylabel('Angle theta')
			cbar = plt.colorbar()
			cbar.set_label('X error')
			plt.savefig(self.test_results_path + "/x_errors_3Dplat.png", bbox_inches="tight")

		elif self.label_style == 'Angle':
			state_errors_np = np.array(state_errors)
			theta_errors = state_errors_np[:,1]			

			plt.figure()
			n, _, _ = plt.hist(theta_errors, bins=100, density = True, color="skyblue", ec="black")
			plt.title("Angle error histogram for {}-predicting model".format(self.label_style))
			plt.savefig(self.test_results_path + "/angle_err_histogram.png", bbox_inches="tight")

			plt.figure()
			fig, ax = plt.subplots()
			plt.scatter(xs, thetas, c=theta_errors, norm=normalize, cmap ='seismic')
			ax.set_xlabel('Position X')
			ax.set_ylabel('Angle theta')
			cbar = plt.colorbar()
			cbar.set_label('Theta error')
			plt.savefig(self.test_results_path + "/theta_error_3Dplat.png", bbox_inches="tight")

		elif self.label_style == "Angle_droneAngle":
			state_errors_np = np.array(state_errors)
			theta_errors = state_errors_np[:,1]
			drone_angle_errors = state_errors_np[:,2]

			plt.figure()
			n, _, _ = plt.hist(theta_errors, bins=100, density = True, color="skyblue", ec="black")
			plt.title("Angle error histogram for {}-predicting model".format(self.label_style))
			plt.savefig(self.test_results_path + "/angle_err_histogram.png", bbox_inches="tight")

			plt.figure()
			n, _, _ = plt.hist(drone_angle_errors, bins=100, density = True, color="skyblue", ec="black")
			plt.title("Drone angle error histogram for {}-predicting model".format(self.label_style))
			plt.savefig(self.test_results_path + "/drone_angle_err_histogram.png", bbox_inches="tight")

			plt.figure()
			fig, ax = plt.subplots()
			plt.scatter(xs, thetas, c=theta_errors, norm=normalize, cmap ='seismic')
			ax.set_xlabel('Position X')
			ax.set_ylabel('Angle theta')
			cbar = plt.colorbar()
			cbar.set_label('Theta error')
			plt.savefig(self.test_results_path + "/theta_error_3Dplat.png", bbox_inches="tight")

			plt.figure()
			fig, ax = plt.subplots()
			plt.scatter(xs, thetas, c=drone_angle_errors, norm=normalize, cmap ='seismic')
			ax.set_xlabel('Position X')
			ax.set_ylabel('Angle theta')
			cbar = plt.colorbar()
			cbar.set_label('X error')
			plt.savefig(self.test_results_path + "/x_errors_3Dplat.png", bbox_inches="tight")

			plt.figure()
			fig, ax = plt.subplots()
			plt.scatter(drone_angles, thetas, c=drone_angle_errors, norm=normalize, cmap ='seismic')
			ax.set_xlabel('Drone angle')
			ax.set_ylabel('Angle theta')
			cbar = plt.colorbar()
			cbar.set_label('Drone angle error')
			plt.savefig(self.test_results_path + "/drone_angle_errors_3D_dt.png", bbox_inches="tight")

			plt.figure()
			fig, ax = plt.subplots()
			plt.scatter(drone_angles, thetas, c=theta_errors, norm=normalize, cmap ='seismic')
			ax.set_xlabel('Drone angle')
			ax.set_ylabel('Angle theta')
			cbar = plt.colorbar()
			cbar.set_label('Theta error')
			plt.savefig(self.test_results_path + "/theta_errors_3D_dt.png", bbox_inches="tight")

		elif self.label_style == "droneAngle":
			state_errors_np = np.array(state_errors)
			drone_angle_errors = state_errors_np[:,2]

			plt.figure()
			n, _, _ = plt.hist(drone_angle_errors, bins=100, density = True, color="skyblue", ec="black")
			plt.title("Drone angle error histogram for {}-predicting model".format(self.label_style))
			plt.savefig(self.test_results_path + "/drone_angle_err_histogram.png", bbox_inches="tight")

			plt.figure()
			fig, ax = plt.subplots()
			plt.scatter(drone_angles, thetas, c=drone_angle_errors, norm=normalize, cmap ='seismic')
			ax.set_xlabel('Drone angle')
			ax.set_ylabel('Angle theta')
			cbar = plt.colorbar()
			cbar.set_label('Drone angle error')
			plt.savefig(self.test_results_path + "/drone_angle_errors_3D_dt.png", bbox_inches="tight")

			
		### Saving summary
		save_file = open(self.test_results_path + '/test_results.txt','w')
		save_file.write('Average loss on test set: ' + str(avg_loss))
		save_file.write('\nAverage absolute error on reward: ' + str(avg_abs_rew_error))
		save_file.write('\nAverage error on reward: ' + str(avg_rew_error))
		save_file.write('\nVariance on reward: ' + str(rew_variance))		
		save_file.write('\nMean square error on reward: ' + str(mean_square_error_reward))
		save_file.write('\nAverage absolute error on state (0 if non applicable): ' + str(avg_abs_state_error))
		save_file.write('\nAverage error on state (0 if non applicable): ' + str(avg_state_error))


#################################################
# 					Main 						#
#################################################		

if __name__ == '__main__':
	# Loading config
	
	parser = argparse.ArgumentParser(description='This script trains a CNN on the images and label created from generate_images.py.')
	parser.add_argument('-c','--computer', help='Computer on which the script is launched.',required=True)
	parser.add_argument('-e','--environment',help='Environment from which the images are created', required=True)
	parser.add_argument('-l', '--label_type', help='Type of label that will be computed', required=True)
	parser.add_argument('-g', '--generation_mode', help='Mode of generation', required = True)
	parser.add_argument('-i', '--test_incremental', help='If specified, will test on incremental dataset', action = 'store_true')
	parser.add_argument('-n', '--model_name', help='Name of the model. Should in the form: env_label_genmode_#. Ex: CP_R_rand_1.', required=True)
	parser.add_argument('-m', '--model', help='Type of the CNN model', required=True)

	args = parser.parse_args()
	
	computer = ut.getComputer(args.computer)
	environment = ut.getImEnv(args.environment)
	label_type = ut.getLabelType(args.label_type)
	gen_mode = ut.getGenMode(args.generation_mode)
	model_name = ut.getModelName(args.model_name)
	model = ut.getModel(args.model)
	
	config = ut.loadYAMLFromFile('config_' + environment + '.yaml')

	if gen_mode == 'incremental':
		raise ValueError("The generation mode cannot be incremental for test_model.py (there must be a related model to test). If you want to test on incrementally generated data, add argument -i True") 
	test_incremental = args.test_incremental

	config['exp']['label_type'] = label_type
	config['exp']['env'] = environment
	config['exp']['gen_mode'] = gen_mode
	config['exp']['model_name'] = model_name
	config['cnn']['model'] = model

	tester = Tester(config, computer, test_incremental)
	#trainer.verify_data()
	tester.test_all_images()
