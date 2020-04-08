
from skimage import io, transform
import os
import shutil
import csv
import matplotlib.pyplot as plt
import copy
import numpy as np
import math
import yaml

#############################################
# 					math					#
#############################################

def limitAbsoluteVal(nb, limit):
	if np.abs(nb) > limit:
		nb = np.sign(nb)*limit
	return nb

def combineStd(stds, means, n, mode='direct'):
	# Combines the standard deviations of several datasets with different means and std deviations
	# Args:
	#	stds: list of standard deviations (can be a list of lists if we want to deal with several dimensions)
	# 	means: list of means (can be a list of lists if we want to deal with several dimensions)
	#	n: number of observations corresponding to each std/mean duo (must be a list of numbers)
	v = [np.power(stds[i], 2) for i in range(len(stds))]
	n = [len(stds[i]) for i in range(len(stds))]
	M = np.mean(means, axis=0)
	N = sum(n)
	elements_s2 = [[n[i] * np.power((means[i]-M), 2)] for i in range(len(means))]
	s2 = np.sum(elements_s2,0)
	if mode == 'sample':
		s1 = sum([(n[i]-1)*v[i] for i in range(len(stds))])
		V = (s1 + s2)/(N-1)
	elif mode == 'direct':
		s1 = sum([(n[i])*v[i] for i in range(len(stds))])
		V = (s1 + s2)/(N)
	else:
		raise ValueError('Combine STD needs mode for computing the std: direct or sample.')		
	STD = np.sqrt(V)
	return STD

def angleLimit(angle):
	while angle > np.pi:
		angle = angle - 2*np.pi
	while angle < -np.pi:
		angle = angle + 2*np.pi
	return angle

def angleDiff(a1, a2):
	return np.arctan2(np.sin(a1-a2), np.cos(a1-a2))



#############################################
# 			Image manipulation				#
#############################################

def rescaleImage(image, output_size):
	h, w = image.shape[:2]
	if isinstance(output_size, int):
		if h > w:
			new_h, new_w = output_size * h / w, output_size
		else:
			new_h, new_w = output_size, output_size * w / h
	else:
		new_h, new_w = output_size
	new_h, new_w = int(new_h), int(new_w)
	img = transform.resize(image, (new_h, new_w))

	return img

#################################################
#   				 Saver 						#
#################################################
def makeDir(path):
	# Check if directory exists, if not, create it
	if not os.path.exists(path):
		try:
			os.makedirs(path)
		except OSError:  
			print ("Creation of the directory %s failed" % path)
		else:  
			print ("Successfully created the directory %s " % path)   


class DataSaver():
	def __init__(self):
		pass

	def resetCSVFile(self, dir_path, csv_title):
		path = dir_path + "/" + csv_title + ".csv"
		if os.path.exists(path):
			os.remove(path)

	def save_data(self, dir_path, csv_title, value):
		makeDir(dir_path)
		path = dir_path + "/" + csv_title + ".csv"
		with open(path, "a") as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(value)

	def saveCSV(self, csv_title, my_list):
	# Save a list in a csv file
		path = self.output_path_root + "/" + csv_title + ".csv"
		with open(path, "w") as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(my_list)
	
	def saveMultiCSV(self, csv_title, list_of_lists, legend):
	# Save several lists in the same csv file. Legend is a list of strings to put in front of each list.
		path = self.output_path_root + "/" + csv_title + ".csv"
		assert len(legend) == len(list_of_lists), "Error writing CSV " + csv_title + ": legend has to have same size than list_of_lists"
		with open(path, "w") as csvfile:
			writer = csv.writer(csvfile)
			i = 0
			for a_list in list_of_lists:
				b_list = [legend[i]] + a_list
				writer.writerow(b_list)
				i += 1

#################################################
#					Drawer 						#
#################################################

class Drawer():
# Allows to save graphs as PNG with experiment details as TXT and data as CSV files.
	def __init__(self, output_path_root):
		self.output_path_root = output_path_root
		makeDir(self.output_path_root)

	def savePlotPNG(self, x, y, x_label, y_label, plot_title):
	# Save plot with only one x-y curve. x and y are lists.
		plt.subplots()
		plt.plot(x,y)
		plt.title(plot_title)
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		output_path = self.output_path_root + "/" + plot_title + ".png"
		plt.savefig(output_path, bbox_inches="tight")

	def saveMultiPlotPNG(self, x, y_list, x_label, y_label, plot_title, legend = False):
	# Save plot with several curves. Curves have the same x values in a list. y_list is a list of list of y data. 
	# legend is a list of strings corresponding to each curve.
		plt.subplots()
		for y_id in range(len(y_list)):
			y = y_list[y_id]
			if legend:
				plt.plot(x,y, label = legend[y_id])
		if legend:
			plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
		  fancybox=True, shadow=True)
		plt.title(plot_title)
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.ylim(bottom=0)
		output_path = self.output_path_root + "/" + plot_title + ".png"
		plt.savefig(output_path, bbox_inches="tight")  

	def saveMultiPlotWithStdPNG(self, x, y_list, std_list, x_label, y_label, plot_title, legend = False):
	# Save plot with several curves and std deviations. Curves have the same x values in a list. y_list is a list of list of y data.
	# std_list is a list of list of std deviations.
	# y_list and std_list must have the same amount of elements. Each of y_list and std_list lists must have the same amount of elements than x. 
	# legend is a list of strings corresponding to each curve.

		### Assertions
		assert len(y_list) == len(std_list)
		if legend:
			assert len(legend) == len(y_list)

		### Plotting curve one by one
		fig, ax = plt.subplots(1)
		for y_id in range(len(y_list)):
			y = y_list[y_id]
			std = std_list[y_id]
			assert len(y) == len(std)
			assert len(y) == len(x)
			if legend:
				ax.plot(x,y, label = legend[y_id])
			else:
				ax.plot(x,y)
			ax.fill_between(x, y+std, y-std, alpha=0.5)
		
		### Adding text
		if legend:
			ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
		  fancybox=True, shadow=True)
		ax.set_title(plot_title)
		ax.set_xlabel(x_label)
		ax.set_ylabel(y_label)
		ax.set_ylim(bottom=0)
		
		### Saving figure
		output_path = self.output_path_root + "/" + plot_title + ".png"
		fig.savefig(output_path, bbox_inches="tight") 

	def saveMultiXPlotWithStdPNG(self, x_list, y_list, std_list, x_label, y_label, plot_title, plot_save_name, legend = False):
	# Save plot with several curves and std deviations. Curves have different x values in a list of list x_list. y_list is a list of list of y data.
	# std_list is a list of list of std deviations.
	# x_list, y_list and std_list must have the same amount of elements. Each of y_list and std_list lists must have the same amount of elements than x. 
	# legend is a list of strings corresponding to each curve.

		### Assertions
		assert len(y_list) == len(std_list)
		assert len(y_list) == len(x_list)
		if legend:
			assert len(legend) == len(y_list)


		### Plotting curve one by one
		fig, ax = plt.subplots(1)

		for i in range(len(y_list)):
			y = y_list[i]
			if (y is not None) and (not np.isnan(np.sum(y))):
					x = x_list[i]
					std = std_list[i]
					assert len(y) == len(std)
					assert len(y) == len(x)
					if legend:
						ax.plot(x,y, label = legend[i])
					else:
						ax.plot(x,y)
					ax.fill_between(x, y+std, y-std, alpha=0.5)
		
		### Adding text
		if legend:
			ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
		  fancybox=True, shadow=True)
		ax.set_title(plot_title)
		ax.set_xlabel(x_label)
		ax.set_ylabel(y_label)
		ax.set_ylim(bottom=0)
		
		### Saving figure
		output_path = self.output_path_root + "/" + plot_save_name + ".png"
		fig.savefig(output_path, bbox_inches="tight") 

	def saveBarPlot(self, numbers, box_titles, plot_title, save_path):
		### Assertions
		assert len(numbers) == len(box_titles)

		### Plotting
		fig = plt.figure()
		ax = fig.add_axes([0,0,1,1])
		ax.bar(box_titles,numbers)
		ax.set_title(plot_title)
		fig.savefig(save_path, bbox_inches="tight")

def loadYAMLFromFile(path):
	file = open(path)
	yaml_string = file.read()
	yaml_dict = yaml.load(yaml_string)
	return yaml_dict

def createYAMLFile(dict, path):
	with open(path, 'w') as outfile:
		yaml.dump(dict, outfile, default_flow_style=False)




#################################
# 			Arg checking		#
#################################
def getComputer(computer):
	if not ((computer == 'mila') or (computer == 'local') or (computer == 'transfer')):
		raise ValueError('Wrong name of computer as argument -c. Options are: local, transfer or mila')
	return computer

def getImEnv(environment):
	if not ((environment == 'cartpole') or (environment == 'duckietown') or (environment == 'duckietown_cam')):
		raise ValueError('Wrong name of environment as argument -e. Options are: cartpole or duckietown')
	return environment
	
def getTrainEnv(env):
	list_env = ['CP_CNN_State',
	 'CP_CNN_Reward',
	 'CP_GT_Reward',
	 'CP_Original_Reward',
	 'CP_Noisy_Reward_10',
	 'CP_Noisy_Reward_20',
	 'CP_Noisy_Reward_30',
	 'DT_GT_Reward',
	 'DTC_GT_Reward',
	 'DT_Noisy_Reward_10',
	 'DT_Noisy_Reward_20',
	 'DT_Noisy_Reward_30',
	 'DT_Noisy_Reward_40',
	 'DT_Noisy_Reward_50',
	 'DT_Noisy_Reward_100',
	 'DT_Noisy_Reward_500',
	 'DT_R_CNN_Reward',
	 'DT_S_CNN_Reward',
	 'DT_Ssplit_CNN_Reward'
	 ]
	if not env in list_env:
		raise ValueError('Wrong name of training environment! Choices are : {}'.format(list_env))
	return env

def getLabelType(label_type):
	list_labels = ['Reward',
	'State',
	'Distance',
	'Angle',
	'Angle_droneAngle',
	'State_droneAngle',
	'droneAngle',
    'dk_cartpole']
	if not label_type in list_labels:
		raise ValueError('Wrong label type as argument -l. Options are: {}'.format(list_labels))
	return label_type

def getGenMode(gen_mode):
	list_modes = [None,
	 'None',
	 'random',
	 'random_weird',
	 'training',
	 'incremental',
	 'random_0',		#random_0 is for Duckietown env only - Duckiebot in center and 
	 'random_1',		#random_1 is for Duckietown env only - angle changes but Duckiebot in the center
	 'random_2',
	 'random_3',
	 'random_straight',
	 'random_0_straight',
	 'random_1_straight',
	 'random_2_straight',
	 'random_3_straight']
	if gen_mode == 'none':
		return ''
	elif not gen_mode in list_modes:
		raise ValueError('Wrong generation mode as argument -g. Options are: {}'.format(list_modes))
	return gen_mode


def getFolderList(folders):
	folder_list = folders.split(',')
	print("List of folders to be explored: ")
	for folder in folder_list:
		print('   - {}'.format(folder))
	return folder_list


def getModelName(model_name):
	print("Model name: {}".format(model_name))
	if model_name == 'None':
		return None
	else:
		return model_name

def getModel(model):
	list_models = [None,
	 'small',   # small homemade model
	 'dk_resnet18_CP',
	 'dk_resnet18_DT',
	 'dk_resnet18_CP_weird',
	 'resnet18',
	 'resnet34',
	 'resnet50',
	 'resnet101',
	 'resnet152']
	if not model in list_models:
		raise ValueError('Wrong model type as argument -m. Options are: {}'.format(list_models))
	return model

#################################
# 		  Boring stuff			#
#################################


def nbOutputs(label_style, env_name):
	if label_style == 'Reward':
		nb_outputs = 1
	elif label_style == 'State' and env_name == 'cartpole':
		nb_outputs = 2
	elif label_style == 'State' and env_name == 'duckietown':
		nb_outputs = 3
	elif label_style == 'Distance':
		nb_outputs = 1
	elif label_style == 'Angle' or label_style == 'droneAngle':
		nb_outputs = 2
	elif label_style == 'Angle_droneAngle':
		nb_outputs = 4
	elif label_style == 'State_droneAngle':
		nb_outputs = 5
	elif label_style == 'dk_DP':
		nb_outputs = 2
	elif label_style == 'dk_DT':
		nb_outputs = 2
	return nb_outputs

#################################
# 		   	  Copying			#
#################################

def copyAndOverwrite(from_path, to_path):
	print("Copying {} to {}".format(from_path, to_path))
	makeDir(to_path)
	### Files in from+path: if exist in to_path, overwrite. Otherwise, add. Do not remove other to_path files.
	files = [file for file in os.listdir(from_path) if os.path.isfile(os.path.join(from_path, file))]
	for file in files:
		if not os.path.exists(os.path.join(to_path, file)):
			shutil.copy(os.path.join(from_path, file), to_path)
		else:
			os.remove(os.path.join(to_path, file))
			shutil.copy(os.path.join(from_path, file), to_path)
	### Folders in from_path: if exists in to_path, copyAndOverwrite it. Otherwise, just copy.
	folders = [folder for folder in os.listdir(from_path) if os.path.isdir(os.path.join(from_path, folder))]
	for folder in folders:
		print(str(os.path.join(to_path, folder)) + ': ' + str(os.path.exists(os.path.join(to_path, folder))))
		if not os.path.exists(os.path.join(to_path, folder)):
			shutil.copytree(os.path.join(from_path, folder), os.path.join(to_path, folder))
		else:
			copyAndOverwrite(os.path.join(from_path, folder), os.path.join(to_path, folder))

def copyAndOverwriteFile(from_path, to_path):
	print("Copying {} to {}".format(from_path, to_path))
	if os.path.exists(to_path):
		os.remove(to_path)
	to_path_root = to_path.rsplit('/', 1)[0]
	makeDir(to_path_root)
	shutil.copy(from_path, to_path)


