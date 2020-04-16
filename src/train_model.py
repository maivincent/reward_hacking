from __future__ import print_function, division
import os
import torch
import datetime
import cv2
import pandas as pd
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from tensorboardX import SummaryWriter
import utils as ut
import shutil
import sys
import argparse
import torchvision.models as models
import resnet
#################################################
#                     Dataloading                    #
#################################################


class ImageLabelDataset(Dataset):
    def __init__(self, csv_file, params_file, label_style, env, transform=None):
        self.label_style = label_style
        self.label_frame = pd.read_csv(csv_file)
        self.params_file = ut.loadYAMLFromFile(params_file)
        self.transform = transform
        self.env = env

    def __len__(self):
        return len(self.label_frame)

    def __getitem__(self, idx):
        img_name = self.label_frame.iloc[idx, 3]
        reward = self.label_frame.iloc[idx, 2].astype('float')
        x = self.label_frame.iloc[idx, 0].astype('float')            
        theta = self.label_frame.iloc[idx, 1].astype('float')
        drone_angle = self.label_frame.iloc[idx, 4].astype('float')
        if self.label_style == 'State':
            label = [x, theta]
        elif self.label_style == 'Reward':
            label = [reward]
        elif self.label_style == 'Angle':
            label = [theta]
        elif self.label_style == 'Angle_droneAngle':
            label = [theta, drone_angle]
        elif self.label_style == 'Distance':
            label = [x]
        elif self.label_style == 'State_droneAngle':
            label = [x, theta, drone_angle]
        elif self.label_style == 'droneAngle':
            label = [drone_angle]
        else:
            assert False # TODO: dk_ stuff
        label = NormalizeLabel(self.env, self.label_style, label)
        label = torch.FloatTensor(label)    
        image = cv2.imread(img_name)
        image = NormalizeImage(image, self.params_file['dataset_stats'])
        sample = {'image': image, 'label': label, 'theta': theta, 'x': x, 'reward': reward, 'drone_angle': drone_angle}
        #print(sample)
        if self.transform:
            sample = self.transform(sample)
        return sample

class DatasetStats(object):
    def __init__(self, path):
        self.csv_file = os.path.join(path, 'data_pairs.csv')
        self.label_frame = pd.read_csv(self.csv_file)
        self.nb_images = len(self.label_frame)
        self.mean, self.std = self.computePixelValueStats(100)

    def computePixelValueStats(self, pix_per_im):
        # For some images, randomly pick pix_per_im pixels and add them to a list.
        # Compute the average and std_dev on all these pixels for each layer.
        print("Computing dataset statistics...")
        means = []
        std_devs = []
        ns = []
        for im in range(int(self.nb_images/10)):
            if im%100 == 0:
                print(im)
            pixel_bank = []
            idx = int(np.random.random()*self.nb_images)
            img_name = self.label_frame.iloc[idx, 3]
            image = cv2.imread(img_name)
            if image is None:
                raise ValueError("Fail to open image: {}".format(img_name))
            for i in range(pix_per_im):
                try:
                    pixel_coords = [int(np.random.random()*image.shape[0]), int(np.random.random()*image.shape[1])]
                except:
                    raise ValueError("Fail to open image: {}".format(img_name))
                pixel_i = image[pixel_coords[0], pixel_coords[1]]
                pixel_bank.append(pixel_i)
            im_mean = np.mean(pixel_bank, axis = 0)
            std_dev = np.std(pixel_bank, axis=0)
            means.append(im_mean)
            std_devs.append(std_dev)
            ns.append(pix_per_im)
        mean = np.mean(means, axis=0)
        std_dev = ut.combineStd(std_devs, means, ns)
        print("Statistics computed - mean: {}, std: {}".format(mean, std_dev))
        return mean, std_dev

    def saveStats(self, path):
        stats_dict = {'dataset_stats': {'mean': self.mean.tolist(), 'std': self.std.tolist()}}
        print(stats_dict)
        ut.createYAMLFile(stats_dict, path)

    def getStats(self, path):
        return stats_dict



class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_h, new_w))
        sample['image'] = img

        return sample

def NormalizeLabel(env, label_style, label):
    if label_style == 'Reward':
        return label
    elif label_style == 'State':
        if env == 'cartpole':
            new_label = [0, 0]
            new_label[0] = label[0]/5 + 0.5     #  -2.5 < x < 2.5 --> 0 < x' < 1
            new_label[1] = label[1] + 0.5         #  -0.5 < theta < 0.5 --> 0 < theta' < 1   (theta is not too hard to predict because it is between -0.5 and 0.5)
        elif env == 'duckietown':
            new_label = [0, 0, 0]
            new_label[0] = label[0] + 0.6        # -0.5 < x < 0.5 --> 0 < x' < 1
            new_label[2] = np.sin(label[1])        # Angle --> sin                             (easier to do with cos/sin to avoid ambiguities in the loss function)
            new_label[1] = np.cos(label[1])        # Angle --> cos                             (easier to do with cos/sin to avoid ambiguities in the loss function)
    elif label_style == 'Angle':
        ## Only consider duckietown for now
        new_label = [0, 0]
        new_label[1] = np.sin(label[0])        # Angle --> sin                             (easier to do with cos/sin to avoid ambiguities in the loss function)
        new_label[0] = np.cos(label[0])        # Angle --> cos                             (easier to do with cos/sin to avoid ambiguities in the loss function)
    elif label_style == 'Angle_droneAngle':
        new_label = [0, 0, 0, 0]
        new_label[1] = np.sin(label[0])        # Angle --> sin                             (easier to do with cos/sin to avoid ambiguities in the loss function)
        new_label[0] = np.cos(label[0])        # Angle --> cos                             (easier to do with cos/sin to avoid ambiguities in the loss function)
        new_label[3] = np.sin(label[1])        # Angle --> sin                             (easier to do with cos/sin to avoid ambiguities in the loss function)
        new_label[2] = np.cos(label[1])        # Angle --> cos                             (easier to do with cos/sin to avoid ambiguities in the loss function)
    elif label_style == 'Distance':
        ## Only consider duckietown for now
        new_label = [0]
        new_label[0] = label[0] + 0.6        # -0.5 < x < 0.5 --> 0 < x' < 1
    elif label_style == 'State_droneAngle':
        new_label = [0, 0, 0, 0, 0]
        new_label[0] = label[0] + 0.6        # -0.5 < x < 0.5 --> 0 < x' < 1
        new_label[2] = np.sin(label[1])        # Angle --> sin                             (easier to do with cos/sin to avoid ambiguities in the loss function)
        new_label[1] = np.cos(label[1])        # Angle --> cos                             (easier to do with cos/sin to avoid ambiguities in the loss function)
        new_label[4] = np.sin(label[2])
        new_label[3] = np.cos(label[2])
    elif label_style == 'droneAngle':
        ## Only consider duckietown for now
        new_label = [0, 0]
        new_label[1] = np.sin(label[0])        # Angle --> sin                             (easier to do with cos/sin to avoid ambiguities in the loss function)
        new_label[0] = np.cos(label[0])        # Angle --> cos                             (easier to do with cos/sin to avoid ambiguities in the loss function)
    return new_label

def UnnormalizeLabel(env, label_style, label):
    if label_style == 'Reward':
        return label
    elif label_style == 'State':
        if env == 'cartpole':
            new_label = [0,0]         # x, theta
            new_label[0] = (label[0]-0.5)*5
            new_label[1] = label[1] - 0.5
        elif env == 'duckietown':
            new_label = [0,0]        # x, theta
            new_label[0] = label[0] - 0.6
            new_label[1] = np.arctan2(label[2], label[1])  # theta = atan2(sin(theta), cos(theta))
    elif label_style == 'Angle':
        new_label = [0]                # theta
        new_label[0] = np.arctan2(label[1], label[0])  # theta = atan2(sin(theta), cos(theta))
    elif label_style == 'Angle_droneAngle':
        new_label = [0, 0]            # theta, droneAngle
        new_label[0] = np.arctan2(label[1], label[0])  # theta = atan2(sin(theta), cos(theta))
        new_label[1] = np.arctan2(label[3], label[2])  # droneAngle = atan2(sin(droneAngle), cos(droneAngle))
    elif label_style == 'Distance':
        new_label = [0]                # x
        new_label[0] = label[0] - 0.6
    elif label_style == 'State_droneAngle':
        new_label = [0, 0, 0]         # x, theta, droneAngle
        new_label[0] = label[0] - 0.6
        new_label[1] = np.arctan2(label[2], label[1])  # theta = atan2(sin(theta), cos(theta))
        new_label[2] = np.arctan2(label[4], label[3])  # theta = atan2(sin(theta), cos(theta))
    elif label_style == 'droneAngle':
        new_label = [0]                # theta
        new_label[0] = np.arctan2(label[1], label[0])  # theta = atan2(sin(theta), cos(theta))
    return new_label

def NormalizeImage(image, stats = None):
    if not stats:
        image = image/np.maximum(256, image.max())
    elif stats:
        mean = stats['mean']
        std = stats['std']
        image = (image-mean)/std
    return image

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image_tensor = torch.from_numpy(image)

        sample['image'] = image_tensor
        return sample



#################################################
#                     Neural Net                    #
#################################################

class RewardFunctionHeadCartPole(nn.Module):
    def __init__(self):
        super(RewardFunctionHeadCartPole, self).__init__()

    def forward(self, x):
        #print(x)
        theta = x[:,:1]
        x = x[:,1:]
        reward = 1 - (3*torch.abs(theta/.5)/4) - (torch.abs(x/2.5)/2)**2
        #print(reward)
        #import pdb; pdb.set_trace()
        return reward # shape = (bs, 1)

class WeirdRewardFunctionHeadCartPole(nn.Module):
    def __init__(self):
        super(WeirdRewardFunctionHeadCartPole, self).__init__()

    def forward(self, x):
        #print(x)
        theta = x[:,:1]
        x = x[:,1:]
        reward = torch.sin(40*theta)*torch.cos(5*x)*torch.atan(24*x*theta)
        #print(reward)
        #import pdb; pdb.set_trace()
        return reward # shape = (bs, 1)         

class RewardFunctionHeadDuckieTown(nn.Module):
    def __init__(self):
        super(RewardFunctionHeadDuckieTown, self).__init__()

    def forward(self, x):
        dist = x[:,:1]
        angle = x[:,1:]
        reward = 1 - 1/2*torch.abs(angle) - 1/2*torch.abs(dist/0.15)**2
        return reward # shape = (bs, 1)


class RewardFunctionHeadModel(nn.Module):
    def __init__(self, net, head):
        super(RewardFunctionHeadModel, self).__init__()
        self.net = net
        self.head = head

    def forward(self, x):
        x = self.net(x)
        x = self.head(x)
        return x


#################################################
#                    Trainer                     #
#################################################

class Trainer():
    def __init__(self, train_set_prefix, config_path, config_exp, config_cnn):
        # Device parameters
        use_gpu = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_gpu else "cpu")
        self.computer = config_exp['computer']

        # Path configuration parameters
        self.training_data_path = config_path['training_data_path']
        self.training_plots_path = config_path['training_plots_path']
        self.training_data_path = config_path['training_data_path']
        self.model_path = config_path['model_path']
        self.inter_model_path = config_path['inter_model_path']
        self.training_set_path = config_path['training_set_path']
        self.testing_set_path = config_path['testing_set_path']
        self.cnn_params_path = config_path['cnn_params_path']
        self.train_losses_path = config_path['cnn_train_losses_path']
        self.test_losses_path = config_path['cnn_test_losses_path']

             # should not be needed ut.makeDir(self.training_data_path)

        # CNN configuration parameters
        self.label_style = config_exp['label_type']
        self.environment = config_exp['environment']
        self.model = config_cnn['model']

        self.nb_epochs = config_cnn['nb_epochs']
        learning_rate = config_cnn['learning_rate']
        print("Learning rate: {}".format(learning_rate))
        self.batch_size = config_cnn['batch_size']
        shuffle = config_cnn['shuffle']
        self.rescale_size = tuple(config_cnn['rescale_size'])
        assert self.rescale_size[0]%4 == 0 and self.rescale_size[1]%4==0   # Check that rescale_size is divisible by 4 (otherwise size computation does not work in the Net structure)
        nb_images = config_exp['nb_train_im']
        self.nb_batches = int(nb_images/self.batch_size)

        # CNN initialization
        self.net = self.initialize_net()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(list(self.net.parameters()), lr=learning_rate)
        dataset_stats = DatasetStats(self.training_set_path)
        dataset_stats.saveStats(self.cnn_params_path)
        del dataset_stats # Takes a lot of memory
        self.train_set = self.load_data(self.training_set_path, self.batch_size, shuffle)
        self.test_set = self.load_data(self.testing_set_path, 1, shuffle)
        self.train_losses = []
        self.test_losses = []
        torch.save(self.net.state_dict(), self.model_path)

        # Others
        self.drawer = ut.Drawer(self.training_plots_path + '/{}_CNN_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), self.label_style))
        self.test_incremental = config_exp['test_incremental']
        if self.test_incremental:
            from test_model import Tester
            config = {}
            config['exp'] = config_exp
            config['exp']['env'] = config['exp']['environment']
            config['paths'] = config_path
            config['cnn'] = config_cnn
            self.tester = Tester(config, self.computer, True)

    def load_data(self, path, batch_size, shuffle):
        composed_transform = transforms.Compose([Rescale(self.rescale_size), ToTensor()])
        labeled_dataset = ImageLabelDataset(os.path.join(path, 'data_pairs.csv'), self.cnn_params_path, self.label_style, self.environment, transform = composed_transform)
        data_loader = DataLoader(labeled_dataset, batch_size = batch_size, shuffle=shuffle, num_workers=4)
        print('Data loaded')
        return data_loader


    def initialize_net(self):
        nb_outputs = ut.nbOutputs(self.label_style, self.environment)

        if self.model == 'dk_resnet18_CP':
            nb_outputs = 2 # FIXME: hard coded
            reward_fn_head = RewardFunctionHeadCartPole()
            net = RewardFunctionHeadModel(models.resnet18(pretrained=False, num_classes=nb_outputs), reward_fn_head)
        elif self.model == 'dk_resnet18_CP_weird':
            nb_outputs = 2 # FIXME: hard coded
            reward_fn_head = WeirdRewardFunctionHeadCartPole()
            net = RewardFunctionHeadModel(models.resnet18(pretrained=False, num_classes=nb_outputs), reward_fn_head)
        elif self.model == 'dk_resnet18_DT':
            nb_outputs = 2 # FIXME: hard coded
            reward_fn_head = RewardFunctionHeadDuckieTown()
            net = RewardFunctionHeadModel(models.resnet18(pretrained=False, num_classes=nb_outputs), reward_fn_head)
        elif self.model == 'resnet18':
            net = models.resnet18(pretrained=False, num_classes=nb_outputs)    
            #### To use in case want the pretrained model: (remove num_classes as pretrained model only comes with original 1000 classes)
            # dim_feats = net.fc.in_features # =1000
            # net.fc = nn.Linear(dim_feats, nb_outputs)  

        elif self.model == 'resnet34':
            net = models.resnet34(pretrained=False, num_classes=nb_outputs)    
        elif self.model == 'resnet50':
            net = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=nb_outputs)
        elif self.model == 'resnet101':
            net = resnet.ResNet(resnet.Bottleneck, [3, 4, 23, 3], num_classes=nb_outputs)
        elif self.model == 'resnet152':
            net = resnet.ResNet(resnet.Bottleneck, [3, 8, 36, 3], num_classes=nb_outputs)

        net = net.float()
        net = net.to(self.device)
        return net

    def net_fwd(self, image):
        self.optimizer.zero_grad()
        outputs = self.net.forward(image.float())
        return outputs

    def compute_loss(self, pred, label):
        loss = self.criterion(pred.float(), label.float())
        return loss

    def net_bwd(self, loss):
        loss.backward()
        self.optimizer.step()

    def verify_data(self):
        for i, data in enumerate(self.train_set, 0):
            #print(data)
            image = data['image']
            label = data['label']
            print(str(i) + ' - TRAIN: Images size: ' + str(image.size()) + ', label size: ' + str(label.size()))


    def plot_grad_flow(self, named_parameters):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.draw()
        plt.pause(0.001)

    def train_one_epoch(self, epoch_nb):
        running_loss = 0.0
        self.net.train()
        for i, data in enumerate(self.train_set, 0):
            image = data['image']
            image = image.to(self.device)
            label = data['label']
            label = label.to(self.device)

            outputs = self.net_fwd(image)
            loss = self.compute_loss(outputs, label)
            self.net_bwd(loss)
            running_loss += loss.item()

            # Display
            sp_disp = 10
            if i % sp_disp == 0 and i != 0:
                writer.add_scalar('train_loss', running_loss/sp_disp, i + self.nb_batches*epoch_nb)
                print('[{}] Train loss: {}'.format(i + self.nb_batches*epoch_nb, running_loss/sp_disp))
                self.train_losses.append([i + self.nb_batches*epoch_nb, running_loss/sp_disp])

                if running_loss/sp_disp <= 0.0001:
                    print("Small running_loss: training finished")
                    torch.save(self.net.state_dict(), self.model_path)
                    print("Model saved.")
                    return True
                else:
                    running_loss = 0
                #if self.computer == 'local':
                #    self.plot_grad_flow(self.net.named_parameters())

            # Testing
            sp_test = 50
            if i % sp_test == sp_test - 1:
                test_loss = self.test()
                self.test_losses.append([i + self.nb_batches*epoch_nb, test_loss])
                writer.add_scalar('test_loss', test_loss, i + self.nb_batches*epoch_nb)

            # Saving learning results
            sp_save_res = 100
            if i % sp_save_res == sp_save_res - 1:
                self.save_results()

            # Saving intermediate
            sp_save = 10000
            if i % sp_save == sp_save - 1:
                torch.save(self.net.state_dict(), os.path.join(self.inter_model_path, 'model_{}.pth'.format(i + self.nb_batches*epoch_nb)))

        torch.save(self.net.state_dict(), self.model_path)
        if self.test_incremental:
            new_test_path = os.path.join(self.training_data_path, 'test_during_training', 'test_results_step_{}'.format(i + self.nb_batches*epoch_nb))
            self.tester.set_test_results_path(new_test_path)
            self.tester.net = self.tester.load_model()
            self.tester.test_all_images()
        print("Epoch done - Model saved.")
        return False

    def save_results(self):
        np.save(self.train_losses_path, self.train_losses)
        np.save(self.test_losses_path, self.test_losses)
        self.save_plot(self.train_losses, 'train')
        self.save_plot(self.test_losses, 'test')

    def save_plot(self, loss_array, mode):
        loss_array = np.array(loss_array)
        if mode == 'train':
            plot_title = 'Train loss over training time'
        elif mode == 'test':
            plot_title = 'Test loss over training time'
        self.drawer.savePlotPNG(loss_array[:,0], loss_array[:,1], 'Steps (batch size: ­{})'.format(self.batch_size), 'Loss', plot_title)

    def test(self):
        print('-----')
        print('Testing...')
        self.net.eval()
        losses = []
        for i, data in enumerate(self.test_set, 0):
            if i <= 100:
                image = data['image']
                image = image.to(self.device)
                label = data['label']
                label = label.to(self.device)
                outputs = self.net_fwd(image)
                #print('Outputs: {}   |   Labels: {}'.format(outputs.float().detach().cpu().numpy()[0], label.float().detach().cpu().numpy()[0]))
                loss = self.compute_loss(outputs, label)
                losses.append(loss.item())
            else:
                break
        test_loss = np.mean(losses)
        print('Test loss: {}'.format(test_loss))
        print('-----')
        self.net.train()
        return test_loss

    def train(self):
        for epoch in range(self.nb_epochs):
            print("Epoch: {}/{}".format(epoch, self.nb_epochs))
            if self.train_one_epoch(epoch):
                print("Finished training - loss below threshold")
                print("Model saved.")
                return
        print("Finished training: max epoch reached")
        torch.save(self.net.state_dict(), self.model_path)
        print("Model saved.")




#################################################
#                     Main                         #
#################################################        

if __name__ == '__main__':
    # Loading general config file
    
    parser = argparse.ArgumentParser(description='This script trains a CNN on the images and label created from generate_images.py.')
    parser.add_argument('-c','--computer', help='Computer on which the script is launched.',required=True)
    parser.add_argument('-e','--environment',help='Environment from which the images are created', required=True)
    parser.add_argument('-l', '--label_type', help='Type of label that will be computed', required=True)
    parser.add_argument('-g', '--generation_mode', help='Mode of generation', required=True)
    parser.add_argument('-n', '--model_name', help='Name of the model. Should in the form: env_label_genmode_#. Ex: CP_R_rand_1.', required=True)
    parser.add_argument('-m', '--model', help='Type of the CNN model', required=True)
    parser.add_argument('-r', '--learning_rate', help='Learning rate', default=None)
    parser.add_argument('-i', '--test_incremental', help='If specified, will test on incremental dataset', action = 'store_true')
    args = parser.parse_args()
    
    computer = ut.getComputer(args.computer)
    environment = ut.getImEnv(args.environment)
    label_type = ut.getLabelType(args.label_type)
    gen_mode = ut.getGenMode(args.generation_mode)
    model_name = ut.getModelName(args.model_name)
    model = ut.getModel(args.model)
    learning_rate = args.learning_rate
    test_incremental = args.test_incremental

    config = ut.loadYAMLFromFile('config_' + environment + '.yaml')

    config['exp']['computer'] = computer
    config['exp']['label_type'] = label_type
    config['exp']['environment'] = environment
    config['exp']['test_incremental'] = test_incremental
    config['exp']['gen_mode'] = gen_mode
    config['exp']['model_name'] = model_name
    config['cnn']['model'] = model
    if learning_rate:
        config['cnn']['learning_rate'] = float(learning_rate)

    # Building config_path
    config_path = config['paths']

    ### Changing temp_root if necessary
    temp_root = ''
    if computer == 'mila':
        # Getting local disk info
        temp_root = os.environ['SLURM_TMPDIR']

    ### CNN saving paths during usage (data training and model)
    use_cnn_path = os.path.join(temp_root, config['paths'][computer]['cnn'])
    use_cnn_label_path = os.path.join(use_cnn_path, environment, label_type, gen_mode, model_name)
    cnn_training_data_path = os.path.join(use_cnn_label_path, 'training_data')
    cnn_training_plots_path = os.path.join(cnn_training_data_path, 'plots')
    cnn_latest_model_path = os.path.join(use_cnn_label_path, 'latest_model.pth')
    cnn_inter_model_path = os.path.join(use_cnn_label_path, 'inter_models')
    cnn_params_path = os.path.join(use_cnn_label_path, 'cnn_params.yaml')
    cnn_train_losses_path = os.path.join(use_cnn_label_path, 'train_losses.npy')
    cnn_test_losses_path = os.path.join(use_cnn_label_path, 'test_losses.npy')
    ut.makeDir(cnn_training_data_path)
    ut.makeDir(cnn_training_plots_path)
    ut.makeDir(cnn_inter_model_path)
    config_path['training_data_path'] = cnn_training_data_path
    config_path['training_plots_path'] = cnn_training_plots_path
    config_path['model_path'] = cnn_latest_model_path
    config_path['inter_model_path'] = cnn_inter_model_path
    config_path['cnn_params_path'] = cnn_params_path
    config_path['cnn_train_losses_path'] = cnn_train_losses_path
    config_path['cnn_test_losses_path'] = cnn_test_losses_path

    ### Image dataset path
    use_images_path = os.path.join(temp_root, config['paths'][computer]['images'])
    training_set_path = os.path.join(use_images_path, environment, gen_mode, 'train')
    testing_set_path = os.path.join(use_images_path, environment, gen_mode, 'test')
    if test_incremental:
        if gen_mode == 'random_weird':
            incremental_mode = 'incremental_weird' 
        else: 
            incremental_mode = 'incremental'
        incremental_testing_set_path = os.path.join(use_images_path, environment, incremental_mode, 'test')
        config_path['incremental_testing_set_path'] = incremental_testing_set_path
    config_path['training_set_path'] = training_set_path
    config_path['testing_set_path'] = testing_set_path

    print('training_set_path: {}'.format(training_set_path))

    # Copying data to local disk from save repo (tmp1/maivincent)
    if computer == 'mila':
        print("Copying images")
        # Copying images 
        save_images_root_path = config['paths'][computer]['save_images'] 
        save_train_images_path = os.path.join(save_images_root_path, environment, gen_mode, 'train')
        save_test_images_path = os.path.join(save_images_root_path, environment, gen_mode, 'test')
        if test_incremental:
            save_test_incremental_images_path = os.path.join(save_images_root_path, environment, incremental_mode, 'test')
            ut.copyAndOverwrite(save_test_incremental_images_path, training_set_path)    
        ut.copyAndOverwrite(save_train_images_path, training_set_path)    
        ut.copyAndOverwrite(save_test_images_path, testing_set_path)    
        print("Copying images: done!")



    ### Training 
    # Initializing trainer
    trainer = Trainer('train_', config_path, config['exp'], config['cnn'])
    print("Initialized trainer.")
    # Initializing writer
    writer = SummaryWriter(logdir=cnn_training_data_path+ '/runs/{}_CNN_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), label_type))
    print("Initialized summary writer.")

    # Training
    nb_epochs = config['cnn']['nb_epochs']
    print("Launching training")
    trainer.train()
    
    writer.close()

    ### Copying results to save repo
    if computer == 'mila':
        # Copying CNN model from local disck to save repo (tmp1/maivincent)
        save_cnn_path = config['paths'][computer]['save_cnn']
        ut.copyAndOverwrite(use_cnn_path, save_cnn_path)
        print("Copying results: done!")
