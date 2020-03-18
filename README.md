# Reward Hacking

## Installation

### Cloning
Cloning the repository and updating the submodules:
```
git clone https://github.com/maivincent/reward_hacking.git
cd reward_hacking
git submodule init
git submodule update
```

### Environment
Conda environment:
```
conda env create -f environment.yml 
conda activate rew_est
```

There may be some problems with `pyglet` (still have not figured out how to set it up correctly automatically).
You need to backtrack the installation to version 1.3.2:
```
pip3 uninstall pyglet
pip3 install --user pyglet==1.3.2
```

### Username (for computations on the Mila cluster)
If you want to use on the Mila cluster, you will need to change `maivince` for your username in:
`config_cartpole.yaml/paths/mila/`
`config_duckietown.yaml/paths/mila`
`config_duckietown.yaml/paths/mila`


## Description

### Objective

This repository contains several scripts allowing to test reward estimation and reward hacking in RL problems. The reward is given to the agent by a CNN that has been previously trained.

### Contents
- `src`
   Source code. Launch the scripts from there.
   **Wrappers**
   - `cartpole_mod_env.py` includes all the wrappers for the cartpole environment.
   - `duckietown_mod_env.py` includes all the wrappers for the Duckietown environment.

   **Configuration files**
   - `config_cartpole.yaml` includes the configuration parameters for the cartpole environment.
   - `config_duckietown.yaml` includes the configuration parameters for the Duckietown environment.
   - `config_duckietown_cam.yaml` includes the configuration parameters for the Duckietown environment with front camera input (not implemented yet).
   - `environment.yml` has information for the Conda environment that is being used. See **Installation** for more details.


   **Utils**
   - `resnet.py` (can be deleted I think, not sure) was for prototyping purposes
   - `utils.py`includes general usage functions, for math, image manipulation, data saving, data copying, plot drawing, script argument validity checking, etc.
   - `pytorch_soft_actor_critic` includes the Soft Actor Critic implementation (slightly modified from [here](https://github.com/pranz24/pytorch-soft-actor-critic)

   **Scripts**
   - `generate_images.py` allows you to generate images from the chosen environments, as well as the corresponding labels.
   - `train_model.py` allows to train a CNN over the images and their respective labels.
   - `test_model.py` allows to test a trained CNN over a test set and produce different statistics about the performances.
   - `rl_solver_sac.py` allows to train an Soft Actor Critic agent on different kinds of environments, including ground truth reward, noisy reward, and CNN-estimated reward.  Learning statistics are saved as well as the policies.
   - `view_rl_policy.py` allows to view in real time the behavior of a trained RL policy.
   - `compare_rl_agents.py` allows to compare the learning statistics of RL agents, averaging the behavior on the same environments and comparing different environments.

- `batches`
   Contains batch files that are useful to launch on the Mila cluster. They are not very well sorted.

- `gym-duckietown`
   Submodule which includes a personal version of the gym-duckietown simulator. It has been modified from the original so that a drone (top-down) view is implemented. The git submodule HEAD should follow the branch `drone_sim`

- `local_results`
	This folder will appear when you produce results on your local machine. It is composed in three main repositories:
   - `images` contains all the images that you have generated, as well as their respective labels in `data_pairs.csv`
   - `cnn` contains the CNN models as well as their training and testing statistics.
   - `rl` contains the RL models as well as their training statistics.

- `cluster_results`
	This is where you should copy the results you get from the cluster, using the same structure (`images` if you want to, `cnn` and `rl`). It will not create itself automatically (you'll need to create the repository).



## Usage

### Script arguments
You will find the necessary arguments as well as their definition by running `python scriptname.py --help`.
More details about the options are in the `utils.py` file, section `Arg checking`.

### Examples

#### Cartpole on the local computer
0. `cd src`
1. Generate images on the local computer, with randomly generated states.
   `python generate_images.py -c local -e cartpole -g random`
2. Train a CNN on this images, using the "Reward" as a label (could be also "State") and a resnet18 CNN (for now, the only implemented). This CNN will be named `my_cnn`
   `python -u train_model.py -c local -e cartpole -g random -n my_cnn -m resnet18 -l Reward`
3. Test the `my_cnn` CNN
   `python -u test_model.py -l Reward -c local -e cartpole -g random -n my_cnn -m resnet18`
4. Train a SAC agent on the Cartpole with the `my_cnn` as a reward provider
   `python -u rl_solver_sac.py -c local -e cartpole -t CP_CNN_Reward -g random -n my_cnn`
5. Train, for comparison, a SAC agent on the Cartpole with the groundtruth reward
   `python -u rl_solver_sac.py -c local -e cartpole -t CP_GT_Reward` 
6. Compare training stats of both environments
   `python compare_rl_agents.py -c local -f CP_CNN_Reward_my_cnn,CP_GT_Reward`
7. View the RL policy for the `CP_CNN_Reward_my_cnn` agent
   `python view_rl_policy.py -c local -e cartpole -n CP_CNN_Reward_my_cnn`


### Storage handling
When running in `-c local` option (on your computer), the storage of the results is automatically made according to the structure described in `local_results` description.

When running in `-c mila` option (on the Mila cluster), everything is stored in the `/network/tmp1/USERNAME/` repository. However, for speed purposes, at the beginning, all the necessary files, such as images and models, are copy-pasted to the local node. This is hardcoded (yes, it was maybe not the best decision). Don't forget to change your Mila username in the configuration files, as described in the **Installation** part.

You can also, for some scripts, run with the `-c transfer` option. This is when you want to use your computer for the script but use results that you got from the cluster in `cluster_results`.

## Troubleshooting

 If you get this error:
 ```
 AttributeError: 'ImageData' object has no attribute 'data'
 ```
 It's probably because you have not backtracked `pyglet` correctly. Refer to the **Installation** instructions to do so.


 When using on a cluster, you will need a virtual screen. In your scripts, use XVFB:
 `xvfb-run -a -s "-screen 0 1400x900x24" python YOUR_SCRIPT.py -A YOUR_ARGUMENT ... ` 