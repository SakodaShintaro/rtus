# Real-Time Recurrent Learning using Trace Units in Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

***(Published in Advances in Neural Information Processing Systems (NeurIPS 2024))***


## Contents

- [Overview](#overview)
- [Repository Contents](#repository-contents)
- [Usage](#usage)




## Overview
This repository contains the code to reproduce the experiments present in our paper titled [Real-Time Recurrent Learning using Trace Units in Reinforcement Learning](https://arxiv.org/pdf/2409.01449) which was accepted at (NeurIPS 2024) 



## Repository Contents
- [src/agents](src/agents): Contains the agents used for the control experiments in the paper: Actor critic and Real-Time Actor critic which combines our architecture with Actor Critic.
- [src/nets](src/nets): The network architectures used in the paper: RTUs (ours), LRUs and GRUs.
- [src/algorithms](src/algorithms/): Contains a vanilla version of PPO, and RealTime PPO which combines our architecture with PPO.
- [src/experiments](src/experiments/): the starter files to run our control experiments.
- [src/envs](src/envs/): contains jax-implementations of POPGym and some wrappers needed to modify brax envs to be partially observable.
- [configs](configs): configuration files to replicate our experiments.


## Usage

### Replicating Control Experiments
The [configs](configs) folder contains some configuration files with the hyperparameters configurations used to generate the results in section 6 of the paper.

For example, to replicate the Ant-P results, you need to run:

```
python src/experiments/rtrl_ppo_experiment.py --config=configs/pobrax_config_rtu_rtrl.yaml
```

For other environments, you can change the following two attributes in the config file:
1.  ```domain```: 

    The domain can be:
    - ```pobrax_p```: for brax experiments with masked velocity (only position).
    - ```pobrax_v```: for brax experiments with masked position (only velocity).
    - ```popjax```: for POPGym experiments.
2.  ```env_name```:
    The environment name can be:

    - ```ant_p```,```cheetah_p```, ```hopper_p```, ```walker_p```: used with ```pobrax_p``` domain.
    - ```ant_v```, ```cheetah_v```, ```hopper_v```, ```walker_v```: used with ```pobrax_v```
    - ```CountRecallEasy```,```RepeatFirstEasy```,```ConcentrationEasy```,```AutoencodeEasy```,```HigherLowerEasy```: used with ```popjax``` domain.

Other important attributes:
1. ```continous_a``` and ```clip_action```: set both to True when the environment has continuous action spaces.

2. ```add_reward_prev_action```: setting this to true, appends the last reward and last action to the observation. This is currently implemented for discrete action spaces only.

For other agents, you can change the ```rec_fn``` attributes in the config file.



## Citation
Please cite our work if you find it useful:

```latex
@inproceedings{elelimy2024real,
    title={Real-Time Recurrent Learning using Trace Units in Reinforcement Learning},
    author={Elelimy, Esraa and White, Adam and Bowling, Michael and White, Martha},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2024}
}
```

## Acknowledgement:
The code in this repo is built on top of these libraries:
- [Brax](https://github.com/google/brax/)
- [PureJaxRL](https://github.com/luchris429/purejaxrl/)
- [popjaxrl](https://github.com/luchris429/popjaxrl)
