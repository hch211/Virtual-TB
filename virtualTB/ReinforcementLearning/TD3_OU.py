#!/usr/bin/env python
# coding: utf-8

import os
import sys
import gym
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
import gym
import math
import torch
import random
import virtualTB
import time, sys
import configparser
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gym import wrappers
from copy import deepcopy
from collections import namedtuple
import pickle
from stable_baselines3 import TD3, HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
import pandas as pd
import gc
import psutil


env = gym.make('VirtualTB-v0')
env_action_space = env.action_space.shape[0]
print('env.action_space.shape[0]: ', env.action_space.shape[0])

env.close()

# In[2]:

FLOAT = torch.FloatTensor
LONG = torch.LongTensor
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# Transition = namedtuple(
#     'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

# gamma = [0.95, 0.97, 0.98, 0.99, 0.995, 0.997, 0.998,
#          0.95, 0.97, 0.98, 0.99, 0.995, 0.997, 0.998,
#          0.98, 0.99, 0.995, 0.997, 0.98, 0.99, 0.995]
# sigma = [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10,
#          0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30,
#          0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20]
# policy_noise = [0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20,
#                 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20,
#                 0.40, 0.40, 0.40, 0.40, 0.20, 0.20, 0.20]
# delay = [2, 2, 2, 2, 2, 2, 2,
#          2, 2, 2, 2, 2, 2, 2,
#          2, 2, 2, 2, 4, 4, 4]

# gamma = [0.98, 0.99, 0.995, 0.997, 0.998]
# # policy_noise = [0.20, 0.20, 0.20, 0.20, 0.20]
# # delay = [2, 2, 2, 2, 2]


gamma = [0.995, 0.995, 0.995, 0.995, 0.995,
         0.995, 0.995, 0.995, 0.995, 0.995, 0.995]
# sigma = [0.25, 0.25, 0.25, 0.25, 0.25,
#          0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
policy_noise = [0.1, 0.2, 0.3, 0.4, 0.5,
                0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
delay = [2, 2, 2, 2, 2,
         1, 2, 3, 4, 5, 6]


# gamma = [0.995]
# sigma = [0.3]
# policy_noise = [0.2]
# delay = [2]

print("gamma length: ", len(gamma))
# print("sigma length: ", len(sigma))
print("policy_noise length: ", len(policy_noise))
print("delay length: ", len(delay))

mean_reward_max = 0
parameter = []

for i in range(len(gamma)):

    tmp_path = "./TD3_model02/"

    for j in range(1):
        # new_logger = configure(tmp_path, ["csv", "tensorboard"],
        #                        log_suffix="_gamma={}_sigma={}_policy_noise={}_delay={}_{}".format(gamma[i], sigma[i],
        #                                                                                           policy_noise[i],
        #                                                                                           delay[i], j))

        new_logger = configure(tmp_path, ["csv", "tensorboard"],
                               log_suffix="_gamma={}_policy_noise={}_delay={}_{}".format(gamma[i],
                                                                                         policy_noise[i],
                                                                                         delay[i], j))

        env = gym.make('VirtualTB-v0')
        env.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        n_actions = env.action_space.shape[-1]
        # print('exploration noise std (sigma): ', sigma[i]*np.ones(n_actions))
        # #exploration noise std
        # action_noise = NormalActionNoise(mean=np.zeros(n_actions),
        #                                  sigma=sigma[i] * np.ones(n_actions))  # sigma = sigma[i]
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                    sigma=0.25 * np.ones(n_actions),
                                                    theta=0.15)

        # Separate evaluation env
        eval_env = gym.make("VirtualTB-v0")
        # Stop training if there is no improvement after more than 3 evaluations
        stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
        eval_callback = EvalCallback(eval_env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1)


        model = TD3("MlpPolicy",
                    env,
                    action_noise=action_noise,  # 两种噪声
                    # #policy noise std <= 0.5
                    target_policy_noise=policy_noise[i],
                    # #delay update frequency
                    policy_delay=delay[i],
                    gamma=gamma[i],  # 0.95~0.997/8
                    learning_rate=0.0001,
                    buffer_size=100000,
                    device=device,
                    verbose=1)
                    # tensorboard_log="./sb3_log/")
                    # tensorboard_log=tmp_path)
        model.set_logger(new_logger)
        # train the model
        model.learn(total_timesteps=int(1e10), callback=eval_callback, log_interval=4)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)

        if mean_reward_max <= mean_reward:
            mean_reward_max = mean_reward
            parameter.clear()
            parameter.append(gamma[i])
            # parameter.append(sigma[i])
            parameter.append(policy_noise[i])
            parameter.append(delay[i])

        print("mean_reward: ", mean_reward)
        print("std_reward: ", std_reward)
        # model.save("./TD3_model02/gamma={}_sigma={}_policy_noise={}_delay={}_{}.pkl".format(gamma[i], sigma[i],
        #                                                                                   policy_noise[i], delay[i], j))

        model.save("./TD3_model03/gamma={}_policy_noise={}_delay={}_{}.pkl".format(gamma[i],
                                                                                   policy_noise[i],
                                                                                   delay[i],j))

        obs = env.reset()

        action, next_state = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        print(info)
        # env.render()
        # print("obs: ", type(obs))
        # print(sys.getsizeof(obs) / 1024 / 1024, 'MB')
        env.close()
        # print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
        # del model
        # gc.collect()


print("mean_reward_max: ", mean_reward_max)
print("parameter: ", parameter)
