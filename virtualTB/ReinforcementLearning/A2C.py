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
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
# StopTrainingOnNoModelImprovement
import pandas as pd
import gc
import psutil

env = gym.make('VirtualTB-v0')
env_action_space = env.action_space.shape[0]
print('env.action_space.shape[0]: ', env.action_space.shape[0] )

env.close()

# In[2]:


FLOAT = torch.FloatTensor
LONG = torch.LongTensor
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# Transition = namedtuple(
#     'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))
gamma = [0.99, 0.995, 0.997, 0.998,
         0.99, 0.995, 0.997, 0.998,
         0.99, 0.995, 0.997, 0.998,
         0.99, 0.995, 0.997, 0.998,
         0.99, 0.995, 0.997, 0.998,
         0.99, 0.995, 0.997, 0.998]
gae_lambda = [0.10, 0.10, 0.10, 0.10,
              0.30, 0.30, 0.30, 0.30,
              0.50, 0.50, 0.50, 0.50,
              0.70, 0.70, 0.70, 0.70,
              0.90, 0.90, 0.90, 0.90,
              0.95, 0.95, 0.95, 0.95]

print("gamma length: ", len(gamma))
print("gae_lambda length: ", len(gae_lambda))

mean_reward_max = 0
parameter = []

for i in range(len(gamma)):

    tmp_path = "./A2C_model01/"

    for j in range(7):
        new_logger = configure(tmp_path, ["csv", "tensorboard"],
                               log_suffix="_gamma={}_gae_lambda={}_{}".format(gamma[i], gae_lambda[i], j))

        env = gym.make('VirtualTB-v0')
        env.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        # Separate evaluation env
        eval_env = gym.make("VirtualTB-v0")
        # Stop training if there is no improvement after more than 3 evaluations
#         stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=5, verbose=1)
#         eval_callback = EvalCallback(eval_env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1)
        eval_callback = EvalCallback(eval_env, eval_freq=1000, verbose=1)


        model = A2C("MlpPolicy",
                    env,
                    gamma=gamma[i],
                    gae_lambda=gae_lambda[i],# 0.1/3/5/7/9/95
                    learning_rate=0.0007,
                    verbose=1)
                    # tensorboard_log="./sb3_log/")
                    # tensorboard_log=tmp_path)
        model.set_logger(new_logger)
        model.learn(total_timesteps=int(1e10), callback=eval_callback)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)

        if mean_reward_max <= mean_reward:
            mean_reward_max = mean_reward
            parameter.clear()
            parameter.append(gamma[i])
            parameter.append(gae_lambda[i])

        print("mean_reward: ", mean_reward)
        print("std_reward: ", std_reward)
        model.save("./A2C_model01/gamma={}_gae_lambda={}_{}.pkl".format(gamma[i], gae_lambda[i], j))

        obs = env.reset()

        action, next_state = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        print(info)
        # env.render()
        env.close()
        print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
        del model
        gc.collect()

print("mean_reward_max: ", mean_reward_max)
print("parameter: ", parameter)





