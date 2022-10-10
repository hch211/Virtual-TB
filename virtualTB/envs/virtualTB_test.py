import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import torch
import torch.nn as nn
from virtualTB.model.ActionModel import ActionModel
from virtualTB.model.LeaveModel import LeaveModel
from virtualTB.model.UserModel import UserModel
from virtualTB.utils import *

class VirtualTB(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.n_item = 5
        self.n_user_feature = 88
        self.n_item_feature = 27
        self.max_c = 100
        self.obs_low = np.concatenate(([0] * self.n_user_feature, [0,0,0]))
        self.obs_high = np.concatenate(([1] * self.n_user_feature, [29,9,100]))
        self.observation_space = spaces.Box(low = self.obs_low, high = self.obs_high, dtype = np.int32)
        self.action_space = spaces.Box(low = -1, high = 1, shape = (self.n_item_feature,), dtype = np.float32)
        self.user_model = UserModel()
        self.user_model.load()
        self.user_action_model = ActionModel()
        self.user_action_model.load()
        self.user_leave_model = LeaveModel()
        self.user_leave_model.load()
        self.reset()

    def seed(self, sd = 0):
        torch.manual_seed(sd)

    @property
    def state(self):
        return np.concatenate((self.cur_user, self.lst_action, np.array([self.total_c])), axis = -1)

    #生成用户
    #cur_user 88d
    #用户特征由user_model生成，128d随机种子说明用户生成是随机的，只不过被单热向量给限制的比较unvariational，可以试一下softmax向量
    def __user_generator(self):
        # with shape(n_user_feature,)
        user = self.user_model.generate()
        
        # 决定用户在哪一页离开
        self.__leave = self.user_leave_model.predict_test(user)
        return user


    def step(self, action):
        # Action: tensor with shape (27, )
        self.lst_action = self.user_action_model.predict(FLOAT(self.cur_user).unsqueeze(0), FLOAT([[self.total_c]]), FLOAT(action).unsqueeze(0)).detach().numpy()[0]
        
        # 看起来，user_action_model的2维输出中，第一维是点击量
        # 第一维的是0-29，第二维是0-9
        reward = int(self.lst_action[0])
        # 点击量 累计
        self.total_a += reward
        self.total_c += 1
        self.rend_action = deepcopy(self.lst_action)
        done = (self.total_c >= self.__leave)
        
        if self.total_c % 2 == 0:
            self.cur_user = self.__user_generator().squeeze().detach().numpy()
        
        # done意味着一个episode结束，也就是一个用户浏览了若干页商品，然后退出了。之后新生成用户，并重置页面
        if done:
            self.cur_user = self.__user_generator().squeeze().detach().numpy()
            self.lst_action = FLOAT([0,0])
            
        return self.state, reward, done, {'CTR': self.total_a / self.total_c / 10}

    def reset(self):
        self.total_a = 0
        self.total_c = 0
        self.cur_user = self.__user_generator().squeeze().detach().numpy()
        self.lst_action = FLOAT([0,0])
        self.rend_action = deepcopy(self.lst_action)
        return self.state

    def render(self, mode='human', close=False):
        print('Current State:')
        print('\t', self.state)
        a, b = np.clip(self.rend_action, a_min = 0, a_max = None)
        print('User\'s action:')
        print('\tclick:%2d, leave:%s, index:%2d' % (int(a), 'True' if self.total_c > self.max_c else 'False', int(self.total_c)))
        print('Total clicks:', self.total_a)