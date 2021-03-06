import numpy as np
import torch
from torch.optim import Adam
import gym
import time

class Buffer:
    """
    could be implemented as simple list
    
    """
    
    def __init__(self, obs_dim, act_dim, buffer_size, gamma=0.99, lam=0.95):
        self.observation_buf = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.action_buf = np.zeros((buffer_size,), dtype=np.float32)
        self.reward_buf = np.zeros((buffer_size,), dtype=np.float32)
        self.return_buf = np.zeros((buffer_size,), dtype=np.float32) #
        self.value_buf = np.zeros((buffer_size,), dtype=np.float32)
        self.advantage_buf = np.zeros((buffer_size,), dtype=np.float32)
        self.logp_buf = np.zeros((buffer_size,), dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        
        self.buffer_size = buffer_size
        self.traj_start_ind = 0
        self.curr_ind = 0
        
    def store(self, obs, act, reward, val, log_p_a):
        
        assert self.curr_ind < self.buffer_size

        self.observation_buf[self.curr_ind] = obs
        self.action_buf[self.curr_ind] = act
        self.reward_buf[self.curr_ind] = reward
        self.value_buf[self.curr_ind] = val
        self.logp_buf[self.curr_ind] = log_p_a
        
        self.curr_ind += 1
        
    def finish_with(self, val):
        
        traj_slice = slice(self.traj_start_ind, self.curr_ind)
        rewards = np.append(self.reward_buf[traj_slice], val)
        values = np.append(self.value_buf[traj_slice], val)
        
        # GAE
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        coeff = self.gamma * self.lam
        self.advantage_buf[traj_slice] = self.calculate_discounted_sums(deltas, coeff)
        
        # rewards-to-go, it is the targets for the value function
        self.return_buf[traj_slice] = self.calculate_discounted_sums(rewards, self.gamma)[:-1] # 最后一个元素不包括
  
        self.traj_start_ind = self.curr_ind

    def calculate_discounted_sums(self, nums_list, coeff):
        temp_discounted_sum = np.zeros((len(nums_list), ), dtype=np.float32)
        discounted_sum = 0
        for i, num in zip(reversed(range(len(nums_list))), reversed(nums_list)):
            discounted_sum += num
            temp_discounted_sum[i] = discounted_sum
            discounted_sum *= coeff
            
        return temp_discounted_sum
    
    def get_buf_data(self):
        
        # TODO
        # self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.observation_buf,
                    act=self.action_buf,
                    ret=self.return_buf,
                    advantages=self.advantage_buf,
                    log_p_a=self.logp_buf)
        
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

    def reset(self):   
        self.traj_start_ind = 0
        self.curr_ind = 0