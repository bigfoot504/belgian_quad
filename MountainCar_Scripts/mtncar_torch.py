# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:04:08 2020
@author: locker
Me building a MountainCar solver from scratch using PyTorch.
"""

import gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import random

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)
            
    # function from super that must be modified for each subclass
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# taken from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
Transition = namedtuple('Transition',
                       ('state', 'action', 'next_state', 'reward'))

# taken from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory[]
        self.poisition = 0
        
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (elf.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__self(self):
        return len(self.memory)

env = gym.make('MountainCar-v0')
model = Model()
tgt_model = model
LEARNING_RATE = 0.001
GAMMA = 0.99 # discount factor
REPLAY_MEMORY = 10_000
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLF(optimizer, step_size=1, gamma=GAMMA)
memory =ReplayMemory(REPLAY_MEMORY)

done = False
best_so_far = [-200]
EPISODES = 2000
# function to decay epsilon from 1 to 0 over EPISODES horizon
epsilon = lambda episodes : 0#(EPISODES - episodes) / EPISODES
UPDATE_TARGET_EVERY = 5
SHOW_EVERY = 1
BATCH_SIZE = 64
SHOW = False
max_pos_ls = []
ep_loss_ls = []
ep_reward_ls = []

for episode in range(EPISODES):
    print(episode)
    
    state = env.reset() # reset current state
    state = torch.from_numpy(state).float() # convert state np2torch
    actions = []
    doneEp = False
    max_pos = -0.4
    ep_loss = 0
    ep_reward = 0
    
    while not doneEp:
        if episode % SHOW_EVERY == 0 and SHOW == True:
            env.render()
        
        # pass current torch state through NN
        Qs = model(state)
        
        # get an action
        if np.random.random() <= epsilon(episode): # Explore
            # get a random action
            action = np.random.randint(0,3)
        else:                             # Exploit
            # use network to get an educated action
            maxQ, action = torch.max(Qs, -1)
            action = action.item()
            
        # Take step
        state_new, reward, doneEp, _ = env.step(action)
        state_new = torch.from_numpy(state_new).float() # convert state np2torch
        
        # pass action through tgt model to get tgt Q (& eventually max future Q)
        tgt_Qs = tgt_model(state_new)
        max_future_Q = reward + GAMMA * torch.max(tgt_Qs) # max future Q is reward + discounted tgt Q
        
        # compute loss & update model
        loss_fn = nn.MSELoss()
        loss = loss_fn(Qs, tgt_Qs)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        # update tgt model if it's time (maybe every 5?)
        if episode % UPDATE_TARGET_EVERY == 0:
            tgt_model = model
        
        # update stats
        state = state_new
        if state[0] > max_pos:
            max_pos = state[0]
        ep_loss += loss.item()
        ep_reward += reward
        
        # if episode is done
        if doneEp:
            scheduler.step()
            
    max_pos_ls.append(max_pos)
    ep_loss_ls.append(ep_loss)
    ep_reward_ls.append(ep_reward)
        
    env.close()
    

    
