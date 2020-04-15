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

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)
            
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

env = gym.make('MountainCar-v0')
state = env.reset()
model = Model()
model(torch.from_numpy(state).float())

done = False
best_so_far = [-200]
EPISODES = 2000

for episode in range(EPISODES):
    env.reset()
    actions = []
    doneEp = False
    
    while not doneEp:
        # get an action
        if np.random.random() <= epsilon: # Explore
            # get a random action
            action = np.random.randint(0,3)
        else:                             # Exploit
            # use network to get an educated action
            
            
        # take step
        state, reward, doneEp, _ = env.step(actions[-1])
        
        # Make done episode only if at goal position
        if state[0] < env.goal_position:
            doneEp = False
    

    
