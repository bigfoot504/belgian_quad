# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:04:08 2020
@author: locker
Me building a MountainCar solver from scratch using PyTorch.

Need to do:
    - Get train function set up properly (& global vs local state, action, reward, new_state variables).
    - Ensure update stats every is set up properly.
    - Get stats recording set up properly (maybe set it up to save them or plots somewhere upon completion?).
    - Set it up to save the model at completion.
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
from collections import namedtuple
import time
from matplotlib import pyplot as plt

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 3)
            
    # function from super that must be modified for each subclass
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# taken from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
Transition = namedtuple('Transition',
                       ('state', 'action', 'reward', 'new_state', 'done'))

# taken from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, transition):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

env = gym.make('MountainCar-v0')
model = Model()
tgt_model = model
LEARNING_RATE = 0.001
GAMMA = 0.99 # discount factor
REPLAY_MEMORY = 50_000
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=GAMMA)
memory = ReplayMemory(REPLAY_MEMORY)

EPISODES = 20_000
EPS_START = 0.99
EPS_END = 0.05
# function to decay epsilon from EPS_START to EPS_END linearly over EPISODES horizon
epsilon = lambda episode : (EPISODES - episode) / EPISODES * (EPS_START - EPS_END) + EPS_END
np2torch = lambda x: torch.Tensor(x)
torch2np = lambda x: x.numpy()
UPDATE_TARGET_EVERY = 5
SHOW_EVERY = 1
BATCH_SIZE = 128
SHOW = False
RUN_AVG_LEN = 100 # length off previous episodes over which running average is computed
max_pos_ls = []
ep_loss_ls = []
ep_reward_ls = []

def train(episode, memory, model, tgt_model):
    # train
    if len(memory) < BATCH_SIZE:
        return model, tgt_model, np2torch([0])
        
    # get random sample from replay memory
    minibatch = memory.sample(BATCH_SIZE)
    # load in states from minibatch
    states_list = np.array([transition[0] for transition in minibatch])
    # Get NN ouputs for each of minibatch current states from current main model
    Qs_list = model(np2torch(states_list))
    # Get corresponding new states from minibatch
    new_states_list = np.array([transition[3] for transition in minibatch])
    # Get NN ouputs for each of minibatch new states states from current target model
    future_Qs_list = tgt_model(np2torch(new_states_list))
    # initialize data to train on
    
    X = [] # state pairs from the game
    y = [] # labels/targets
    
    # get new Q's for each in minibatch
    for index, (state, action, reward, new_state, done) in enumerate(minibatch):
        # compute new Q using max future Q
        # or just reward if new_state is a terminal state
        if not done:
            max_future_Q = torch.max(future_Qs_list[index])
            # compute target Q value for index-th instance in minibatch
            new_Q = reward + GAMMA * max_future_Q
        else:
            new_Q = reward
        
        # recursive update to Q_{argmax{tgt Q}} from tgt model
        Qs = Qs_list[index]
        # superimpose max future Q update value onto that action's Q val in Qs list (update current qs)
        Qs[action] = new_Q

        # instead of passing image, pass state tuple
        X.append(state)
        y.append(Qs)
    
    # compute loss & update model
    loss_fn = nn.MSELoss()
    loss = loss_fn(model(np2torch(X)), Qs)
    #model.zero_grad()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # update tgt model if it's time (maybe every 5 episodes?)
    if episode % UPDATE_TARGET_EVERY == 0:
        tgt_model = model
        
    return model, tgt_model, loss
        


# Main Loop
for episode in tqdm(range(EPISODES)):
    #print(episode)
    
    state = env.reset() # reset current state
    actions = []
    doneEp = False
    max_pos = -0.4
    ep_loss = 0
    ep_reward = 0
    max_pos_run_avg = []
    ep_loss_run_avg = []
    ep_reward_run_avg = []
    
    while not doneEp:
        if episode % SHOW_EVERY == 0 and SHOW == True:
            env.render()
        
        # pass current torch state through NN
        Qs = model(np2torch(state))
        
        # get an action
        if np.random.random() <= epsilon(episode): # Explore
            # get a random action
            action = np.random.randint(0,3)
        else:                             # Exploit
            # use network to get an educated action
            maxQ, action = torch.max(Qs, -1)
            action = action.item()
            
        # Take step
        new_state, reward, doneEp, _ = env.step(action)
        
        # Store the transition in memory
        memory.push((state, action, reward, new_state, doneEp))
            
        '''
        # pass action through tgt model to get tgt Q (& eventually max future Q)
        tgt_Qs = tgt_model(np2torch(new_state))
        max_future_Q = reward + GAMMA * torch.max(tgt_Qs) # max future Q is reward + discounted tgt Q
        '''
        
        model, tgt_model, loss = train(episode, memory, model, tgt_model)
        
        if episode % SHOW_EVERY == 0 and SHOW == True:
            env.close()
        
        # update stats
        state = new_state
        if state[0] > max_pos:
            max_pos = state[0]
        ep_loss += loss.item()
        ep_reward += reward
        
        # if episode is done
        if doneEp:
            scheduler.step()
            
    # Update Episode Stats
    max_pos_ls.append(max_pos)
    ep_loss_ls.append(ep_loss)
    ep_reward_ls.append(ep_reward)
    # Update Running Averages
    if episode >= RUN_AVG_LEN:
        max_pos_run_avg.append(  max_pos_ls[  episode-RUN_AVG_LEN+1 : episode])
        ep_loss_run_avg.append(  ep_loss_ls[  episode-RUN_AVG_LEN+1 : episode])
        ep_reward_run_avg.append(ep_reward_ls[episode-RUN_AVG_LEN+1 : episode])



plt.plot(np.arange(len(max_pos_ls)), 
         np.array(     max_pos_ls))
plt.xlabel('Episode Num')
plt.ylabel('Max Position')
plt.title('Max Position Achieved by End of Episode')
plt.show()
plt.savefig(          'max_pos_ls.png')

plt.plot(np.arange(len(ep_loss_ls)), 
         np.array(     ep_loss_ls))
plt.xlabel('Episode Num')
plt.ylabel('Loss')
plt.title('MSE_Loss by End of Episode') # what is loss even computing again?***
plt.show()
plt.savefig(          'ep_loss_ls.png')

plt.plot(np.arange(len(ep_reward_ls)), 
         np.array(     ep_reward_ls))
plt.xlabel('Episode Num')
plt.ylabel('Reward')
plt.title('Total Reward by End of Episode') # NOT the same as max future Q
plt.show()
plt.savefig(          'ep_reward_ls.png')

plt.plot(np.arange(len(max_pos_run_avg)), 
         np.array(     max_pos_run_avg))
plt.xlabel('Episode Num')
plt.ylabel('Max Position Avg Last 100 Episodes')
plt.title('Max Position Running Average Last 100 Episodes')
plt.show()
plt.savefig(          'max_pos_run_avg.png')

plt.plot(np.arange(len(ep_loss_run_avg)), 
         np.array(     ep_loss_run_avg))
plt.xlabel('Episode Num')
plt.ylabel('Loss Avg Last 100 Episodes')
plt.title('MSE_Loss Running Average Last 100 Episodes')
plt.show()
plt.savefig(          'ep_loss_run_avg.png')

plt.plot(np.arange(len(ep_reward_run_avg)), 
         np.array(     ep_reward_run_avg))
plt.xlabel('Episode Num')
plt.ylabel('Reward Avg Last 100 Episodes')
plt.title('Episode Reward Running Average Last 100 Episodes')
plt.show()
plt.savefig(          'ep_reward_run_avg.png')


# save model
PATH = "C:/Users/locker/Documents/Python_Projects/MountainCar_Scripts/model_save_test.pt"
torch.save(model.state_dict(), PATH)

np.savetxt("C:/Users/locker/Documents/Python_Projects/MountainCar_Scripts/max_pos_ls.csv",        max_pos_ls,        delimiter=",")
np.savetxt("C:/Users/locker/Documents/Python_Projects/MountainCar_Scripts/ep_loss_ls.csv",        ep_loss_ls,        delimiter=",")
np.savetxt("C:/Users/locker/Documents/Python_Projects/MountainCar_Scripts/ep_reward_ls.csv",      ep_reward_ls,      delimiter=",")
np.savetxt("C:/Users/locker/Documents/Python_Projects/MountainCar_Scripts/max_pos_run_avg.csv",   max_pos_run_avg,   delimiter=",")
np.savetxt("C:/Users/locker/Documents/Python_Projects/MountainCar_Scripts/ep_loss_run_avg.csv",   ep_loss_run_avg,   delimiter=",")
np.savetxt("C:/Users/locker/Documents/Python_Projects/MountainCar_Scripts/ep_reward_run_avg.csv", ep_reward_run_avg, delimiter=",")
