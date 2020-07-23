# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:04:08 2020
@author: locker
Me building a MountainCar solver from scratch using PyTorch.
Uses a combination of practices from sentdex's RL tutorials at pythonprogramming.net
and the RL tutorial on pytorch.org.

Try out the basic version with just one network. As opposed to mtncar_torch.py,
which uses a main model and a target model. The two network model is taking a
while to train and getting some odd results. Let's simplify the model here and
see if we get anythhing useful.
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

env = gym.make('MountainCar-v0')

STATE_DIM = len(env.observation_space.low)
NUM_ACTIONS = env.action_space.n
LEARNING_RATE = 0.001
GAMMA = 0.99 # discount factor
REPLAY_MEMORY = 50_000
EPISODES = 5_000
EPS_START = 0.99
EPS_END = 0.05
SHOW_EVERY = 1
SHOW = False
BATCH_SIZE = 128
RUN_AVG_LEN = 100 # length off previous episodes over which running average is computed
LOAD_PREV_MODEL = False # to load an old model to demo or build upon rather than training a new one from scratch
dirPATH = "/home/labuser/github/belgian_quad/MountainCar_Scripts/mtncar_basic_dqn_torch_results/"

# Define model class
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, NUM_ACTIONS)

    # function from super that must be modified for each subclass
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out

# This special tuple is part of a different optional method
# taken from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
Transition = namedtuple('Transition',
                       ('state', 'action', 'reward', 'new_state', 'done'))

# Replay memory object
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



# Initialize model network
model = Model()
# option to load old/previously-trained model
if LOAD_PREV_MODEL == True:
    model.load_state_dict(torch.load(f"{dirPATH}model_save_test_basic2.pt"))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=GAMMA)
memory = ReplayMemory(REPLAY_MEMORY)

# function to decay epsilon from EPS_START to EPS_END linearly over EPISODES horizon
epsilon = lambda episode : (EPISODES - episode) / EPISODES * (EPS_START - EPS_END) + EPS_END

np2torch = lambda x: torch.Tensor(x)
torch2np = lambda x: x.numpy()
# Note: also can go torch to float using vname.item()

max_pos_ls = []
ep_loss_ls = []
ep_reward_ls = []

def train(episode, memory, model):
    # train
    if len(memory) < BATCH_SIZE:
        return model, np2torch([0])

    # get random sample from replay memory
    minibatch = memory.sample(BATCH_SIZE)
    # load in states from minibatch
    states_list = np.array([transition[0] for transition in minibatch])
    # Get NN ouputs for each of minibatch current states from current main model
    Qs_list = model(np2torch(states_list))
    # Get corresponding new states from minibatch
    new_states_list = np.array([transition[3] for transition in minibatch])
    # Get NN ouputs for each of minibatch new states states from current target model
    future_Qs_list = model(np2torch(new_states_list))
    # initialize data to train on

    X = [] # state pairs from the game
    y = torch.zeros([BATCH_SIZE, NUM_ACTIONS]) # labels/targets

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
        X.append(state) # doesn't X end up being same as states_list?
        y[index,:] = Qs # y is unique though and we will need that

    # compute loss & update model
    loss_fn = nn.MSELoss()
    #print(np2torch(X).shape); print(Qs.shape) # temp
    loss = loss_fn(model(np2torch(X)), y)
    # PyTorch accumulates gradients by default, so they need to be reset in each pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return model, float(loss)



# Main Loop
for episode in tqdm(range(EPISODES)):

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
            env.close()
            env.render()

        # pass current torch state through NN
        Qs = model(np2torch(state))

        # get an action
        if np.random.random() <= epsilon(episode): # Explore
            # get a random action
            action = env.action_space.sample()
        else:                             # Exploit
            # use network to get an educated action
            maxQ, action = torch.max(Qs, -1)
            action = action.item()
        
        # Take step
        new_state, reward, doneEp, _ = env.step(action)

        # Push transition to memory
        memory.push((state, action, reward, new_state, doneEp))

        '''
        # pass action through tgt model to get tgt Q (& eventually max future Q)
        tgt_Qs = tgt_model(np2torch(new_state))
        max_future_Q = reward + GAMMA * torch.max(tgt_Qs) # max future Q is reward + discounted tgt Q
        '''

        model, loss = train(episode, memory, model)

        # if episode % SHOW_EVERY == 0 and SHOW == True:
        #     env.close()

        # update stats
        state = new_state
        if state[0] > max_pos:
            max_pos = state[0]
        ep_loss += loss
        ep_reward += reward

        # if episode is done
        # if doneEp:
        #     scheduler.step()

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
plt.savefig(          'max_pos_ls2.png')

plt.plot(np.arange(len(ep_loss_ls)),
         np.array(     ep_loss_ls))
plt.xlabel('Episode Num')
plt.ylabel('Loss')
plt.title('MSE_Loss by End of Episode') # what is loss even computing again?***
plt.show()
plt.savefig(          'ep_loss_ls_basic2.png')

plt.plot(np.arange(len(ep_reward_ls)),
         np.array(     ep_reward_ls))
plt.xlabel('Episode Num')
plt.ylabel('Reward')
plt.title('Total Reward by End of Episode') # NOT the same as max future Q
plt.show()
plt.savefig(          'ep_reward_ls_basic2.png')

plt.plot(np.arange(len(max_pos_run_avg)),
         np.array(     max_pos_run_avg))
plt.xlabel('Episode Num')
plt.ylabel('Max Position Avg Last 100 Episodes')
plt.title('Max Position Running Average Last 100 Episodes')
plt.show()
plt.savefig(          'max_pos_run_avg_basic2.png')

plt.plot(np.arange(len(ep_loss_run_avg)),
         np.array(     ep_loss_run_avg))
plt.xlabel('Episode Num')
plt.ylabel('Loss Avg Last 100 Episodes')
plt.title('MSE_Loss Running Average Last 100 Episodes')
plt.show()
plt.savefig(          'ep_loss_run_avg_basic2.png')

plt.plot(np.arange(len(ep_reward_run_avg)),
         np.array(     ep_reward_run_avg))
plt.xlabel('Episode Num')
plt.ylabel('Reward Avg Last 100 Episodes')
plt.title('Episode Reward Running Average Last 100 Episodes')
plt.show()
plt.savefig(          'ep_reward_run_avg_basic2.png')


# save model
#dirPATH = "C:/Users/locker/Documents/Python_Projects/MountainCar_Scripts/"
#dirPATH = "/home/uxv_swarm/github/belgian_quad/MountainCar_Scripts/"
dirPATH = "/home/labuser/github/belgian_quad/MountainCar_Scripts/mtncar_basic_dqn_torch_results/"
torch.save(model.state_dict(), f"{dirPATH}model_save_test_basic.pt")

np.savetxt(f"{dirPATH}max_pos_ls_basic2.csv",        max_pos_ls,        delimiter=",")
np.savetxt(f"{dirPATH}ep_loss_ls_basic2.csv",        ep_loss_ls,        delimiter=",")
np.savetxt(f"{dirPATH}ep_reward_ls_basic2.csv",      ep_reward_ls,      delimiter=",")
np.savetxt(f"{dirPATH}max_pos_run_avg_basic2.csv",   max_pos_run_avg,   delimiter=",")
np.savetxt(f"{dirPATH}ep_loss_run_avg_basic2.csv",   ep_loss_run_avg,   delimiter=",")
np.savetxt(f"{dirPATH}ep_reward_run_avg_basic2.csv", ep_reward_run_avg, delimiter=",")
