# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:04:08 2020
@author: locker
Me building a MountainCar solver from scratch using PyTorch.
Uses a combination of practices from sentdex's RL tutorials at pythonprogramming.net
and the RL tutorial on pytorch.org.

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

env = gym.make('MountainCar-v0')

STATE_DIM = len(env.observation_space.low)
NUM_ACTIONS = env.action_space.n
LEARNING_RATE = 0.001
GAMMA = 0.99 # discount factor
REPLAY_MEMORY = 50_000
EPISODES = 20_000
EPS_START = 0.99
EPS_END = 0.05
UPDATE_TARGET_EVERY = 5
SHOW_EVERY = 1
SHOW = False
BATCH_SIZE = 128
RUN_AVG_LEN = 100 # length off previous episodes over which running average is computed
LOAD_PREV_MODEL = False # to load an old model to demo or build upon rather than training a new one from scratch
dirPATH = "/home/labuser/github/belgian_quad/MountainCar_Scripts/mtncar_torch_results/"

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



# Initialize model and tgt_model networks
model = Model()
tgt_model = Model()
# option to load old/previously-trained model
if LOAD_PREV_MODEL == True:
    model.load_state_dict(torch.load(f"{dirPATH}model_save_test.pt"))
# Copy weights from main model onto tgt_model
tgt_model.load_state_dict(model.state_dict())
tgt_model.eval()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=GAMMA)
memory = ReplayMemory(REPLAY_MEMORY)

# function to decay epsilon from EPS_START to EPS_END linearly over EPISODES horizon
epsilon = lambda episode : (EPISODES - episode) / EPISODES * (EPS_START - EPS_END) + EPS_END

np2torch = lambda x: torch.Tensor(x)
torch2np = lambda x: x.numpy()
# Note: also can go torch to float using vname.item()

def train(episode, memory, model, tgt_model):
    t = time.time()
    tflags = np.zeros(15)

    # train
    if len(memory) < BATCH_SIZE:
        return model, tgt_model, 0, tflags#np2torch([0])

    tflags[0] = time.time()-t; t = time.time()

    # get random sample from replay memory
    minibatch = memory.sample(BATCH_SIZE)
    tflags[1] = time.time()-t; t = time.time()
    # load in states from minibatch
    states_list = np.array([transition[0] for transition in minibatch])
    tflags[2] = time.time()-t; t = time.time()
    # Get NN ouputs for each of minibatch current states from current main model
    Qs_list = model(np2torch(states_list))
    tflags[3] = time.time()-t; t = time.time()
    # Get corresponding new states from minibatch
    new_states_list = np.array([transition[3] for transition in minibatch])
    tflags[4] = time.time()-t; t = time.time()
    # Get NN ouputs for each of minibatch new states states from current target model
    future_Qs_list = tgt_model(np2torch(new_states_list))
    tflags[5] = time.time()-t; t = time.time()
    # initialize data to train on

    X = [] # state pairs from the game
    tflags[6] = time.time()-t; t = time.time()
    y = torch.zeros([BATCH_SIZE, env.action_space.n]) # labels/targets
    tflags[7] = time.time()-t; t = time.time()

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
    tflags[8] = time.time()-t; t = time.time()

    # compute loss & update model
    loss_fn = nn.MSELoss()
    tflags[9] = time.time()-t; t = time.time()
    #print(np2torch(X).shape); print(Qs.shape) # temp
    loss = loss_fn(model(np2torch(X)), y)
    tflags[10] = time.time()-t; t = time.time()
    # PyTorch accumulates gradients by default, so they need to be reset in each pass
    optimizer.zero_grad()
    tflags[11] = time.time()-t; t = time.time()
    loss.backward()
    tflags[12] = time.time()-t; t = time.time()
    optimizer.step()
    tflags[13] = time.time()-t; t = time.time()

    # update tgt model if it's time (maybe every 5 episodes?)
    if episode % UPDATE_TARGET_EVERY == 0:
        tgt_model.load_state_dict(model.state_dict())
    tflags[14] = time.time()-t; t = time.time()
    
    return model, tgt_model, float(loss), tflags



max_pos_ls = []
ep_loss_ls = []
ep_reward_ls = []

# Main Loop
for episode in tqdm(range(EPISODES)):
    input(f'Press ENTER to start episode = {episode}')

    state = env.reset() # reset current state
    actions = []
    doneEp = False
    max_pos = -0.4
    ep_loss = 0
    ep_reward = 0
    # max_pos_run_avg = []
    # ep_loss_run_avg = []
    # ep_reward_run_avg = []

    t = time.time()
    tflag0 = []
    tflag1 = []
    tflag2 = []
    tflag3 = []
    tflag4 = []
    tflag5 = []
    tflag6 = []
    tflag7 = []
    tflag8 = []
    tflag9 = []
    tflag10 = []
    tflag11 = []
    tflag12 = []
    tflag13 = []
    tflag14 = []
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

        model, tgt_model, loss, tflags = train(episode, memory, model, tgt_model) # tflags are temporary for debugging time
        tflag0.append(tflags[0])
        tflag1.append(tflags[1])
        tflag2.append(tflags[2])
        tflag3.append(tflags[3])
        tflag4.append(tflags[4])
        tflag5.append(tflags[5])
        tflag6.append(tflags[6])
        tflag7.append(tflags[7])
        tflag8.append(tflags[8])
        tflag9.append(tflags[9])
        tflag10.append(tflags[10])
        tflag11.append(tflags[11])
        tflag12.append(tflags[12])
        tflag13.append(tflags[13])
        tflag14.append(tflags[14])

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

    tot_time = [sum(tflag0), sum(tflag1), sum(tflag2), sum(tflag3), sum(tflag4), sum(tflag5), sum(tflag6), sum(tflag7),
                sum(tflag8), sum(tflag9), sum(tflag10), sum(tflag11), sum(tflag12), sum(tflag13), sum(tflag14)] 
    print(f'\nflag 0: sum = {tot_time[0]:.8f}, mean = {np.mean(tflag0):.8f}, cv = {np.std(tflag0)/np.mean(tflag0):.8f}, prop_time = {tot_time[0]/sum(tot_time):.4f}')
    print(f'flag 1: sum = {tot_time[1]:.8f}, mean = {np.mean(tflag1):.8f}, cv = {np.std(tflag1)/np.mean(tflag1):.8f}, prop_time = {tot_time[1]/sum(tot_time):.4f}')
    print(f'flag 2: sum = {tot_time[2]:.8f}, mean = {np.mean(tflag2):.8f}, cv = {np.std(tflag2)/np.mean(tflag2):.8f}, prop_time = {tot_time[2]/sum(tot_time):.4f}')
    print(f'flag 3: sum = {tot_time[3]:.8f}, mean = {np.mean(tflag3):.8f}, cv = {np.std(tflag3)/np.mean(tflag3):.8f}, prop_time = {tot_time[3]/sum(tot_time):.4f}')
    print(f'flag 4: sum = {tot_time[4]:.8f}, mean = {np.mean(tflag4):.8f}, cv = {np.std(tflag4)/np.mean(tflag4):.8f}, prop_time = {tot_time[4]/sum(tot_time):.4f}')
    print(f'flag 5: sum = {tot_time[5]:.8f}, mean = {np.mean(tflag5):.8f}, cv = {np.std(tflag5)/np.mean(tflag5):.8f}, prop_time = {tot_time[5]/sum(tot_time):.4f}')
    print(f'flag 6: sum = {tot_time[6]:.8f}, mean = {np.mean(tflag6):.8f}, cv = {np.std(tflag6)/np.mean(tflag6):.8f}, prop_time = {tot_time[6]/sum(tot_time):.4f}')
    print(f'flag 7: sum = {tot_time[7]:.8f}, mean = {np.mean(tflag7):.8f}, cv = {np.std(tflag7)/np.mean(tflag7):.8f}, prop_time = {tot_time[7]/sum(tot_time):.4f}')
    print(f'flag 8: sum = {tot_time[8]:.8f}, mean = {np.mean(tflag8):.8f}, cv = {np.std(tflag8)/np.mean(tflag8):.8f}, prop_time = {tot_time[8]/sum(tot_time):.4f}')
    print(f'flag 9: sum = {tot_time[9]:.8f}, mean = {np.mean(tflag9):.8f}, cv = {np.std(tflag9)/np.mean(tflag9):.8f}, prop_time = {tot_time[9]/sum(tot_time):.4f}')
    print(f'flag 10: sum = {tot_time[10]:.8f}, mean = {np.mean(tflag10):.8f}, cv = {np.std(tflag10)/np.mean(tflag10):.8f}, prop_time = {tot_time[10]/sum(tot_time):.4f}')
    print(f'flag 11: sum = {tot_time[11]:.8f}, mean = {np.mean(tflag11):.8f}, cv = {np.std(tflag11)/np.mean(tflag11):.8f}, prop_time = {tot_time[11]/sum(tot_time):.4f}')
    print(f'flag 12: sum = {tot_time[12]:.8f}, mean = {np.mean(tflag12):.8f}, cv = {np.std(tflag12)/np.mean(tflag12):.8f}, prop_time = {tot_time[12]/sum(tot_time):.4f}')
    print(f'flag 13: sum = {tot_time[13]:.8f}, mean = {np.mean(tflag13):.8f}, cv = {np.std(tflag13)/np.mean(tflag13):.8f}, prop_time = {tot_time[13]/sum(tot_time):.4f}')
    print(f'flag 14: sum = {tot_time[14]:.8f}, mean = {np.mean(tflag14):.8f}, cv = {np.std(tflag14)/np.mean(tflag14):.8f}, prop_time = {tot_time[14]/sum(tot_time):.4f}')
    [print(sum(tot_time[0:i+1])/sum(tot_time)) for i in range(15)]

    # Update Episode Stats
    max_pos_ls.append(max_pos)
    ep_loss_ls.append(ep_loss)
    ep_reward_ls.append(ep_reward)
    # Update Running Averages
    # if episode >= RUN_AVG_LEN:
    #     max_pos_run_avg.append(  max_pos_ls[  episode-RUN_AVG_LEN+1 : episode])
    #     ep_loss_run_avg.append(  ep_loss_ls[  episode-RUN_AVG_LEN+1 : episode])
    #     ep_reward_run_avg.append(ep_reward_ls[episode-RUN_AVG_LEN+1 : episode])



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
plt.savefig(          'ep_loss_ls2.png')

plt.plot(np.arange(len(ep_reward_ls)),
         np.array(     ep_reward_ls))
plt.xlabel('Episode Num')
plt.ylabel('Reward')
plt.title('Total Reward by End of Episode') # NOT the same as max future Q
plt.show()
plt.savefig(          'ep_reward_ls2.png')

# plt.plot(np.arange(len(max_pos_run_avg)),
#          np.array(     max_pos_run_avg))
# plt.xlabel('Episode Num')
# plt.ylabel('Max Position Avg Last 100 Episodes')
# plt.title('Max Position Running Average Last 100 Episodes')
# plt.show()
# plt.savefig(          'max_pos_run_avg2.png')
#
# plt.plot(np.arange(len(ep_loss_run_avg)),
#          np.array(     ep_loss_run_avg))
# plt.xlabel('Episode Num')
# plt.ylabel('Loss Avg Last 100 Episodes')
# plt.title('MSE_Loss Running Average Last 100 Episodes')
# plt.show()
# plt.savefig(          'ep_loss_run_avg2.png')
#
# plt.plot(np.arange(len(ep_reward_run_avg)),
#          np.array(     ep_reward_run_avg))
# plt.xlabel('Episode Num')
# plt.ylabel('Reward Avg Last 100 Episodes')
# plt.title('Episode Reward Running Average Last 100 Episodes')
# plt.show()
# plt.savefig(          'ep_reward_run_avg2.png')


# save model
#dirPATH = "C:/Users/locker/Documents/Python_Projects/MountainCar_Scripts/"
#dirPATH = "/home/uxv_swarm/github/belgian_quad/MountainCar_Scripts/"
dirPATH = "/home/labuser/github/belgian_quad/MountainCar_Scripts/mtncar_torch_results/"
torch.save(model.state_dict(), f"{dirPATH}model_save_test2.pt")

np.savetxt(f"{dirPATH}max_pos_ls2.csv",        max_pos_ls,        delimiter=",")
np.savetxt(f"{dirPATH}ep_loss_ls2.csv",        ep_loss_ls,        delimiter=",")
np.savetxt(f"{dirPATH}ep_reward_ls2.csv",      ep_reward_ls,      delimiter=",")
# np.savetxt(f"{dirPATH}max_pos_run_avg2.csv",   max_pos_run_avg,   delimiter=",")
# np.savetxt(f"{dirPATH}ep_loss_run_avg2.csv",   ep_loss_run_avg,   delimiter=",")
# np.savetxt(f"{dirPATH}ep_reward_run_avg2.csv", ep_reward_run_avg, delimiter=",")
