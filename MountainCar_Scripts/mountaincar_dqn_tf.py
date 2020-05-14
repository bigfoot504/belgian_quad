'''
This is the code base I am currently working on developing.

Work in progress to make a simple DQN to play and solve mountain car.
Input into the NN must be the state.
Output must be the action to take.

***Need to go back and read this article to see how to proceed:
https://medium.com/@ts1829/solving-mountain-car-with-q-learning-b77bf71b1de2
And use help from https://pythonprogramming.net/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
for conceptual help.
Next task is to look at the -200 reward thing below (that came from the blobenv dqn
script and see how to change it to fit this one).
'''

import gym
import tensorflow as tf
from tensorflow.keras import layers
from keras.callbacks import TensorBoard # bc tf....Tensorbaord hasn't _write_logs attribute
from collections import deque
import time
import os
from tqdm import tqdm
import numpy as np
import random

# uncomment if you want to watch; or filepath None
LOAD_MODEL = None#"models/2x64_MountainCar_DQN_Custom___-95.00max__-95.00avg__-95.00min__1586188121.model"

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 20_000    # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000 # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64            # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5        # Terminal states (end of episodes)
MODEL_NAME = '2x64_MountainCar_DQN_Custom'
MIN_REWARD = -200              # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 20_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False        # change to True if want to render

# For stats
ep_rewards = [-100]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

# Own Tensorboard class; prevents keras from writing new log files for every fit
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)
        

env = gym.make('MountainCar-v0')


class DQNAgent():
    def __init__(self):
        
        # Main model
        # gets trained every step
        self.model = self.create_model()
        
        # Target model
        # this is what we .predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
                
        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{MODEL_NAME}-{int(time.time())}")
        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
    
    
    def create_model(self):
        
        if LOAD_MODEL is not None: # allows us to load a model from before instead of training a new one from scratch
            print(f"Loading {LOAD_MODEL}")
            model = tf.keras.models.load_model(LOAD_MODEL)
            print(f"Model {LOAD_MODEL} loaded!")
        else:
            model = tf.keras.Sequential()
            # Adds a densely-connected layer with 64 units to the model:
            model.add(layers.Dense(64, activation='relu', input_shape=env.env.observation_space.shape))
            # Add another:
            model.add(layers.Dense(64, activation='relu'))
            # Add an output layer with 3 output units: (0,1,2 for left, no action, right)
            model.add(layers.Dense(env.action_space.n, activation='linear'))
            
            model.compile(loss="mse",
                          optimizer=tf.keras.optimizers.Adam(lr=0.001),
                          metrics=['accuracy'])
        return model
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
        # transition is state, action, reward, new state, whether it was done

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0:3]
    
    def train(self, terminal_state, step):
        
        # first, check: should we train?
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        # Note: still going w/ minibatches & replay memory here b/c still 
        # could help train NN. Only, our states are pos,vel pairs instead of images.
        # And we don't need special image-based layers nor the 4-image thing to
        # see movement across images.

        # get random sample from replay memory
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # load in states from minibatch
        current_states = np.array([transition[0] for transition in minibatch])
        # FYI: one transition instance in minibatch looks like this:
        # (array([-0.68369063, -0.01586011]), <-- current state
        # 0,                                  <-- action
        # -1.0,                               <-- reward
        # array([-0.69939568, -0.01570505]),  <-- new state
        # False)                              <-- whether done
        # Get NN ouputs for each of minibatch current states from current main model
        current_qs_list = self.model.predict(current_states)

        # Get corresponding new states from minibatch
        new_current_states = np.array([transition[3] for transition in minibatch])
        # Get NN ouputs for each of minibatch new states states from current target model
        future_qs_list = self.target_model.predict(new_current_states)
        
        X = [] # state pairs from the game
        y = [] # labels/targets

        # get max future q's for each in minibatch
        for index, (current_states, action, reward, new_current_states, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                # compute target Q value for index-th instance in minibatch
                new_q = reward + DISCOUNT * max_future_q
            else:
                reward = 0 # temp; try out see if it improves results
                new_q = reward

            # get current Qs from index-th instance in minibatch
            current_qs = current_qs_list[index]
            # superimpose max future Q update value onto that action's Q val in Qs list (update current qs)
            current_qs[action] = new_q

            # instead of passing image, pass state tuple
            X.append(current_state)
            y.append(current_qs)
            
        self.model.fit(np.array(X), np.array(y), batch_size = MINIBATCH_SIZE,
        verbose=0, shuffle=False if terminal_state else None)

        # updating to determine if we want to update target_model yet
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
            
        
        
agent = DQNAgent()

for episode in tqdm(range(1, EPISODES+1), ascii=True, unit="episode"):
    # ascii=True is for windows users

    agent.tensorboard.step = episode

    episode_reward = 0
    step = 1
    current_state = env.reset()

    done = False

    # Run the episode
    while not done:
        if np.random.random() > epsilon: # exploit
            action = np.argmax(agent.get_qs(current_state))
        else:                            # explore
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)

        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                       epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if max_reward > MIN_REWARD: # only save if at least 1 episode in batch of 50 has a win
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        # let epsilon reset for 4 runs through this
        epsilon = (5000 - episode) / 5000 # we can change EPISODES value NOW
        if epsilon < 0: epsilon += 1
        if epsilon < 0: epsilon += 1
        if epsilon < 0: epsilon += 1
        #epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

env.close()



























