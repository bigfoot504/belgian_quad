'''
Generates some blobs and lets them float around randomly.
Blobs are intended to simulate UxVs.
This script is meant to build out a Deep Q-learning algorithm to accomplish an objective.
First scenario, try to make one drone fly to food.
'''

import numpy as np
import keras.backend.tensorflow_backend as backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2

#import pygame

STARTING_BLUE_BLOBS = 10
STARTING_RED_BLOBS = 10

WIDTH = 800
HEIGHT = 600
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)


# Blob Environment class
# Taken from dqn-part.py, but in this scenario, the learning will not be picture-based
# ***need to modify
class BlobEnv:
    SIZE = 10
    #RETURN_IMAGES = True
    MOVE_PENALTY = 1
    ENEMY_PENALTY = 300
    FOOD_REWARD = 25
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    ACTION_SPACE_SIZE = 9
    PLAYER_N = 1 # player key in dict
    FOOD_N = 2   # food key in dict
    ENEMY_N = 3  # enemy key in dict
    # the dict! (colors)
    d = {1: (255, 175, 0), # orange
         2: (0, 255, 0),   # green
         3: (0, 0, 255)}   # blue

    def reset(self):
        self.player = Blob(self.SIZE)
        self.food = Blob(self.SIZE)
        while self.food == self.player:
            self.food = Blob(self.SIZE)
        self.enemy = Blob(self.SIZE)
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Blob(self.SIZE)

        self.episode_step = 0

        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:
            observation = (self.player-self.food) + (self.player-self.enemy)
        return observation

    # ***need to modify
    def step(self, action):
        self.episode_step += 1
        self.player.action(action)

        #### MAYBE ###
        # self.enemy.move()
        # self.food.move()
        ##############

        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
        else:
            new_observation = (self.player-self.food) + (self.player-self.enemy)

        if self.player == self.enemy:
            reward = -self.ENEMY_PENALTY
        elif self.player == self.food:
            reward = self.FOOD_REWARD
        else:
            reward = -self.MOVE_PENALTY

        done = False
        if reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY or self.episode_step >= 200:
            done = True

        return new_observation, reward, done

    def render(self): # whether or not we want to render it
        img = self.get_image()
        img = img.resize((300, 300))       # resizing so we can see our agent in all its glory.
        cv2.imshow("image", np.array(img)) # show it!
        cv2.waitKey(1)

    # FOR CNN #
    def get_image(self):  # so we can pull an exact image from our env
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]  # sets the food location tile to green color
        env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]  # sets the enemy location to red
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img
        # this CNN will use image as input (vs before used delta to food or enemy as input)


env = BlobEnv()

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

# Memory fraction, used mostly when training multiple agents on same machine
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))


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


# Agent class
class DQNAgent:
    def __init__(self):

        # main model; gets trained every step
        self.model = self.create_model()

        # Target model; this is what we predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{MODEL_NAME}-{int(time.time())}")
        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(256, (3, 3), input_shape=env.OBSERVATION_SPACE_VALUES))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))

        model.add(Dense(env.ACTION_SPACE_SIZE, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
        # transition is observation space, action, reward, new obs space, whether it was done

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]
        # * unpacks state

    def train(self, terminal_state, step):

        # first, check: should we train?
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []  # images from the game
        y = []  # labels/targets

        for index, (current_states, action, reward, new_current_states, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X) / 255, np.array(y), batch_size=MINIBATCH_SIZE,
                       verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # updating to determine if we want to update target_model yet
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


# Blob parent class
class Blob:

    # Blob is born (when a Blob is created, this is going to run)
    def __init__(self, color, x_boundary, y_boundary, size_range=(4, 8), movement_range=(-1, 1)):
        # size_range & movement_range can be specified or go to those defaults
        self.size = random.randint(size_range[0], size_range[1])
        self.color = color  # assigns color attribute to self object
        self.x_boundary = x_boundary  # from window WIDTH
        self.y_boundary = y_boundary  # from window HEIGHT
        self.x = random.randrange(0, self.x_boundary)  # like randint(0,WIDTH-1)
        self.y = random.randrange(0, self.y_boundary)
        '''
        if color == BLUE: # assign locations based on color
            self.x = random.randrange(0, self.x_boundary / 4)
            self.y = random.randrange(0, self.y_boundary)
        elif color == RED:
            self.x = random.randrange(self.x_boundary*3/4, self.x_boundary)
            self.y = random.randrange(0, self.y_boundary)
        '''
        self.movement_range = movement_range

    def __str__(self):  # return position
        return f"Blob ({self.x}, {self.y})"

    def __sub__(self, other):  # subtract one blob from another (get (x,y) distance)
        return (self.x - other.x, self.y - other.y)

    def __eq__(self, other):  # check if two blobs over each other
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        '''
        Gives us 9 total movement options. (0,1,2,3,4,5,6,7,8)
        '''
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)
        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=-1, y=0)
        elif choice == 6:
            self.move(x=0, y=1)
        elif choice == 7:
            self.move(x=0, y=-1)
        elif choice == 8:
            self.move(x=0, y=0)

    def move(self, move_x=False, move_y=False):
        # If no value for x, move randomly
        if not move_x:
            self.x += random.randint(self.movement_range[0], self.movement_range[1])
        else:
            self.x += move_x

        # If no value for y, move randomly
        if not move_y:
            self.y += random.randint(self.movement_range[0], self.movement_range[1])
        else:
            self.y += move_y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.x_boundary - 1:
            self.x = self.x_boundary - 1
        if self.y < 0:
            self.y = 0
        elif self.y > self.y_boundary - 1:
            self.y = self.y_boundary - 1

    def move_drift(self, drift_dir, speed=1):
        # specify a random drift direction
        # drift_dir in degrees 0-360
        move_x = int(round(np.sin(drift_dir * np.pi / 180))) + random.randint(-1, 1) * speed
        move_y = -int(round(np.cos(drift_dir * np.pi / 180))) + random.randint(-1, 1) * speed
        self.move(move_x, move_y)

    # End of Blob class


# BlueBlob class inherits from parent Blob class
class BlueBlob(Blob):

    def __init__(self, color, x_boundary, y_boundary, size_range=(4,8), movement_range=(-1,1)):
        # Overwrites __init__, but keep default values here so they can be specified when super init is called
        super().__init__(color, x_boundary, y_boundary, size_range, movement_range)
        self.color = BLUE
        # Overwrites x,y coord's specified by super init called above
        self.x = random.randrange(0, self.x_boundary / 4)
        self.y = random.randrange(0, self.y_boundary)

    def move_drift(self, drift_dir, speed=1):
        # drift randomly in a direction
        # drift_dir in degrees 0-360
        super().move_drift(drift_dir, speed)


# RedBlob class inherits from parent Blob class
class RedBlob(Blob):

    def __init__(self, color, x_boundary, y_boundary, size_range=(4,8), movement_range=(-1,1)):
        # Overwrites __init__, but keep default values here so they can be specified when super init is called
        super().__init__(color, x_boundary, y_boundary, size_range, movement_range)
        self.color = RED
        # Overwrites x,y coord's specified by super init called above
        self.x = random.randrange(self.x_boundary * 3 / 4, self.x_boundary)
        self.y = random.randrange(0, self.y_boundary)

    def move_drift(self, drift_dir, speed=1):
        # drift randomly in a direction
        # drift_dir in degrees 0-360
        super().move_drift(drift_dir, speed)


agent = DQNAgent()

for episode in tqdm(range(1, EPISODES+1), ascii=True, unit="episode"):
    agent.tensorboard.step = episode

    episode_reward = 0
    step = 1
    current_state = env.reset()

    done = False

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        new_state, reward, done = env.step(action)

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
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)


'''
game_display = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("UxV Blob World")
clock = pygame.time.Clock()

def draw_environment(blob_list, t):
    game_display.fill(WHITE) # clears out frame, so we can redraw new frame on top
    
    for blob_dict in blob_list:
    # there's a blob_dict for blue blobs and one for red blobs
    # blob_list consists of 2 things: the blue dict and the red dict
        for blob_id in blob_dict:
            blob = blob_dict[blob_id]
            pygame.draw.circle(game_display, blob.color, [blob.x, blob.y], blob.size)

            blob.move_drift(t,1)

    pygame.display.update()  # updates our screen; backend builds screen; update sends build to the screen
    
def main():
    blue_blobs = dict(enumerate([BlueBlob(BLUE,WIDTH,HEIGHT,(8,8)) for i in range(STARTING_BLUE_BLOBS)]))
    red_blobs = dict(enumerate([RedBlob(RED,WIDTH,HEIGHT,(8,8)) for i in range(STARTING_RED_BLOBS)]))
    t = 0 # use t for time construct
    while True:
        for event in pygame.event.get(): # grabs event from pygame's events
            if event.type == pygame.QUIT: # pygame QUIT event (like clicking "X" in corner)
                pygame.quit()
                quit()

        t += 2  # move the needle so that blobs go in circles
        draw_environment([blue_blobs,red_blobs], t)
        clock.tick(200) # 60fps cap
        #print(red_blob.x, red_blob.y)

if __name__ == '__main__':
    main()
'''
