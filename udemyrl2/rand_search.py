# like random_search.py

import gym
import numpy as np
import matplotlib.pyplot as plt

# returns 1 if dot prod is positive, 0 o/w
def get_action(s, w):
    return 1 if s.dot(w) > 0 else 0


# plays an episode choosing a random action
def play_one_episode(env, params):
    observation = env.reset() # reset env to begin new episode
    done = False
    t = 0                     # track length of episode

    while not done and  t < 10_000:
        env.render()
        t += 1                                             # index time
        action = get_action(observation, params)           # choose action
        observation, reward, done, info = env.step(action) # perform action
        # note we are ignoring the rewards here
        if done:
            break

    return t


# purpose is to keep track of multiple episode lengths and then return the avg
def play_multiple_episodes(env, T, params):
    # T is number of times to play
    episode_lengths = np.empty(T)

    for i in range(T):
        episode_lengths[i] = play_one_episode(env, params)

    avg_length = episode_lengths.mean()
    print("avg length:", avg_length)
    return avg_length


def random_search(env):
    episode_lengths = [] # list to be filled with lengths of the episodes
    best = 0
    params = None
    for t in range(100):
        new_params = np.random.random(4)*2 - 1 # 4 r.v.'s ~U(-1,1)
        # play 100 and get avg length
        avg_length = play_multiple_episodes(env, 100, new_params)
        # store avg length in list
        episode_lengths.append(avg_length)

        # keep rand params if best, o/w will make new ones
        if avg_length > best:
            params = new_params
            best = avg_length
    return episode_lengths, params


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    episode_lengths, params = random_search(env)
    plt.plot(episode_lengths)
    plt.show()

    # play a final set of episodes
    print("***Final run with final weights***")
    play_multiple_episodes(env, 100, params)

'''
env = gym.make('CartPole-v0')

env.reset()

box = env.observation_space

# generate some random actions for cartpole
NUM_WT_ADJUST = 50
NUM_EPISODES = 100 # per wt adjust
for _ in range(NUM_WT_ADJUST):
    new_wts = random.random()
    for _ in range(NUM_EPISODES):
        observation, reward, done, _ = env.step(env.action_space.sample())
    if avg_ep_len > best_so_far:


done = False
i = 0
while not done:
    i += 1
    observation, reward, done, _ = env.step(env.action_space.sample())
print(i)
'''
