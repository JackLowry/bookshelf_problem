import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

from bookshelf_dqn import DQN, ReplayMemory, Transition
from bookshelf_env import Bookshelf

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

bookshelf = Bookshelf()
bookshelf.reset()
frames = 0
while not bookshelf.gym.query_viewer_has_closed(bookshelf.viewer):
    if frames % 200 == 0:
        bookshelf.reset()
        # bookshelf.observe()
    bookshelf.step(None)
    frames = frames + 1
bookshelf.cleanup()



policy_net = DQN().to(bookshelf.device)
target_net = DQN().to(bookshelf.device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()


steps_done = 0

direction_disc_num = 16


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1


    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            directions, heatmap = policy_net(state)["pred"]

            heatmap_on_box = heatmap[state[:, :, -1] > 0]
            action_points = state[:, torch.argmax(heatmap_on_box, 1), :3]

            action_dirs = directions[:, torch.argmax(action_dirs, 1)]
            actions = torch.cat((action_points, action_dirs), dim=-1)
            return actions
    else:
        box_points = state[state[:, :, -1] > 0][:, :, :3]
        batch_num = box_points.shape[0]
        action_num = box_points.shape[1]
        action_points = box_points[:, torch.randint(action_num, (batch_num,)), :]
        action_dirs = torch.randint(direction_disc_num, (state.shape[0]))
        actions = torch.cat((action_points, action_dirs), dim=-1)
        return actions


optimizer = optim.SGD(policy_net.parameters(), lr=0.01, momentum=0.9)
memory = ReplayMemory(10000)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=bookshelf.device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # # Compute V(s_{t+1}) for all next states.
    # # Expected values of actions for non_final_next_states are computed based
    # # on the "older" target_net; selecting their best reward with max(1)[0].
    # # This is merged based on the mask, such that we'll have either the expected
    # # state value or 0 in case the state was final.
    # next_state_values = torch.zeros(BATCH_SIZE, device=bookshelf.device)
    # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # # Compute the expected Q values
    # expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    #assume gamma is 0, need to compute next value states soon
    gamma = 0
    
    y = reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, y.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

bookshelf = Bookshelf()
frames = 0
while not bookshelf.gym.query_viewer_has_closed(bookshelf.viewer):
    if frames % 200 == 0:
        bookshelf.reset()
        state = bookshelf.observe()
        print(state)
    bookshelf.step(None)
    frames = frames + 1
bookshelf.cleanup()

num_episodes = 50
for i_episode in range(num_episodes):
    # Initialize the environment and state
    bookshelf.reset()
    state = bookshelf.observe()
    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=bookshelf.device)

        next_state = bookshelf.observe()

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()