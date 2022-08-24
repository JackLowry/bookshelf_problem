from bookshelf_env import Bookshelf
from bookshelf_dqn import DQN, ReplayMemory, Transition

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
from util import estimate_pc_normals
# from pytorch3d.ops import estimate_pointcloud_normals
env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()



BATCH_SIZE = 64
GAMMA = 0.1
EPS_START = 0.5
EPS_END = 0.1
EPS_DECAY = 200
TARGET_UPDATE = 1


bookshelf = Bookshelf()

policy_net = DQN().to(bookshelf.device)
target_net = DQN().to(bookshelf.device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()


steps_done = 0

direction_disc_num = 16



def select_action(state):
    global steps_done
    dir_disc = 16


    points, organized_points = state
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1


    possible_angles = torch.linspace(0, 2*math.pi, dir_disc).to(points.device)
    possible_dirs = torch.stack((torch.cos(possible_angles), torch.sin(possible_angles)), dim=-1)

    normals, local_coord_frame = estimate_pc_normals(organized_points)
    camera_vector = torch.Tensor([[1, 0, 0]]).to(points.device)
    camera_point = torch.Tensor([[0, 0, 1]]).to(points.device).unsqueeze(1).expand(normals.shape[0], normals.shape[1], 3)
    # normals = torch.where((torch.sum(normals*(camera_vector.expand(normals.shape)), -1) >= 0).unsqueeze(-1).expand(normals.shape), normals, -normals)
    
    pos_dist = torch.sum(((points[:, :, 0:3]+normals)-camera_point)**2, dim=-1).unsqueeze(-1).expand(-1, -1, 3)
    neg_dist = torch.sum(((points[:, :, 0:3]-normals)-camera_point)**2, dim=-1).unsqueeze(-1).expand(-1, -1, 3)
    
    normals = torch.where(pos_dist > neg_dist, -normals, normals)
    #
    if sample > eps_threshold:
        with torch.no_grad():
            directions, heatmap = policy_net(points)["preds"]

            # heatmap_on_box = heatmap[points[:, :, -1] > 0]
            heatmap_on_box = heatmap
            best_point_idx = torch.argmax(heatmap_on_box, 1).squeeze()
            best_dir_idx = directions[torch.arange(0, points.shape[0]), best_point_idx, :]
            best_dir_idx = torch.argmax(best_dir_idx, 1)
            action_points = points[torch.arange(0, points.shape[0]), best_point_idx, :3]
            action_normals = normals[torch.arange(0, points.shape[0]), best_point_idx, :]
            action_dirs = possible_dirs[best_dir_idx]
            return (action_points, action_normals, action_dirs), (best_point_idx, best_dir_idx)
    else:
        # box_points = state[state[:, :, -1] > 0][:, :, :3]
        box_points = points[:, :, :3]
        batch_num = box_points.shape[0]
        action_num = box_points.shape[1]
        rand_point_idx = torch.randint(action_num, (batch_num,)).to(points.device)
        action_points = box_points[torch.arange(0, points.shape[0]), rand_point_idx, :]
        action_normals = normals[torch.arange(0, points.shape[0]), rand_point_idx, :]
        rand_dir_idx = torch.randint(0, direction_disc_num, (points.shape[0],)).to(points.device)
        action_dirs = possible_dirs[rand_dir_idx, :]
        return (action_points, action_normals, action_dirs), (rand_point_idx, rand_dir_idx)

LEARNING_RATE = 0.005
optimizer = optim.SGD(policy_net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.03125)
memory = ReplayMemory(10000)

episode_rewards = []
episode_losses = []
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # # Compute a mask of non-final states and concatenate the batch elements
    # # (a final state would've been the one after which simulation ended)
    # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
    #                                       batch.next_state)), device=bookshelf.device, dtype=torch.bool)
    # non_final_next_states = torch.cat([s for s in batch.next_state
    #                                             if s is not None])
    state_batch = torch.stack(batch.state)
    action_points_idx, action_dirs_idx = zip(*batch.action)
    action_points_idx = torch.stack(action_points_idx)
    action_dirs_idx = torch.stack(action_dirs_idx)
    reward_batch = torch.stack(batch.reward)
    next_state_batch = torch.stack(batch.next_state)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_dir_values, state_point_values = policy_net(state_batch)['preds']
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_dir_values, next_state_point_values = target_net(next_state_batch)['preds']
    # Compute the expected Q values
    GAMMA = 0.5
    expected_point_action_values = (torch.amax(next_state_dir_values, dim=(1, 2)) * GAMMA) + reward_batch
    expected_dir_action_values = (torch.amax(next_state_point_values, dim=(1, 2)) * GAMMA) + reward_batch

    #assume gamma is 0, need to compute next value states soon
    
    y_point = torch.clone(state_point_values)
    y_point[torch.arange(0, BATCH_SIZE), action_points_idx, :] = expected_point_action_values.unsqueeze(-1).to(torch.float)
    y_dir = torch.clone(state_dir_values)
    y_dir[torch.arange(0, BATCH_SIZE), action_points_idx, action_dirs_idx] = expected_dir_action_values.to(torch.float)
    # Compute Huber loss
    point_criterion = nn.SmoothL1Loss(reduce=False)
    point_loss = point_criterion(state_point_values, y_point)

    dir_criterion = nn.SmoothL1Loss(reduce=False)
    dir_loss = dir_criterion(state_dir_values, y_dir)

    loss = point_loss + dir_loss
    loss = loss.sum()
    episode_losses.append(loss)
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-1, 1)
    optimizer.step()



def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_reward = torch.tensor(episode_rewards, dtype=torch.float)
    durations_loss = torch.tensor(episode_losses, dtype=torch.float)
    fig, ax = plt.subplots(1, 2, num=2)
    plt.suptitle('Training...')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Episodes')
    ax[0].plot(durations_loss.numpy())
    ax[1].set_ylabel('Reward')
    ax[1].set_xlabel('Episodes')
    ax[1].plot(durations_reward.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_reward) >= 100:
        r_means = durations_reward.unfold(0, 100, 1).mean(1).view(-1)
        r_means = torch.cat((torch.zeros(99), r_means))
        ax[1].plot(r_means.numpy())
        l_means = durations_loss.unfold(0, 100, 1).mean(1).view(-1)
        l_means = torch.cat((torch.zeros(99), l_means))
        ax[0].plot(l_means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

frames = 0
# while not bookshelf.gym.query_viewer_has_closed(bookshelf.viewer):
#     if frames % 200 == 0:
#         bookshelf.reset()
#         state = bookshelf.observe()
#         print(state)
#     bookshelf.step(None)
#     frames = frames + 1
# bookshelf.cleanup()

num_episodes = 2000
for i_episode in range(num_episodes):
    # Initialize the environment and state
    bookshelf.reset()
    state = bookshelf.observe()
    # Select and perform an action
    action, action_idxs = select_action(state)
    rewards = bookshelf.do_actions(action)

    next_state = bookshelf.observe()


    # Store the transition in memory
    for batch_i in range(rewards.shape[0]):
        m_action = (action_idxs[0][batch_i], action_idxs[1][batch_i])
        memory.push(state[0][batch_i], m_action, next_state[0][batch_i], rewards[batch_i])

    # Move to the next state
    state = next_state

    # Perform one step of the optimization (on the policy network)
    optimize_model()
    episode_rewards.append(torch.mean(rewards))
    plot_durations()
    
    print(f"Episode {i_episode}")
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    if i_episode % 10 == 0:
        torch.save(policy_net.state_dict(), f"ckpt/model_{i_episode}")
        plt.savefig("model_stats.png")

    

bookshelf.cleanup()
print('Complete')
plt.ioff()
plt.savefig("test_run.png")
plt.show()