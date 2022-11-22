from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
from future import standard_library
standard_library.install_aliases()

from builtins import input
from builtins import range
from builtins import object
import itertools
try:
    import MalmoPython
except ImportError:
    import malmo.MalmoPython as MalmoPython

import json
import logging
import math
import os
import random
import sys
import time
import malmoutils
import matplotlib.pyplot as plt
import plotting_1
from collections import deque, namedtuple
import numpy as np
from random import randint
# TODO:调整好maze地图
malmoutils.fix_print()
actionSet = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]
size = 0
# ===========================================================================================================================
class StateProcessor():
    """
    Process a grid map state.
    """
    def __init__(self):
        pass

    def gridProcess(self, world_state):
        msg = world_state.observations[-1].text
        observations = json.loads(msg)
        grid = observations.get(u'floor13x13', 0)
        # print("observations______", observations)
        # Xpos = observations.get(u'XPos', 0)
        # Zpos = observations.get(u'ZPos', 0)
        obs = np.array(grid)
        # print(grid)
        obs = np.reshape(obs, [13, 13, 1])
        # if int(5+Zpos) > 12 or int(Xpos+5) > 12:
        #     print("Zpos", Zpos)
        #     print("Xpos", Xpos)
        # obs[6][6] = "human"
        # print((int)(5+ Zpos), (int)(Xpos+5))
        # print(obs)
        obs[obs == "air"] = 0
        obs[obs == "beacon"] = 1
        obs[obs == "glass"] = 1
        obs[obs == "carpet"] = 2
        obs[obs == "fire"] = 3
        obs[obs == "netherrack"] = 3
        obs[obs == "sea_lantern"] = 4
        obs[obs == "emerald_block"] = 5
        # obs[obs == "grass"] = 5
        # obs[obs == "human"] = 6
 
        return obs
    
    def squeeze_dim(self, state_to_be_squeezed):
        """
        Args:
            state_to_be_squeezed: [size, size, 1]
        Returns:
            state_squeezed: [size, size]
        """
        return np.squeeze(state_to_be_squeezed)
    
    def change2feature(self, state_to_be_changed):
        """
        Args:
            state_to_be_changed: [size, size]
        Returns:
            state_featured: [N_feature, size, size] # N_feature == 7
        """
        ft1 = np.where(state_to_be_changed=='0', 1, 0)
        ft2 = np.where(state_to_be_changed=='1', 1, 0)
        ft3 = np.where(state_to_be_changed=='2', 1, 0)
        ft4 = np.where(state_to_be_changed=="3", 1, 0)
        ft5 = np.where(state_to_be_changed=="4", 1, 0)
        ft6 = np.where(state_to_be_changed=="5", 1, 0)
        # ft7 = np.where(state_to_be_changed=="6", 1, 0)
        # ft7 = np.zeros(shape=(13, 13))
        # ft7[6][6] = 1   # 表示这里是人
        # ft8 = np.where(state_to_be_changed=="7", 1, 0)
        state_featured = np.array([ft1, ft2, ft3, ft4, ft5, ft6])
        return state_featured


class Estimator(nn.Module): # TODO:之后单独放在一个文件里
    def __init__(self, name, save_dir=None):
        super().__init__()
        self.name = name
        self.save_dir = save_dir
        # define the layers and maybe the conv layers are not so good
        self.conv1 = nn.Conv2d(6, 25, 6, 1)
        self.conv2 = nn.Conv2d(32, 16, 3, 1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(25*8*8, 200) # (1024, 512)
        self.fc2 = nn.Linear(200, 64)  # (512, 256)
        self.fc3 = nn.Linear(64, len(actionSet)) # (64, len(actionSet))
        # loss
        self.loss = nn.SmoothL1Loss()
        # optimizer
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=0.01)
        self.fc1.weight.data.normal_(0, 0.1)   # 原来是0, 0.1
        self.fc2.weight.data.normal_(0, 0.1)   # 原来是0, 0.1
        self.fc3.weight.data.normal_(0, 0.1)   # 原来是0, 0.1

    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

    def save_model(self, pt_file):
        torch.save(self, f=pt_file)

    def load_model(self, pt_file):
        self = torch.load(pt_file)
        self.eval()


def copy_model_parameters(from_model, to_model):
    """
    Copies the model parameters of one estimator to another.
    Args:
      from_model: Estimator to copy the paramters from
      to_model: Estimator to copy the parameters to
    """
    to_model.load_state_dict(from_model.state_dict())

def make_epsilon_greedy_policy(estimator, nA):
    """
        Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
        Args:
            estimator: An estimator that returns q values for a given state
            nA: Number of actions in the environment.
        Returns:
            A function that takes the (sess, state, epsilon) as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.
    """
    def policy_fn(state, epsilon):
        """
        Args:
            state: shape[4, size, size]
            epsilon: decaying with step_num
        """
        A = [0] * len(actionSet)
        # 第一步，找出最大的q_value预测值的action
        approx_q_values = estimator.forward(torch.unsqueeze(state, 0)).detach().numpy()
        max_a = np.argmax(approx_q_values)
        # 第二步，概率赋值
        ordinary_prob = epsilon / nA
        for i in range(len(actionSet)):
            A[i] = ordinary_prob
        
        A[max_a] += 1.0 - epsilon
        return A # 数组 1 * 4
    
    return policy_fn

class Mission_Loader():
    def __init__(self) -> None:
        pass
    
    def mission_load(self, mission_file):
        with open(mission_file, 'r') as f:
            print("Loading mission from %s" % mission_file)
            mission_xml = f.read()
            my_mission = MalmoPython.MissionSpec(mission_xml, True)
        my_mission.removeAllCommandHandlers()
        my_mission.allowAllDiscreteMovementCommands()
        my_mission.setViewpoint(2)
        return my_mission

    def start_mission(self, my_mission, agent_host, max_retries, my_clients, expID, my_mission_record, agentID):
        for retry in range(max_retries):
            try:
                agent_host.startMission(my_mission, my_clients, my_mission_record, agentID, "%s" % (expID))
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2.5)
    
    def mission_load_by_num(self, mission_file):
        mazeNum = randint(0, 4)
        mission_file = os.path.join(mission_file, "maze%s.xml" % mazeNum)
        currentMission = mission_file
        return currentMission
    
    def get_world_state(self, agent_host):
        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            print(".", end="")
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
        agent_host.sendCommand("look -1")
        agent_host.sendCommand("look -1")
        while world_state.is_mission_running and all(e.text == '{}' for e in world_state.observations):
            world_state = agent_host.peekWorldState()
        return world_state

def show_train_loss(mean_loss, episode_num):
    try:
        train_loss_lines.remove(train_loss_lines[0])
    except Exception:
        pass
    train_loss_lines = plt.plot(episode_num, mean_loss, 'r', lw = 2)
    plt.title('train_loss')
    plt.xlabel('num_episode')
    plt.ylabel('loss')
    plt.legend(['train_loss'])
    if os.path.exists('./train_loss_img/episode_{}.jpg'.format(episode_num[-1]-1)):
        os.remove('./train_loss_img/episode_{}.jpg'.format(episode_num[-1]-1))
    with open(os.path.join('.', 'train_loss_img', 'episode_{}.jpg'.format(episode_num[-1])), 'wb') as f:
        plt.savefig(f) ## 保存图片

def show_success_rate(success_rate, episode_num_20):
    try:
        success_rate_lines.remove(success_rate_lines[0])
    except Exception:
        pass
    success_rate_lines = plt.plot(episode_num_20, success_rate, 'b', lw=2)
    plt.title('success_rate')
    plt.xlabel('num_episode')
    plt.ylabel('success_rate')
    plt.legend(['success_rate'])
    if os.path.exists('./success_rate/episode_{}.jpg'.format(episode_num_20[-1]-20)):
        os.remove('./success_rate/episode_{}.jpg'.format(episode_num_20[-1]-20))
    with open(os.path.join('.', 'success_rate', 'episode_{}.jpg'.format(episode_num_20[-1])), 'wb') as f:
        plt.savefig(f)

def deep_q_learning(agent_host,
                    q_estimator,
                    target_estimator,
                    state_processor,
                    num_episodes,
                    experiment_dir,
                    replay_memory_size=50000,
                    replay_memory_init_size=5000,
                    update_target_estimator_every=1000,
                    discount_factor=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=8000,
                    batch_size=32,
                    record_video_every=100):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.
    Args:
        q_estimator: Estimator object used for the q values
        target_estimator: Estimator object used for the targets
        state_processor: A StateProcessor object
        num_episodes: Number of episodes to run for
        experiment_dir: Directory to save Tensorflow summaries in
        replay_memory_size: Size of the replay memory
        replay_memory_init_size: Number of random experiences to sample when initializing
          the reply memory.
        update_target_estimator_every: Copy parameters from the Q estimator to the
          target estimator every N steps
        discount_factor: Gamma discount factor
        epsilon_start: Chance to sample a random action when taking an action.
          Epsilon is decayed over time and this is the start value
        epsilon_end: The final minimum value of epsilon after decaying is done
        epsilon_decay_steps: Number of steps to decay epsilon over
        batch_size: Size of batches to sample from the replay memory
        record_video_every: Record a video every N episodes
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    mission_loader = Mission_Loader()

    mission_file = agent_host.getStringArgument('mission_file')
    mission_file = os.path.join(mission_file, "maze0.xml")
    currentMission = mission_file
    my_mission = mission_loader.mission_load(mission_file=mission_file)

    my_clients = MalmoPython.ClientPool()
    my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000))  # add Minecraft machines here as available

    max_retries = 3
    agentID = 0
    total_t = 0     # 在一个episode中的total_t
    expID = 'Deep_q_learning memory'

    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    # The replay memory
    replay_memory = []

    # Keeps track of useful statistics
    stats = plotting_1.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    
    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    monitor_path = os.path.join(experiment_dir, "monitor")
    train_loss_dir = os.path.join('.', 'train_loss_img')
    success_rate_dir = os.path.join('.', 'success_rate')

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)
    if not os.path.exists(train_loss_dir):
        os.makedirs(train_loss_dir)
    if not os.path.exists(success_rate_dir):
        os.makedirs(success_rate_dir)
    # Create save_dir
    save_dir = os.path.abspath("./save")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_file = os.path.join(save_dir, 'train_pytorch_q_net(2).pt')


    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # The policy we're following
    policy = make_epsilon_greedy_policy(
        q_estimator,
        len(actionSet))

    my_mission_record = malmoutils.get_default_recording_object(agent_host,
                                                                "save_%s-rep" % (expID))
    # start mission
    mission_loader.start_mission(my_mission=my_mission, agent_host=agent_host, 
                                 max_retries=max_retries, my_clients=my_clients,
                                 expID=expID, my_mission_record=my_mission_record,
                                 agentID=agentID)

    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        print("Sleeping")
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
    print()
    agent_host.sendCommand("look -1")
    agent_host.sendCommand("look -1")
    print("Populating replay memory...")

    while world_state.is_mission_running and all(e.text == '{}' for e in world_state.observations):
        print("Sleeping....")
        world_state = agent_host.peekWorldState()

    # Populate the replay memory with initial experience
    
    while world_state.number_of_observations_since_last_state <= 0 and world_state.is_mission_running:
        # print("Sleeping")
        time.sleep(0.1)
        world_state = agent_host.peekWorldState()
    
    state = state_processor.gridProcess(world_state)  # MALMO ENVIRONMENT Grid world NEEDED HERE/ was env.reset()
    state = state_processor.squeeze_dim(state)
    state = state_processor.change2feature(state)

    ###############################################################################
    # 预先填充replay容器
    ###############################################################################
    for i in range(replay_memory_init_size):
        print("%s th replay memory" % i)
        mission_file = agent_host.getStringArgument('mission_file')
        if i % 20 == 0:
            currentMission = mission_file = mission_loader.mission_load_by_num(mission_file)
        else:
            mission_file = currentMission
        
        print("Mission File:", mission_file)

        # 生成动作概率、选择动作、执行动作
        action_probs = policy(torch.from_numpy(state.astype('float32')), epsilons[min(total_t, epsilon_decay_steps - 1)])
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        agent_host.sendCommand(actionSet[action])

        # 观察执行后状态
        world_state = agent_host.peekWorldState()
        num_frames_seen = world_state.number_of_video_frames_since_last_state

        # 观察是否完成了状态转移/执行完动作
        while world_state.is_mission_running and world_state.number_of_video_frames_since_last_state == num_frames_seen:
            world_state = agent_host.peekWorldState()
        # done 获取方法：判断是否mission还在执行中
        done = not world_state.is_mission_running

        # 如果mission未结束
        if world_state.is_mission_running:
            # Getting the reward from taking a step, 重新判断world_state
            while world_state.number_of_rewards_since_last_state <= 0:
                time.sleep(0.1)
                world_state = agent_host.peekWorldState()
            # reward~!!
            reward = world_state.rewards[-1].getValue()
            print("1)Just received the reward: %s on action: %s " % (reward, actionSet[action]))

            while world_state.is_mission_running and all(e.text == '{}' for e in world_state.observations):
                world_state = agent_host.peekWorldState()

            # 如果已经done了
            if not world_state.is_mission_running:
                # next_state
                next_state = state
                # done
                done = not world_state.is_mission_running
                print("1)Action: %s, Reward: %s, Done: %s" % (actionSet[action], reward, done))
                # add replay_memory
                replay_memory.append(Transition(state, action, reward, next_state, done))
                
                # restart mission for next round of memory generation
                my_mission = mission_loader.mission_load(mission_file=mission_file)
                mission_loader.start_mission(my_mission=my_mission, agent_host=agent_host,
                                             max_retries=max_retries, my_clients=my_clients,
                                             expID=expID, my_mission_record=my_mission_record,
                                             agentID=agentID)
                
                # get new world_state for next new start state
                world_state = mission_loader.get_world_state(agent_host=agent_host)
                state = state_processor.gridProcess(world_state)  # Malmo GetworldState? / env.reset()
                state = state_processor.squeeze_dim(state)
                state = state_processor.change2feature(state)
                # state = np.stack([state] * 4, axis=0)   # shape of [4, size, size]
                # state = np.expand_dims(state, 0)

            # if not done
            else:
                # next_state
                next_state = state_processor.gridProcess(world_state)
                next_state = state_processor.squeeze_dim(next_state)
                next_state = state_processor.change2feature(next_state)
                # next_state = np.append(state[1:, :, :], np.expand_dims(next_state, 0), axis=0)
                # next_state = np.expand_dims(next_state, 0)
                # done
                done = not world_state.is_mission_running
                print("1)Action: %s, Reward: %s, Done: %s" % (actionSet[action], reward, done))
                # add replay_memory
                replay_memory.append(Transition(state, action, reward, next_state, done))

                # update state
                state = next_state
        
        # 如果mission结束了
        else:
            # reward
            if len(world_state.rewards) > 0:
                reward = world_state.rewards[-1].getValue()
            else:
                reward = 0

            # next_state
            next_state = state

            # done
            done = not world_state.is_mission_running

            print("2)Just received the reward: %s on action: %s " % (reward, actionSet[action]))
            print("2)Action: %s, Reward: %s, Done: %s" % (actionSet[action], reward, done))

            # 加入经验值
            replay_memory.append(Transition(state, action, reward, next_state, done))


            # restart mission for next round of memory generation
            my_mission = mission_loader.mission_load(mission_file=mission_file)
            mission_loader.start_mission(my_mission=my_mission, agent_host=agent_host,
                                            max_retries=max_retries, my_clients=my_clients,
                                            expID=expID, my_mission_record=my_mission_record,
                                            agentID=agentID)

            # get new world_state for next new start state
            world_state = mission_loader.get_world_state(agent_host=agent_host)
            state = state_processor.gridProcess(world_state)  # Malmo GetworldState? / env.reset()
            state = state_processor.squeeze_dim(state)
            state = state_processor.change2feature(state)
        total_t += 1
    print("Finished populating memory")

    #######################################################################################
    # 正式训练
    #######################################################################################
    mean_loss = []
    episode_num = []
    currentMission = mission_file
    num_success = 0
    num_20 = 0
    success_rate = []
    for i_episode in range(num_episodes):
        print("%s-th episode" % i_episode)
        episode_num.append(i_episode+1)

        # Start mission if not the first episode
        if i_episode != 0:
            mission_file = agent_host.getStringArgument('mission_file')
            if i_episode%20 == 0:
                num_success = num_20 = 0
                currentMission = mission_file = mission_loader.mission_load_by_num(mission_file)
            else:
                mission_file = currentMission
            my_mission = mission_loader.mission_load(mission_file=mission_file)
            my_mission.forceWorldReset()
            my_mission.setViewpoint(2)
            my_clients = MalmoPython.ClientPool()
            my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000))  # add Minecraft machines here as available

            max_retries = 3
            agentID = 0
            expID = 'Deep_q_learning '
            my_mission_record = malmoutils.get_default_recording_object(agent_host,
                                                                        "save_%s-rep%d" % (expID, i))
            for retry in range(max_retries):
                try:
                    agent_host.startMission(my_mission, my_clients, my_mission_record, agentID, "%s-%d" % (expID, i))
                    break
                except RuntimeError as e:
                    if retry == max_retries - 1:
                        print("Error starting mission:", e)
                        exit(1)
                    else:
                        time.sleep(2.5)
            # restart mission and get world_state
            world_state = agent_host.getWorldState()
            print("Waiting for the mission to start", end=' ')
            while not world_state.has_mission_begun:
                print(".", end="")
                time.sleep(0.1)
                world_state = agent_host.getWorldState()
                for error in world_state.errors:
                    print("Error:", error.text)
        agent_host.sendCommand("look -1")
        agent_host.sendCommand("look -1")
        # 获取初始状态state
        while world_state.is_mission_running and all(e.text == '{}' for e in world_state.observations):
            print("Sleeping....")
            world_state = agent_host.peekWorldState()
        state = state_processor.gridProcess(world_state)  # Malmo GetworldState? / env.reset()
        state = state_processor.squeeze_dim(state)
        state = state_processor.change2feature(state)
            
        # loss初始化
        loss = None
        total_loss = 0

        # One step in the environment
        for t in itertools.count():
            total_t += 1

            # Epsilon for this time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]

            # Maybe update the target estimator
            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(q_estimator, target_estimator)
                print("\nCopied model parameters to target network.")
            
            # Print out which step we're on, useful for debugging.
            print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                t, total_t, i_episode + 1, num_episodes, loss), end="")
            # sys.stdout.flush()
            if loss != None:
                total_loss += loss
            # 生成动作概率、选择动作、执行动作
            action_probs = policy(torch.from_numpy(state.astype('float32')), epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            agent_host.sendCommand(actionSet[action])

            world_state = agent_host.peekWorldState()

            num_frames_seen = world_state.number_of_video_frames_since_last_state
            # 判断是否通过动作变化，成功转移了状态
            while world_state.is_mission_running and world_state.number_of_video_frames_since_last_state == num_frames_seen:
                world_state = agent_host.peekWorldState()

            done = not world_state.is_mission_running
            print(" IS MISSION FINISHED? ", done)


            # 如果还没有停止当前的mission
            if world_state.is_mission_running:
                while world_state.number_of_rewards_since_last_state <= 0:
                    time.sleep(0.1)
                    world_state = agent_host.peekWorldState()
                reward = world_state.rewards[-1].getValue()
                while world_state.is_mission_running and all(e.text == '{}' for e in world_state.observations):
                    world_state = agent_host.peekWorldState()
                # 如果还在运行，则从正在运行的世界环境中读取新状态
                if world_state.is_mission_running:
                    next_state = state_processor.gridProcess(world_state)
                    next_state = state_processor.squeeze_dim(next_state)
                    next_state = state_processor.change2feature(next_state)
                    # next_state = np.append(state[1:, :, :], np.expand_dims(next_state, 0), axis=0)  # TODO:这里为啥是append(state[::1:])
                    # next_state = np.expand_dims(next_state, 0)
                # 如果没有在运行，则当前状态就是下一个状态，同时标记done=True
                else:
                    print("Mission finished prematurely")
                    next_state = state
                    done = not world_state.is_mission_running

                # If our replay memory is full, pop the first element
                if len(replay_memory) == replay_memory_size:
                    replay_memory.pop(0)
                # Save transition to replay memory
                replay_memory.append(Transition(state, action, reward, next_state, done))
                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t

                ##############
                # 训练
                ##############
                samples = random.sample(replay_memory, batch_size)
                states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))
                # all get shape of [batch_size, ?(, ...)]

                q_values_next = q_estimator.forward(torch.from_numpy(next_states_batch.astype('float32'))).detach().numpy()
                # print()
                # print(q_values_next)
                # print()
                best_actions = np.argmax(q_values_next, axis=1)
                # 我们设计target_q值时，是选择的下一个状态的最大value
                # 在DQN中，下一个状态的最大value直接使用target_net来计算并取所有动作对应q值最大者
                # 在DDQN中，为避免估计得到的target值过大，会改为：先用动态更新的q_eval_net来找到下一个状态的最优动作
                #   然后使用target_net来估计当前状态的所有动作对应的q值，然后索引所找到的下一个状态的最优动作对应的target_q值，进而
                # 更新target
                q_values_next_target = target_estimator.forward(torch.from_numpy(next_states_batch.astype('float32')))
                targets_batch = torch.from_numpy(reward_batch.astype('float32')) + torch.from_numpy(np.invert(done_batch).astype(np.float32)) * \
                                discount_factor * q_values_next_target[np.arange(batch_size), best_actions]
                states_batch = np.array(states_batch)
                old_values_batch = q_estimator.forward(torch.from_numpy(states_batch.astype('float32')))[np.arange(batch_size), action_batch]
                # optimizer
                q_estimator.optimizer.zero_grad()
                # loss
                loss = q_estimator.loss(targets_batch, old_values_batch)
                loss.backward()
                q_estimator.optimizer.step()



            if done:
                # get reward
                if len(world_state.rewards)>0:
                    reward = world_state.rewards[-1].getValue()
                    if int(reward) > 200:
                        num_success += 1
                else:
                    print("IDK no reward")
                    reward= 0
                print("Just received the reward: %s on action: %s " % (reward, actionSet[action]))

                # 当前状态即作为下一个状态
                next_state = state

                # 放入memory
                if len(replay_memory) == replay_memory_size:
                    replay_memory.pop(0)
                replay_memory.append(Transition(state, action, reward, next_state, done))

                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t

                samples = random.sample(replay_memory, batch_size)
                states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))
                # 训练！！
                q_values_next = q_estimator.forward(torch.from_numpy(next_states_batch.astype('float32'))).detach().numpy()
                best_actions = np.argmax(q_values_next, axis=1)
                # 我们设计target_q值时，是选择的下一个状态的最大value
                # 在DQN中，下一个状态的最大value直接使用target_net来计算并取所有动作对应q值最大者
                # 在DDQN中，为避免估计得到的target值过大，会改为：先用动态更新的q_eval_net来找到下一个状态的最优动作
                #   然后使用target_net来估计当前状态的所有动作对应的q值，然后索引所找到的下一个状态的最优动作对应的target_q值，进而
                # 更新target
                q_values_next_target = target_estimator.forward(torch.from_numpy(next_states_batch.astype('float32')))
                targets_batch = torch.from_numpy(reward_batch.astype('float32')) + torch.from_numpy(np.invert(done_batch).astype(np.float32)) * \
                                discount_factor * q_values_next_target[np.arange(batch_size), best_actions]
                states_batch = np.array(states_batch)
                old_values_batch = q_estimator.forward(torch.from_numpy(states_batch.astype('float32')))[np.arange(batch_size), action_batch]
                # optimizer
                q_estimator.optimizer.zero_grad()
                # loss
                loss = q_estimator.loss(targets_batch, old_values_batch)
                loss.backward()
                q_estimator.optimizer.step()
                if t != 0:
                    mean_loss.append(float(total_loss / t))
                show_train_loss(mean_loss=mean_loss, episode_num=episode_num)
                num_20 += 1
                print('----------success_rate in episode {0}/20: {1}%----------'.format(num_20, num_success / num_20 * 100))
                print("End of Episode")
                break

        # 训练结束，保存模型
        if i_episode == num_episodes - 1:
            q_estimator.save_model(save_file)

        yield total_t, plotting_1.EpisodeStats(
            episode_lengths=stats.episode_lengths[:i_episode + 1],
            episode_rewards=stats.episode_rewards[:i_episode + 1])


    return stats

# Main body=======================================================
if __name__ == '__main__':
    agent_host = MalmoPython.AgentHost()

    # Find the default mission file by looking next to the schemas folder:
    schema_dir = None
    try:
        # schema_dir = os.environ['mazes']
        schema_dir = "maze"
    except KeyError:
        print("MALMO_XSD_PATH not set? Check environment.")
        exit(1)
    mission_file = os.path.abspath(schema_dir)  # Integration test path
    if not os.path.exists(mission_file):
        print("Could not find Maze.xml under MALMO_XSD_PATH")
        exit(1)

    # add some args
    agent_host.addOptionalStringArgument('mission_file',
                                        'Path/to/file from which to load the mission.', mission_file)
    agent_host.addOptionalFlag('load_model', 'Load initial model from model_file.')
    agent_host.addOptionalStringArgument('model_file', 'Path to the initial model file', '')
    agent_host.addOptionalFlag('debug', 'Turn on debugging.')
    agent_host.setRewardsPolicy(MalmoPython.RewardsPolicy.LATEST_REWARD_ONLY)
    agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
    malmoutils.parse_command_line(agent_host)

    # Where we save our checkpoints and graphs
    experiment_dir = os.path.abspath("./experiments/{}".format("DeepQLearning"))

    # Create estimators
    q_estimator = Estimator(name="q", save_dir=experiment_dir)
    target_estimator = Estimator(name="target_q")

    # State processor
    state_processor = StateProcessor()
    for t, stats in deep_q_learning(agent_host,
                                    q_estimator,
                                    target_estimator,
                                    state_processor,
                                    num_episodes=500,
                                    experiment_dir=experiment_dir,
                                    replay_memory_size=10000,    # 改为2000？
                                    replay_memory_init_size=50,    # 改为500？
                                    update_target_estimator_every=100,
                                    discount_factor=0.99,
                                    epsilon_start=0.2,
                                    epsilon_end=0.1,
                                    epsilon_decay_steps=8000,
                                    batch_size=32,
                                    record_video_every=100):
        # 迭代器返回
        print("Episode {0}'s Reward is: {1}".format(t, stats.episode_rewards[-1]))

    # something to write here
    # ======================================================================================