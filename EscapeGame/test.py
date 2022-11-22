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
from train import StateProcessor, Estimator, make_epsilon_greedy_policy, Mission_Loader


malmoutils.fix_print()
actionSet = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]
# ===========================================================================================================================
def show_success_rate(maze_num, success_rate):
    try:
        bars.remove(bars[0])
    except Exception:
        pass

    maze_num = [i+1 for i in range(maze_num)]
    with open('./success_test/success_rate.png', 'wb') as f:
        color = ['red','black','peru','orchid','deepskyblue']
        x_label = ['Maze{}'.format(i-1) for i in maze_num]
        plt.xticks(maze_num, x_label)   # x label
        bars = plt.bar(maze_num, success_rate, color=color)    # y label
        plt.title('success_rate')
        plt.grid(True, linestyle=':', color='r', alpha=0.6)
        plt.savefig(f)

def test(agent_host, q_estimator, epsilon, save_file):
    # total test file number
    num_episodes = 1000

    # load the model
    q_estimator.load_model(save_file)

    # define the mission_loader and state_processor
    mission_loader, state_processor = Mission_Loader(), StateProcessor()
    policy = make_epsilon_greedy_policy(q_estimator, len(actionSet))
    # client
    my_clients = MalmoPython.ClientPool()
    my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000))

    max_retries = 3
    agentID = 0
    expID = 'Deep_q_learning memory'

    # some eval params
    maze_num = 120
    success_num = [0]*maze_num
    run_num = [0]*maze_num
    success_rate = [0.0]*maze_num
    currentMission_num = 0

    if not os.path.exists('./success_test'):
        os.makedirs('./success_test')
    
    for i_episode in range(num_episodes):
        print("{}-th episode".format(i_episode + 1))
        # load the file
        mission_file = agent_host.getStringArgument('mission_file')
        if i_episode != 0:
            run_num[currentMission_num] += 1
            success_rate[currentMission_num] = success_num[currentMission_num] / run_num[currentMission_num]
            show_success_rate(maze_num, success_rate)
        currentMission_num = randint(0, 120)
        currentMission = mission_file = os.path.join(mission_file, 'maze%s.xml' % currentMission_num)
#         else:
#             mission_file = currentMission
        my_mission = mission_loader.mission_load(mission_file=mission_file)
        my_mission.forceWorldReset()
        my_mission.setViewpoint(2)
        my_clients = MalmoPython.ClientPool()
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000))  # add Minecraft machines here as available

        my_mission_record = malmoutils.get_default_recording_object(agent_host, 'save_%s-rep%d' % (expID, i_episode))

        for retry in range(max_retries):
            try:
                agent_host.startMission(my_mission, my_clients, my_mission_record, agentID, "%s-%d" % (expID, i_episode))
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2.5)
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
        # get current state
        while world_state.is_mission_running and all(e.text == '{}' for e in world_state.observations):
            print("sleeping ...")
            world_state = agent_host.peekWorldState()
        state = state_processor.gridProcess(world_state)
        state = state_processor.squeeze_dim(state)
        state = state_processor.change2feature(state)
        done = False
        for t in itertools.count():
            action_probs = policy(torch.from_numpy(state.astype('float32')), epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            print("Step {} @ Episode {}/{}, Action: {}".format(t, i_episode+1, num_episodes, actionSet[action]), end="")
            agent_host.sendCommand(actionSet[action])

            world_state = agent_host.peekWorldState()

            num_frames_seen = world_state.number_of_video_frames_since_last_state

            # 观察是否完成了状态转移/执行完动作
            while world_state.is_mission_running and world_state.number_of_video_frames_since_last_state == num_frames_seen:
                world_state = agent_host.peekWorldState()
            # done 获取方法：判断是否mission还在执行中
            done = not world_state.is_mission_running

            # mission not over
            if world_state.is_mission_running:
                while world_state.number_of_rewards_since_last_state <= 0:
                    time.sleep(0.1)
                    world_state = agent_host.peekWorldState()
                reward = world_state.rewards[-1].getValue()
                print(",Reward: %s" % (reward))
                while world_state.is_mission_running and all(e.text == '{}' for e in world_state.observations):
                    world_state = agent_host.peekWorldState()
                
                if world_state.is_mission_running:
                    state = state_processor.gridProcess(world_state)
                    state = state_processor.squeeze_dim(state)
                    state = state_processor.change2feature(state)
                else:
                    print("Mission finished prematurely")
                    # state = state
                    done = not world_state.is_mission_running
            
            if done:
                # get reward
                if len(world_state.rewards)>0:
                    reward = world_state.rewards[-1].getValue()
                    if int(reward) >= 200:
                        success_num[currentMission_num] += 1
                else:
                    print("IDK no reward")
                    reward = 0
                print("Last action: just received the reward: %s on action: %s " % (reward, actionSet[action]))

                # state = state
                print('End of Episode')
                break

if __name__ == '__main__':
    agent_host = MalmoPython.AgentHost()
    # Find the default mission file by looking next to the schemas folder:
    schema_dir = None
    try:
        schema_dir = "maze"
    except KeyError:
        print("MALMO_XSD_PATH not set? Check environment.")
        exit(1)
    mission_file = os.path.abspath(schema_dir)  # Integration test path
    # print('here is mission_file_name', mission_file)
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

    q_estimator = Estimator(name="q")
    save_file = './save/train_pytorch_q_net(3).pt'
    epsilon = 0.2
    test(agent_host=agent_host, q_estimator=q_estimator, epsilon=epsilon, save_file=save_file)