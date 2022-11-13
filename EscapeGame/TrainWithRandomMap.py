from __future__ import print_function
# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

# The "Cliff Walking" example using Q-learning.
# From pages 148-150 of:
# Richard S. Sutton and Andrews G. Barto
# Reinforcement Learning, An Introduction
# MIT Press, 1998
from future import standard_library

standard_library.install_aliases()
from builtins import input
from builtins import range
from builtins import object
import itertools
import MalmoPython
import json
import logging
import math
import os
import random
import sys
import time
import malmoutils

import plotting
from collections import deque, namedtuple
import numpy as np
from random import randint

# if sys.version_info[0] == 2:
#     # Workaround for https://github.com/PythonCharmers/python-future/issues/262
#     import Tkinter as tk
# else:
#     import tkinter as tk

malmoutils.fix_print()

actionSet = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]

# ===========================================================================================================================


'''

DEEP Learning code comes here

'''


class StateProcessor():
    """
    Processes a raw Atari images. Resizes it and converts it to grayscale.
    """

    def __init__(self):
      


    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [10, 10, 1] Maze RGB State
        Returns:
            A processed [84, 84] state representing grayscale values.
        """
        val = sess.run(self.output, {self.input_state: state})
        # print("!!!!!!!!!!!!!!!!!!!",val)
        return val


def gridProcess(state):
    msg = state.observations[-1].text
    observations = json.loads(msg)
    grid = observations.get(u'floor10x10', 0)
    Xpos = observations.get(u'XPos', 0)
    Zpos = observations.get(u'ZPos', 0)
    obs = np.array(grid)
    obs = np.reshape(obs, [9, 9, 1])
    # obs[(int)(5+ Zpos)][ (int)(5+ Xpos)] = "human"

    # for i in range(obs.shape[0]):
    #     for j in range(obs.shape[1]):
    #         if obs[i,j] ==""
    # print("Zpos, Xpos, ", Zpos, Xpos)
    # print("~~~~~~~~~~~~~~~~~~", obs)
    # exit(0)
    obs[obs == "carpet"] = 0
    obs[obs == "sea_lantern"] = 1
    # obs[obs == "human"] = 3
    obs[obs == "fire"] = 4
    obs[obs == "emerald_block"] = 5
    obs[obs == "beacon"] = 6
    obs[obs == "air"] = 7
    # obs[obs == "grass"] = 8
    # print("Here is obs", obs)
    # exit(0)
    return obs


class Estimator():
    """Q-Value Estimator neural network.
    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, scope="estimator", summaries_dir=None):
        
    def _build_model(self):
      

    def predict(self, sess, s):
        """
        Predicts action values.
        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, 1, 10, 10, 1]
        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated
          action values.
        """
        # print("S's shape:",s.shape)
        return sess.run(self.predictions, {self.X_pl: s})

    def update(self, sess, s, a, y):
        """
        Updates the estimator towards the given targets.
        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, 1, 10, 10, 1]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]
        Returns:
          The calculated loss on the batch.
        """
       


def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.
    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    


def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.
    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    """

    def policy_fn(sess, observation, epsilon):
        

        return A

    return policy_fn


def deep_q_learning(sess,
                    agent_host,
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
        sess: Tensorflow Session object
        env: OpenAI environment
        q_estimator: Estimator object used for the q values
        target_estimator: Estimator object used for the targets
        state_processor: A StateProcessor object
        num_episodes: Number of episodes to run for
        experiment_dir: Directory to save Tensorflow summaries in
        replay_memory_size: Size of the replay memory
        replay_memory_init_size: Number of random experiences to sampel when initializing
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
    mission_file = agent_host.getStringArgument('mission_file')
    mission_file = os.path.join(mission_file, "Maze0.xml")
    currentMission = mission_file
    with open(mission_file, 'r') as f:
        print("Loading mission from %s" % mission_file)
        mission_xml = f.read()
        my_mission = MalmoPython.MissionSpec(mission_xml, True)
    my_mission.removeAllCommandHandlers()
    my_mission.allowAllDiscreteMovementCommands()
    my_mission.setViewpoint(2)
    my_clients = MalmoPython.ClientPool()
    my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000))  # add Minecraft machines here as available
    # my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10001))

    max_retries = 3
    agentID = 0
    expID = 'Deep_q_learning memory'

    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    # The replay memory
    replay_memory = []

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    monitor_path = os.path.join(experiment_dir, "monitor")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)

    saver = 
    # Load a previous checkpoint if we find one
    latest_checkpoint = 

    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    total_t = 

    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # The policy we're following
    policy = make_epsilon_greedy_policy(
        q_estimator,
        len(actionSet))

    my_mission_record = malmoutils.get_default_recording_object(agent_host,
                                                                "save_%s-rep" % (expID))

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

    state = gridProcess(world_state)  # MALMO ENVIRONMENT Grid world NEEDED HERE/ was env.reset()
    state = state_processor.process(sess, state)
    state = np.stack([state] * 4, axis=2)

    for i in range(replay_memory_init_size):
        print("%s th replay memory" % i)
        mission_file = agent_host.getStringArgument('mission_file')
        if i % 20 == 0:
            mazeNum = randint(0, 4)
            mission_file = os.path.join(mission_file, "Maze%s.xml" % mazeNum)
            currentMission = mission_file
        else:
            mission_file = currentMission

        print("Mission File:", mission_file)
        action_probs = policy(sess, state, epsilons[min(total_t, epsilon_decay_steps - 1)])
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        # next_state, reward, done, _ = env.step(actionSet[action]) # Malmo send command for the action
        # print("Sending command: ", actionSet[action])
        agent_host.sendCommand(actionSet[action])

        world_state = agent_host.peekWorldState()

        num_frames_seen = world_state.number_of_video_frames_since_last_state

        while world_state.is_mission_running and world_state.number_of_video_frames_since_last_state == num_frames_seen:
            world_state = agent_host.peekWorldState()

        done = not world_state.is_mission_running

        if world_state.is_mission_running:

            # Getting the reward from taking a step
            while world_state.number_of_rewards_since_last_state <= 0:
                time.sleep(0.1)
                world_state = agent_host.peekWorldState()
            reward = world_state.rewards[-1].getValue()
            print("1)Just received the reward: %s on action: %s " % (reward, actionSet[action]))

            while world_state.is_mission_running and all(e.text == '{}' for e in world_state.observations):
                world_state = agent_host.peekWorldState()
            # world_state = agent_host.getWorldState()

            if not world_state.is_mission_running:
                # reward = 0
                next_state = state
                done = not world_state.is_mission_running
                print("1)Action: %s, Reward: %s, Done: %s" % (actionSet[action], reward, done))
                replay_memory.append(Transition(state, action, reward, next_state, done))
                # restart mission for next round of memory generation

                with open(mission_file, 'r') as f:
                    print("Loading mission from %s" % mission_file)
                    mission_xml = f.read()
                    my_mission = MalmoPython.MissionSpec(mission_xml, True)
                my_mission.removeAllCommandHandlers()
                my_mission.allowAllDiscreteMovementCommands()
                my_mission.setViewpoint(2)

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

                world_state = agent_host.getWorldState()
                while not world_state.has_mission_begun:
                    print(".", end="")
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                agent_host.sendCommand("look -1")
                agent_host.sendCommand("look -1")
                while world_state.is_mission_running and all(e.text == '{}' for e in world_state.observations):
                    world_state = agent_host.peekWorldState()
                state = gridProcess(world_state)  # Malmo GetworldState? / env.reset()
                state = state_processor.process(sess, state)
                state = np.stack([state] * 4, axis=2)

            else:
                next_state = gridProcess(world_state)
                next_state = state_processor.process(sess, next_state)
                next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
                done = not world_state.is_mission_running
                print("1)Action: %s, Reward: %s, Done: %s" % (actionSet[action], reward, done))
                replay_memory.append(Transition(state, action, reward, next_state, done))
                state = next_state

        else:
            done = not world_state.is_mission_running
            if len(world_state.rewards) > 0:
                reward = world_state.rewards[-1].getValue()
            else:
                reward = 0
            print("2)Just received the reward: %s on action: %s " % (reward, actionSet[action]))
            next_state = state
            print("2)Action: %s, Reward: %s, Done: %s" % (actionSet[action], reward, done))
            replay_memory.append(Transition(state, action, reward, next_state, done))

            # restart mission for next round of memory generation

            with open(mission_file, 'r') as f:
                print("Loading mission from %s" % mission_file)
                mission_xml = f.read()
                my_mission = MalmoPython.MissionSpec(mission_xml, True)
            my_mission.removeAllCommandHandlers()
            my_mission.allowAllDiscreteMovementCommands()
            my_mission.setViewpoint(2)

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

            world_state = agent_host.getWorldState()
            while not world_state.has_mission_begun:
                print(".", end="")
                time.sleep(0.1)
                world_state = agent_host.getWorldState()
            agent_host.sendCommand("look -1")
            agent_host.sendCommand("look -1")
            while world_state.is_mission_running and all(e.text == '{}' for e in world_state.observations):
                world_state = agent_host.peekWorldState()

            state = gridProcess(world_state)  # Malmo GetworldState? / env.reset()
            state = state_processor.process(sess, state)
            state = np.stack([state] * 4, axis=2)

        # time.sleep(0.2)
    print("Finished populating memory")

    # Record videos
    # Use the gym env Monitor wrapper
    # env = Monitor(env,
    #               directory=monitor_path,
    #               resume=True,
    #               video_callable=lambda count: count % record_video_every ==0)

    # NEED TO RECORD THE VIDEO AND SAVE TO THE SPECIFIED DIRECTORY
    currentMission = mission_file
    for i_episode in range(num_episodes):
        print("%s-th episode" % i_episode)

        if i_episode != 0:
            mission_file = agent_host.getStringArgument('mission_file')
            if i_episode%20 == 0:
                mazeNum = randint(0, 4)
                mission_file = os.path.join(mission_file,"Maze%s.xml"%mazeNum)
                currentMission = mission_file
            else:
                mission_file = currentMission

            with open(mission_file, 'r') as f:
                print("Loading mission from %s" % mission_file)
                mission_xml = f.read()
                my_mission = MalmoPython.MissionSpec(mission_xml, True)
            my_mission.removeAllCommandHandlers()
            my_mission.allowAllDiscreteMovementCommands()
            # my_mission.requestVideo(320, 240)
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
        # Save the current checkpoint
       

        while world_state.is_mission_running and all(e.text == '{}' for e in world_state.observations):
            world_state = agent_host.peekWorldState()
        # world_state = agent_host.getWorldState()
        state = gridProcess(world_state)  # MalmoGetWorldState?
        state = state_processor.process(sess, state)
        state = np.stack([state] * 4, axis=2)
        loss = None

        # One step in the environment
        for t in itertools.count():

            # Epsilon for this time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]

            # Add epsilon to Tensorboard
            episode_summary = 
            episode_summary.value.add(simple_value=epsilon, tag="epsilon")
            q_estimator.summary_writer.add_summary(episode_summary, total_t)

            # Maybe update the target estimator
            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(sess, q_estimator, target_estimator)
                print("\nCopied model parameters to target network.")

            # Print out which step we're on, useful for debugging.
            print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                t, total_t, i_episode + 1, num_episodes, loss), end="")
            sys.stdout.flush()

            # Take a step
            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            # next_state, reward, done, _ = env.step(actionSet[action]) # Malmo AgentHost send command?
            # print("Sending command: ", actionSet[action])
            agent_host.sendCommand(actionSet[action])

            world_state = agent_host.peekWorldState()

            num_frames_seen = world_state.number_of_video_frames_since_last_state

            while world_state.is_mission_running and world_state.number_of_video_frames_since_last_state == num_frames_seen:
                world_state = agent_host.peekWorldState()

            done = not world_state.is_mission_running
            print(" IS MISSION FINISHED? ", done)
            if world_state.is_mission_running:
                while world_state.number_of_rewards_since_last_state <= 0:
                    time.sleep(0.1)
                    world_state = agent_host.peekWorldState()
                reward = world_state.rewards[-1].getValue()
                print("Just received the reward: %s on action: %s " % (reward, actionSet[action]))

                while world_state.is_mission_running and all(e.text == '{}' for e in world_state.observations):
                    world_state = agent_host.peekWorldState()
                # world_state = agent_host.getWorldState()

                if world_state.is_mission_running:
                    next_state = gridProcess(world_state)
                    next_state = state_processor.process(sess, next_state)
                    next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
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

                # Sample a minibatch from the replay memory
                samples = random.sample(replay_memory, batch_size)
                states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

                # Calculate q values and targets (Double DQN)
                q_values_next = q_estimator.predict(sess, next_states_batch)
                best_actions = np.argmax(q_values_next, axis=1)
                q_values_next_target = target_estimator.predict(sess, next_states_batch)
                targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                                discount_factor * q_values_next_target[np.arange(batch_size), best_actions]

                # Perform gradient descent update
                states_batch = np.array(states_batch)
                loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)
                if done:
                    print("End of episode")
                    break
                state = next_state
                total_t += 1

            if done:
                # while world_state.number_of_rewards_since_last_state <=0:
                #     time.sleep(0.1)
                #     print("Sleeping...zzzz")
                #     world_state = agent_host.peekWorldState()
                if len(world_state.rewards)>0:
                    reward = world_state.rewards[-1].getValue()
                else:
                    print("IDK no reward")
                    reward= 0
                # reward = 0
                print("Just received the reward: %s on action: %s " % (reward, actionSet[action]))

                next_state = state

                replay_memory.append(Transition(state, action, reward, next_state, done))

                if len(replay_memory) == replay_memory_size:
                    replay_memory.pop(0)

                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t

                samples = random.sample(replay_memory, batch_size)
                states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

                # Calculate q values and targets (Double DQN)
                q_values_next = q_estimator.predict(sess, next_states_batch)
                best_actions = np.argmax(q_values_next, axis=1)
                q_values_next_target = target_estimator.predict(sess, next_states_batch)
                targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                                discount_factor * q_values_next_target[np.arange(batch_size), best_actions]

                # Perform gradient descent update
                states_batch = np.array(states_batch)
                loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)

                print("End of Episode")
                break

            # state = next_state
            # total_t += 1

        # Add summaries to tensorboard
        print("Adding to tensorboard summaries !!!!")
        episode_summary = 
        episode_summary.value.add(simple_value=stats.episode_rewards[i_episode], node_name="episode_reward",
                                  tag="episode_reward")
        episode_summary.value.add(simple_value=stats.episode_lengths[i_episode], node_name="episode_length",
                                  tag="episode_length")
        q_estimator.summary_writer.add_summary(episode_summary, total_t)
        q_estimator.summary_writer.flush()

        yield total_t, plotting.EpisodeStats(
            episode_lengths=stats.episode_lengths[:i_episode + 1],
            episode_rewards=stats.episode_rewards[:i_episode + 1])
    # time.sleep(0.2)
    # env.monitor.close()
    return stats


# Main body=======================================================

agent_host = MalmoPython.AgentHost()

# Find the default mission file by looking next to the schemas folder:
schema_dir = None
try:
    # schema_dir = os.environ['mazes']
    schema_dir = "mazes"
except KeyError:
    print("MALMO_XSD_PATH not set? Check environment.")
    exit(1)
mission_file = os.path.abspath(schema_dir)  # Integration test path
# if not os.path.exists(mission_file):
#     mission_file = os.path.abspath(os.path.join(schema_dir, '..',
#                                                 'sample_missions', 'mazes1'))  # Install path
if not os.path.exists(mission_file):
    print("Could not find Maze.xml under MALMO_XSD_PATH")
    exit(1)

# add some args
agent_host.addOptionalStringArgument('mission_file',
                                     'Path/to/file from which to load the mission.', mission_file)
# agent_host.addOptionalFloatArgument('alpha',
#                                     'Learning rate of the Q-learning agent.', 0.1)
# agent_host.addOptionalFloatArgument('epsilon',
#                                     'Exploration rate of the Q-learning agent.', 0.01)
# agent_host.addOptionalFloatArgument('gamma', 'Discount factor.', 0.99)
agent_host.addOptionalFlag('load_model', 'Load initial model from model_file.')
agent_host.addOptionalStringArgument('model_file', 'Path to the initial model file', '')
agent_host.addOptionalFlag('debug', 'Turn on debugging.')
agent_host.setRewardsPolicy(MalmoPython.RewardsPolicy.LATEST_REWARD_ONLY)
agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
malmoutils.parse_command_line(agent_host)


# Where we save our checkpoints and graphs
experiment_dir = os.path.abspath("./experiments/{}".format("DeepQLearning"))

# Create a glboal step variable
global_step = 

# Create estimators
q_estimator = Estimator(scope="q", summaries_dir=experiment_dir)
target_estimator = Estimator(scope="target_q")

# State processor
state_processor = StateProcessor()



# ======================================================================================