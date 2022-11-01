from __future__ import print_function
import os
import sys
import time
import datetime
import json
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import PReLU

# Exploration factor
epsilon = 0.1


class Experience(object):
    def __init__(self, model, max_memory=100, discount=0.95):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = model.output_shape[-1]

    def remember(self, episode):
        # episode = [envstate, action, reward, envstate_next, game_over]
        # memory[i] = episode
        # envstate == flattened 1d maze cells info, including rat cell (see method: observe)


    def predict(self, envstate):


    def get_data(self, data_size=10):
        # envstate 1d size (1st element of episode
 


def qtrain(model, world):



def setupMission():
    mission_file = './farm.xml'
    global my_mission, my_mission_record
    with open(mission_file, 'r') as f:
        print("Loading mission from %s" % mission_file)
        mission_xml = f.read()
        my_mission = MalmoPython.MissionSpec(mission_xml, True)

    my_mission_record = MalmoPython.MissionRecordSpec()


# Attempt to start a mission:
def startMission():
    max_retries = 3
    for retry in range(max_retries):
        try:
            agent_host.startMission(my_mission, my_mission_record)
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:", e)
                exit(1)
            else:
                time.sleep(2)

# Loop until mission starts:


def waitUntilMissionStart():
    print("Waiting for the mission to start ", end=' ')
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)

    print()
    print("Mission running ", end=' ')


def missionLoop(model, world):
    world_state = agent_host.getWorldState()
    my_agent = MyAgent(world_state)
    while world_state.is_mission_running:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        my_agent.updateWorldState(world_state)
        if my_agent.takeAction():
            print(my_agent.takeAction())
            agent_host.sendCommand(my_agent.takeAction())
        for error in world_state.errors:
            print("Error:", error.text)
    print()
    print("Mission ended")
# Mission has ended.


if __name__ == "__main__":
    world = World()
    model = build_model(world.world)
    setupMission()
    startMission()
    waitUntilMissionStart()
    missionLoop(model, world)

if __name__ == "__main__":
    world = World()
    model = build_model(world.world)
    qtrain(model, world)
