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

# Tutorial sample #2: Run simple mission using raw XML

from builtins import range
import logging
from farmworld import World
from myagent import Experience
import MalmoPython
import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn

actionMap = {0: 'movenorth 1', 1: 'movesouth 1',
             2: 'moveeast 1', 3: 'movewest 1', 4: 'hotbar.2 1', 5: 'hotbar.1 1', 6: 'tp 0.5 4 0.5'}


def trans_torch(list1):
    list1 = np.array(list1)
    l1 = np.where(list1 == 1)
    l2 = np.where(list1 == 2)
    b = np.array([l1, l2])
    b = torch.FloatTensor(b).to("cuda:0")
    return b

class MyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer0 = nn.Linear(8, 64)
        self.layer1 = nn.Linear(64, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 4)
        self.activate = nn.GELU()

    def forward(self, x):
        x = torch.FloatTensor(x).to("cuda:0")
        x = self.activate(self.layer0(x))
        x = self.activate(self.layer1(x))
        x = self.activate(self.layer2(x))
        x = self.layer3(x)
        return x


def teleport_to_sheep(world):
    if world.world_state:
        for entity in world.world_state["entities"]:
            if entity["name"] == "Sheep":
                return "tp " + str(entity["x"]) + " 4 " + str(entity["z"] - 1)
    return ""


def take_action(agent_host, world, action):
    if action == 'hold_wheat':
        # assume wheat is in slo t2
        agent_host.sendCommand("hotbar.2 1")
        agent_host.sendCommand("hotbar.2 0")
    elif action == 'hide_wheat':
        agent_host.sendCommand("hotbar.1 1")
        agent_host.sendCommand("hotbar.1 0")
    elif action == 'teleport_to_sheep':
        agent_host.sendCommand(teleport_to_sheep(world))
    else:
        agent_host.sendCommand(action)


def correct_coords(world, agent_host, action):
    x, z = world.coords
    if x % 0.5 != 0 or z % 0.5 != 0:
        x_delta = -0.5 if action == 3 else 0.5
        z_delta = -0.5 if action == 0 else 0.5
        agent_host.sendCommand(
            "tp " + str(int(x) + x_delta) + " 4 " + str(int(z) + z_delta))


if __name__ == "__main__":
    logging.basicConfig(filename='./training_info.log', filemode='w', level=logging.DEBUG)
    world = World()
    model = MyNet()
    if os.path.exists('FCnet.pth'):
        model = torch.load('FCnet.pth')
        logging.info("continue training")
    model.to("cuda:0")
    max_memory = 100000
    epsilon = 0.1
    experience = Experience(model, max_memory=max_memory, discount=0.5)
    agent_host = MalmoPython.AgentHost()
    # -- set up the mission -- #
    mission_file = './farm.xml'
    with open(mission_file, 'r') as f:
        logging.info("Loading mission from %s" % mission_file)
        mission_xml = f.read()
        my_mission = MalmoPython.MissionSpec(mission_xml, True)

    max_retries = 10
    num_repeats = 100
    mission_avg_rewards = []
    mission_max_rewards = []
    mission_num_actions = []
    mission_losses = []

    f = open("numactions.txt", "a")
    f1 = open("average_rewards.txt", "w")
    f2 = open("max_rewards.txt", "w")
    f3 = open("loss.txt", "w")
    for i in range(num_repeats):
        rewards = []
        world.reset()
        logging.info('Repeat %d of %d' % (i + 1, num_repeats))

        my_mission_record = MalmoPython.MissionRecordSpec()

        for retry in range(max_retries):
            try:
                agent_host.startMission(my_mission, my_mission_record)
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    logging.info("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2.5)

        logging.info("Waiting for the mission to start\n")
        world_state = agent_host.getWorldState()
        envstate = world.observe(world_state)
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                logging.info("Error:", error.text)
        # Hide the wheat to start each mission.
        # 用稻草引诱羊
        agent_host.sendCommand("hotbar.2 1")
        agent_host.sendCommand("hotbar.2 0")
        # -- run the agent in the world -- #
        num_actions = 0
        prev_envstate = envstate
        action = 0
        while world_state.is_mission_running:
            time.sleep(0.3)
            prev_envstate = envstate
            prev_action = action

            if world.shouldReturn:
                print('Return action: ', end=' ')
                action = world.returnToStart()
                print(str(action))
            elif np.random.rand() < epsilon:
                print('Random action: ', end=' ')
                action = random.choice(world.getValidActions())
                print(str(action))
            else:
                print('Predicted action: ', end=' ')
                action = torch.argmax(experience.predict(prev_envstate)).data.cpu()
                action = int(action)
                print(str(action))

            take_action(agent_host, world, actionMap[action])
            num_actions += 1

            world_state = agent_host.getWorldState()
            envstate, reward, game_status = world.update_state(world_state, action)
            rewards.append(reward)

            # Correct the agent's coordinates in case a sheep pushed it
            correct_coords(world, agent_host, actionMap[action])

            game_over = game_status == 'win' or game_status == 'lose'
            episode = (prev_envstate, prev_action, reward, envstate, game_over)
            experience.remember(episode)

            if not world.shouldReturn:
                loss_fn = nn.HuberLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
                loss = experience.train(criterion=loss_fn, optimizer=optimizer, epoch=5)
                mission_losses.append(loss)
            if experience.train_count % 100 == 0:
                torch.save(model, 'FCnet.pth')
            if game_over:
                agent_host.sendCommand("quit")
                break
        # -- clean up -- #
        if i % 100 == 0:
            epsilon /= 2
        # compute average reward, and max reward
        template = "Iteration: {:d} | Average Reward: {:.4f} | Max Reward: {:.4f}"
        avg_reward = sum(rewards) / len(rewards)
        mission_avg_rewards.append(avg_reward)
        max_reward = max([r for r in rewards if r != -1])  # ignore -1 rewards
        mission_max_rewards.append(max_reward)
        mission_num_actions.append(num_actions)
        logging.info(template.format(i, avg_reward if rewards else 0, max_reward if rewards else 0))
        logging.info("num actions: " + str(num_actions))
        time.sleep(0.5)  # (let the Mod reset)
    logging.info("All mission average rewards: " + str(mission_avg_rewards))
    logging.info("All mission max rewards: " + str(mission_max_rewards))
    logging.info("All mission number of actions: " + str(mission_num_actions))
    logging.info("All mission loss: " + str(mission_losses))
    f1.write("\n".join((str(x) for x in mission_avg_rewards)))
    f2.write("\n".join((str(x) for x in mission_max_rewards)))
    f.write("\n".join((str(x) for x in mission_num_actions)))
    f3.write("\n".join((str(x) for x in mission_losses)))
    f.close()
    f2.close()
    f1.close()
    f3.close()
    # h5file = "model" + ".h5"
    json_file = "model" + ".json"
    with open(json_file, "w") as outfile:
        json.dump(model.to_json(), outfile)

    logging.info("Done.")
