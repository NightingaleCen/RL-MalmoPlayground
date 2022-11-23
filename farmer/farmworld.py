import numpy as np
import json
import math
import time

GATE_COORDINATES = (8, -3)


class World:
    def __init__(self):
        self.reset()

    def reset(self):
        self.coords = (0, 0)
        self.prevCoords = (0, 0)
        self.state = (0, 0)
        self.total_reward = 0
        self.total_steps = 100
        self.sheeps = set()
        self.finished = 0

        self.actions = 7
        self.prevAction = None
        self.world = np.zeros((21, 21))
        self.world_state = None
        self.shouldReturn = False
        self.holding_wheat = False

    # 只允许agent选择移动的action
    def getValidActions(self):
        return [0, 1, 2, 3]

    def game_status(self):
        if self.total_steps > 0:
            if self.finished >= 3:
                return "win"
            else:
                return "playing"
        else:
            if self.finished >= 2:
                return "win"
            else:
                return "lose"

    def observe(self, world_state):
        envstate = np.zeros(8)
        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            ob = json.loads(msg)
            index = 2
            for i in ob['entities']:
                # 防止溢出21*21的格子
                x = round(i["x"] - 0.5)
                z = round(i["z"] - 0.5)
                if i["name"] == "Agnis":
                    envstate[0] = x
                    envstate[1] = z
                elif i["name"] == "Sheep":
                    envstate[index] = x
                    envstate[index + 1] = z
                    index += 2
        return envstate

    def agentInPen(self):
        x, z = self.state
        return 5 > x > 0 and -1 > z > -5

    def sheepInPen(self, x, z):
        return 6 > x > 0 and -1 > z > -5

    def returnToStart(self):
        x, z = self.state
        time.sleep(0.1)

        if self.agentInPen():
            if self.shouldReturn:
                self.shouldReturn = False
                return 5
            else:
                return 6

        if x > 9 and z < -1:
            return 3
        elif x < 8 and z > -3:
            return 2
        elif z > -3:
            return 0
        else:
            return 3

    def update_state(self, world_state, action):
        envstate = np.zeros(8)
        status = self.game_status()
        reward = 0
        self.finished = 0
        if action not in self.getValidActions():
            reward -= 20

        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            ob = json.loads(msg)
            self.world_state = ob
            index = 2
            for i in ob['entities']:
                if i["name"] == "Agnis":  # 自己名字
                    # 防止溢出21*21的格子
                    x = round(i["x"] - 0.5)
                    z = round(i["z"] - 0.5)
                    # 更新自己所处的坐标、世界的信息等等
                    self.prevCoords = self.coords
                    self.coords = (i["x"], i["z"])
                    self.world[x][z] = 1
                    self.state = (x, z)
                    envstate[0] = x
                    envstate[1] = z
                # 基于羊的数量计算reward
                elif i["name"] == "Sheep":
                    x = i["x"]
                    z = i["z"]
                    envstate[index] = x
                    envstate[index + 1] = z
                    index += 2
                    row, col = self.state
                    # 计算人与羊的距离
                    dist = (x - row) ** 2 + (z - col) ** 2
                    if dist <= 6:
                        # 距离羊接近的奖励
                        if self.sheepInPen(x, z):
                            # 羊进入栅栏，则给20奖励
                            reward += 50
                            self.finished += 1
                            if self.finished >= 2:
                                print("!!!!!!!!!!!!!!!!   WIN   !!!!!!!!!!!!!!!!")
                                self.shouldReturn == True
                        elif i["id"] not in self.sheeps:
                            # 领上新羊，给5奖励
                            self.sheeps.add(i["id"])
                    else:
                        # 与羊的距离惩罚，越近惩罚越低
                        reward -= dist / 50
                    # 防止溢出21*21的格子
                    x = round(i["x"] - 0.5)
                    z = round(i["z"] - 0.5)
                    # 本格标记为羊
                    self.world[x][z] = 2
            if len(self.sheeps) >= 2:
                # 计算自己与栅栏门的距离
                row, col = self.state
                dx = row - GATE_COORDINATES[0]
                dz = col - GATE_COORDINATES[1]
                if col < -1:
                    dx = row - 1
                    dz = col + 3
                dist3 = math.sqrt(dx ** 2 + dz ** 2)
                reward = 0.2 * reward + (20 - 1.5 * dist3)
                if len(self.sheeps) >= 3:
                    reward = (30 - 2 * dist3)
            else:
                reward -= 10
        reward += 5 * len(self.sheeps)
        self.total_reward += reward
        self.prevAction = action
        return envstate, reward, status
