from __future__ import print_function
import copy
import numpy as np
import torch
import logging


class Experience(object):
    def __init__(self, model, max_memory=100000, discount=0.95):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.train_samples = 20
        self.train_count = 0
        self.num_actions = 7
        self.target_model = copy.deepcopy(model)

    # def kickoff(self, episode):
    #     rewards = []
    #     for m in self.memory:
    #         rewards.append(m[2])
    #     rewards = np.array(rewards)
    #     args = np.argpartition(rewards, 10)[:10]
    #     if episode[2] < rewards[args[0]]:
    #         return False
    #     index = np.random.randint(0, 10)
    #     print("kickoff reward:" + str(rewards[args[index]]))
    #     logging.info("kickoff reward:" + str(rewards[args[index]]))
    #     del self.memory[args[index]]
    #     return True

    def remember(self, episode):
        # episode = [envstate, action, reward, envstate_next, game_over]
        envstate, action, reward, envstate_next, game_over = episode
        x, z = envstate[0], envstate[1]
        x = int(x)
        z = int(z)
        if z == 0 and x <= 7 and action == 0:
            reward -= 30
        elif z <= -4 and action == 0:
            reward -= 30
        elif z >= 19 and action == 1:
            reward -= 30
        elif x == 0 and action == 3:
            reward -= 30
        elif x >= 19 and action == 2:
            reward -= 30
        elif x == int(envstate_next[0]) and z == int(envstate_next[1]):
            return
        reward /= 100   # smoothing
        episode = (envstate, action, reward, envstate_next, game_over)
        if len(self.memory) < self.max_memory:
            self.memory.append(episode)
        else:
            # if self.kickoff(episode):  # kickoff不太奏效！
            del self.memory[0]
            self.memory.append(episode)

    def predict(self, envstate):
        if envstate is not None:
            return self.model(envstate)

    def train(self, criterion, optimizer, epoch=10):
        inputs = []
        targets = []
        actions = []
        # 随机选取20条经验
        for i in range(self.train_samples):
            e_num = np.random.choice(range(len(self.memory)))
            e = self.memory[e_num]
            state, action, reward, state_next, game_over = e
            inputs.append(state)
            # 利用target网络，计算回报的预测值
            q_next = self.target_model(state_next).cpu().detach()
            target = reward + self.discount * q_next.max(0)[0].data.numpy()
            targets.append(target)
            actions.append([action])

        actions = torch.LongTensor(actions).to("cuda:0")
        # 进行训练
        for t in range(epoch):
            self.train_count += 1
            reward_pred = self.model(inputs).gather(1, actions).permute(1, 0)[0]
            reward_target = torch.FloatTensor(np.array(targets)).to("cuda:0")
            loss = criterion(reward_pred, reward_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if self.train_count % 1000 == 0:
            # 每隔1000代，target网络更新一次
            logging.info("update target net")
            self.target_model = copy.deepcopy(self.model)
        return loss
