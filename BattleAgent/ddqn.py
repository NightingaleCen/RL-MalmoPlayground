import torch
import torch.nn as nn
import torch.nn.functional as F
import configparser
import pickle
import numpy as np

class DQNModule(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.f1 = nn.Linear(input_size, 24)
        self.f1.weight.data.normal_(0, 0.1)
        self.f2 = nn.Linear(24, 32)
        self.f2.weight.data.normal_(0, 0.1)
        self.f3 = nn.Linear(32, 16)
        self.f3.weight.data.normal_(0, 0.1)
        self.f4 = nn.Linear(16, output_size)
        self.f4.weight.data.normal_(0, 0.1)
    
    def forward(self, x):
        x = self.f1(x)
        x = F.relu(x)
        x = self.f2(x)
        x = F.relu(x)
        x = self.f3(x)
        x = F.relu(x)
        return self.f4(x)

class DQNAgent:
    def __init__(self, state_size, action_size) -> None:
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.action_size = action_size
        self.epsilon = float(config.get('DEFAULT', 'EPSILON'))
        self.epsilon_min = float(config.get('DEFAULT', 'EPSILON_MIN'))
        self.epsilon_decay = float(config.get('DEFAULT', 'EPSILON_DECAY'))
        self.gamma = float(config.get('DEFAULT', 'GAMMA'))
        self.memory_capacity = int(config.get('DEFAULT', 'MEMORY_CAPACITY'))
        self.replace_steps = int(config.get('DEFAULT', 'REPLACE_STEPS'))
        LR = float(config.get('DEFAULT', 'LEARNING_RATE'))
        self.target_net, self.eval_net = DQNModule(state_size, action_size), DQNModule(state_size, action_size)
        self.memory = np.zeros((self.memory_capacity, state_size * 2 + 2)) # (s[:7], a[7], r[8], s_[9:])
        self.memory_count = 0
        self.steps = 0
        self.optimizer = torch.optim.SGD(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.SmoothL1Loss()

    def update_target_model(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        index = len(self.memory) % self.memory_capacity          
        mem = np.append(state, action)
        mem = np.append(mem, reward)
        mem = np.append(mem, next_state)
        self.memory[index] = mem
        if self.memory_count < self.memory_capacity:
            self.memory_count += 1

    def act(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        if np.random.normal() >= self.epsilon:
            actions = self.eval_net.forward(state).flatten()
            action = torch.max(actions, 0)[1].data.numpy()
        else:
            action = np.array([np.random.randint(0, self.action_size)])
        return action.item()

    def replay(self, batch_size):
        if self.steps % self.replace_steps == 0:
            self.update_target_model()

        sample_index = np.random.choice(self.memory_count,batch_size)
        batch_s = torch.FloatTensor(self.memory[sample_index, :7])
        batch_s_ = torch.FloatTensor(self.memory[sample_index, 9:])
        batch_a = torch.LongTensor(self.memory[sample_index, 7]).unsqueeze(1)
        batch_r = torch.LongTensor(self.memory[sample_index, 8]).unsqueeze(1)

        q_eval = self.eval_net.forward(batch_s).gather(1, batch_a)
        with torch.no_grad():
            q_next = self.target_net.forward(batch_s_)
            q_target = batch_r + self.gamma * q_next.max(1, keepdim=True)[0]
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad() 
        loss.backward() 
        self.optimizer.step()
        self.steps += 1
        return loss.item()

    def save(self, file_name):
        model = {}
        model['epsilon'] = self.epsilon
        model['target'] = self.target_net.state_dict()
        model['eval'] = self.eval_net.state_dict()
        model['memo'] = self.memory
        model['memo_count'] = self.memory_count
        with open(file_name, 'wb') as f:
            pickle.dump(model, f)

    def load(self, file_name):
        with open(file_name, 'rb') as f:
            model = pickle.load(f)
        self.epsilon =model['epsilon']
        self.target_net.load_state_dict(model['target'])
        self.eval_net.load_state_dict(model['eval'])
        self.memory = model['memo']
        self.memory_count = model['memo_count']