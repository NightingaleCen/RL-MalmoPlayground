import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# import torch.distributions as distributions
class PPO(nn.Module):
    def __init__(self, in_ndim, pi_ndim, v_ndim, Dorger):
        super(PPO, self).__init__()
        # some basic params
        self.gamma = Dorger.gamma
        self.lr = Dorger.alpha
        self.lmbda = 0.95
        self.eps_clip = 0.2

        self.data = []

        # 如果训练不好，考虑是不是因为状态的数值都是几百，并且不同状态下的数值差异相对较小（比如-310 与 -312的区别）
        self.fc1 = nn.Linear(in_ndim, 512)
        # self.fc2 = nn.Linear(512, 16)
        self.fc_pi = nn.Linear(512, pi_ndim)
        self.fc_v = nn.Linear(512, v_ndim)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc_pi(x)
        # print('before softmax', x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc_v(x)
        return x

    def put_data(self, transition):
        """
        注意,这里transition必须是一个list类型   TODO:可能需要对s本身的数值进行修改,因为差异太小了
        """
        self.data.append(transition)
    
    def make_batch(self):
        s_lst, a_lst, r_lst, next_s_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for t in self.data:
            s, a, r, next_s, prob_a, done = t
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            next_s_lst.append(next_s)
            prob_a_lst.append(prob_a)
            done_mask = 0 if done else 1    # 如果成功到达终点则mask=0，其他情况则为1
            done_lst.append([done_mask])
        s, a, r, next_s, prob_a, done_mask = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst),\
                                             torch.tensor(r_lst, dtype=torch.float), torch.tensor(next_s_lst, dtype=torch.float),\
                                             torch.tensor(prob_a_lst), torch.tensor(done_lst, dtype=torch.float)
        self.data = []
        return s, a, r, next_s, prob_a, done_mask

    def train_net(self):
        s, a, r, next_s, prob_a, done_mask = self.make_batch()

        td_target = r + self.gamma * self.v(next_s) * done_mask
        delta = td_target - self.v(s)
        delta = delta.detach().numpy()

        advantage_lst = []
        advantage = 0.0
        for delta_t in delta[::-1]:
            # GAE算法
            advantage = self.gamma * self.lmbda * advantage + delta_t[0]
            advantage_lst.append(advantage)
        advantage_lst.reverse()
        advantage = torch.tensor(advantage_lst, dtype=torch.float)

        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1, a.reshape(-1, 1))
        
        ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a.reshape(-1, 1)))

        surr1 = torch.dot(torch.squeeze(ratio), advantage)
        surr2 = torch.dot(torch.clamp(torch.squeeze(ratio), min=1-self.eps_clip, max=1+self.eps_clip), advantage)
    
        loss = -torch.min(surr1, surr2).mean() + F.mse_loss(self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()