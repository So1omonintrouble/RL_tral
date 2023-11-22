#
# import torch
# import torch.nn.functional as F
# import numpy as np
# try:
#     dict1==0
# except:
#     dict1={
#         'a':0
#     }
# try:
#     net==0
# except:
#     class Qnet(torch.nn.Module):
#         def __init__(self, state_dim, hidden_dim, action_dim):
#             super(Qnet, self).__init__()
#             self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
#             self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
#         def forward(self, x):
#             x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
#             return self.fc2(x)
#     net=Qnet(3,128,3)
# save=net.state_dict()
# fc1_w=save['fc1.weight']
# w1=fc1_w.numpy()
# #print(list(w1.shape))
# fc1_b=save['fc1.bias']
# b1=fc1_b.numpy()
# #print(list(b1.shape))
# fc2_w=save['fc2.weight']
# w2=fc2_w.numpy()
# #print(list(w2.shape))
# fc2_b=save['fc2.bias']
# b2=fc2_b.numpy()
# #print(list(b2.shape))
# save=net.state_dict()
# fc1_w=save['fc1.weight']
# w1=fc1_w.numpy()
# #print(list(b2.shape))
# w1=w1.T
# w1=w1[0]
# w1=w1.tolist()
global dict1
global net
global c1
try:
    c1 == 0
except:
    c1 = 0
import torch
import torch.nn.functional as F
import numpy as np

try:
    dict1 == 0
except:
    dict1 = {
        'a': 0
    }
try:
    net == 0
except:
    class Qnet(torch.nn.Module):
        def __init__(self, state_dim, hidden_dim, action_dim):
            super(Qnet, self).__init__()
            self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
            self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

        def forward(self, x):
            x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
            return self.fc2(x)


    net = Qnet(3, 128, 3)
save = net.state_dict()
##处理
fc1_w = save['fc1.weight']
fc1_b = save['fc1.bias']
fc2_w = save['fc2.weight']
fc2_b = save['fc2.bias']

fc1_w = fc1_w.numpy()
fc1_b = fc1_b.numpy()
fc2_w = fc2_w.numpy()
fc2_b = fc2_b.numpy()

fc1_w = fc1_w.T

fc1_w_1 = fc1_w[0]
fc1_w_2 = fc1_w[1]
fc1_w_3 = fc1_w[2]

fc2_w_1 = fc2_w[0]
fc2_w_2 = fc2_w[1]
fc2_w_3 = fc2_w[2]
##最后输出到示波器
fc1_w_1 = fc1_w_1.tolist()
fc1_w_2 = fc1_w_2.tolist()
fc1_w_3 = fc1_w_3.tolist()

fc1_b = fc1_b.tolist()

fc2_w_1 = fc2_w_1.tolist()
fc2_w_2 = fc2_w_2.tolist()
fc2_w_3 = fc2_w_3.tolist()

fc2_b = fc2_b.tolist()

##还原
save['fc1.weight']=torch.from_numpy(np.array([fc1_w_1,fc1_w_2,fc1_w_3]).T)
save['fc1.bias']=torch.from_numpy(np.array(fc1_b))
save['fc2.weight']=torch.from_numpy(np.array([fc2_w_1,fc2_w_2,fc2_w_3]))
save['fc2.bias']=torch.from_numpy(np.array(fc2_b))

net.load_state_dict(save)


class DQN_simp:
    def __init__(self,net,learning_rate,epsilon):
        self.q_net=net
        self.optimizer=torch.optim.Adam(self.q_net.parameters(),lr=learning_rate)
        self.gamma = 0.98
        self.epsilon = epsilon
    def take_action(self,state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(3)
        else:
            state = torch.tensor(state,dtype=torch.float)
            action = self.q_net(state).argmax().item()
        return action
    def update(self,state,action,reward,state1):
        state = torch.tensor(state,dtype=torch.float)
        action = torch.tensor(action,dtype=torch.float)
        reward = torch.tensor(action,dtype=torch.float)
        state1 = torch.tensor(state1,dtype=torch.float)

        q_value = self.q_net(state)[action]
        max_next_q_value = self.q_net(state1).max()
        q_target = reward + self.gamma * max_next_q_value
        dqn_loss = torch.mean(F.mse_loss(q_value,q_target))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()


