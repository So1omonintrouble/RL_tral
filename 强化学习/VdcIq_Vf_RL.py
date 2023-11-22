# 把元件外部参数传入内部，主要实现字符串到列表的转换
global test_args
if test_args is None:
    test_args = {
        "fc1_w_1": args['fc1_w_1'],
        "fc1_w_2": args['fc1_w_2'],
        "fc1_w_3": args['fc1_w_3'],
        "fc1_b_i": args['fc1_b_i'],
        "fc2_w_1": args['fc2_w_1'],
        "fc2_w_2": args['fc2_w_2'],
        "fc2_w_3": args['fc2_w_3'],
        "fc2_b_i": args['fc2_b_i']
    }
fc1_w_1 = test_args['fc1_w_1']
fc1_w_2 = test_args['fc1_w_2']
fc1_w_3 = test_args['fc1_w_3']
fc1_b = test_args['fc1_b_i']
fc2_w_1 = test_args['fc2_w_1']
fc2_w_2 = test_args['fc2_w_2']
fc2_w_3 = test_args['fc2_w_3']
fc2_b = test_args['fc2_b_i']
# print('test',test_args['fc2_b_i'],flush=True)
fc1_w_1 = list(map(float, fc1_w_1.split(',')))
if len(fc1_w_1) != 128:
    fc1_w_1 = 0
fc1_w_2 = list(map(float, fc1_w_2.split(',')))
if len(fc1_w_2) != 128:
    fc1_w_2 = 0
fc1_w_3 = list(map(float, fc1_w_3.split(',')))
if len(fc1_w_3) != 128:
    fc1_w_3 = 0
fc1_b = list(map(float, fc1_b.split(',')))
if len(fc1_b) != 128:
    fc1_b = 0

fc2_w_1 = list(map(float, fc2_w_1.split(',')))
if len(fc2_w_1) != 128:
    fc2_w_1 = 0
fc2_w_2 = list(map(float, fc2_w_2.split(',')))
if len(fc2_w_2) != 128:
    fc2_w_2 = 0
fc2_w_3 = list(map(float, fc2_w_3.split(',')))
if len(fc2_w_3) != 128:
    fc2_w_3 = 0
fc2_b = list(map(float, fc2_b.split(',')))
if len(fc2_b) != 3:
    fc2_b = 0
# 定义全局变量：神经网络，强化学习agent，缓存区replay_buffer
global net
global agent
global replay_buffer
global episode_return
global state
global action
global number_step

import torch
import torch.nn.functional as F
import numpy as np
import random
import collections
import rl_utils

# 以下操作可以得到net，下面是具体逻辑解释
# 如果是第一次第一步仿真时创建net
# 第二步之后由于fc1_w_1=0，使用上一步的global net
# 如果是第二次之后的仿真，第一步由于不存在net，且参数传输数值均为0，使得fc1_w_1!=0，需要把参数打包到net中
# 第二步之后均有net，因此不再需要参数打包


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
    if fc1_w_1 != 0:
        # 打包到net中
        save = net.state_dict()
        save['fc1.weight'] = torch.from_numpy(np.array([fc1_w_1, fc1_w_2, fc1_w_3]).T)
        save['fc1.bias'] = torch.from_numpy(np.array(fc1_b))
        save['fc2.weight'] = torch.from_numpy(np.array([fc2_w_1, fc2_w_2, fc2_w_3]))
        save['fc2.bias'] = torch.from_numpy(np.array(fc2_b))
        net.load_state_dict(save)
# 以下操作得到agent，缓存区replay_buffer
try:
    agent == 0
except:
    class DQN:
        ''' DQN算法 '''

        def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                     epsilon, target_update, device):
            self.action_dim = action_dim
            self.q_net = Qnet(state_dim, hidden_dim,
                              self.action_dim).to(device)  # Q网络
            # 目标网络
            self.target_q_net = Qnet(state_dim, hidden_dim,
                                     self.action_dim).to(device)
            # 使用Adam优化器
            self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                              lr=learning_rate)
            self.gamma = gamma  # 折扣因子
            self.epsilon = epsilon  # epsilon-贪婪策略
            self.target_update = target_update  # 目标网络更新频率
            self.count = 0  # 计数器,记录更新次数
            self.device = device

        def take_action(self, state):  # epsilon-贪婪策略采取动作
            if np.random.random() < self.epsilon:
                action = np.random.randint(self.action_dim)
            else:
                state = torch.tensor([state], dtype=torch.float).to(self.device)
                action = self.q_net(state).argmax().item()
            return action

        def update(self, transition_dict):
            states = torch.tensor(transition_dict['states'],
                                  dtype=torch.float).to(self.device)
            actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
                self.device)
            rewards = torch.tensor(transition_dict['rewards'],
                                   dtype=torch.float).view(-1, 1).to(self.device)
            next_states = torch.tensor(transition_dict['next_states'],
                                       dtype=torch.float).to(self.device)
            dones = torch.tensor(transition_dict['dones'],
                                 dtype=torch.float).view(-1, 1).to(self.device)
            q_values = self.q_net(states).gather(1, actions)  # Q值
            # 下个状态的最大Q值
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
                -1, 1)
            q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                    )  # TD误差目标
            dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
            self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
            dqn_loss.backward()  # 反向传播更新参数
            self.optimizer.step()
            if self.count % self.target_update == 0:
                self.target_q_net.load_state_dict(
                    self.q_net.state_dict())  # 更新目标网络
            self.count += 1


    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    state_dim = 3
    action_dim = 3
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update, device)

try:
    replay_buffer == 0
except:
    class ReplayBuffer:
        ''' 经验回放池 '''

        def __init__(self, capacity):
            self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

        def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
            self.buffer.append((state, action, reward, next_state, done))

        def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
            transitions = random.sample(self.buffer, batch_size)
            state, action, reward, next_state, done = zip(*transitions)
            return np.array(state), action, reward, np.array(next_state), done

        def size(self):  # 目前buffer中数据的数量
            return len(self.buffer)


    replay_buffer = ReplayBuffer(buffer_size)

# 等4s暂态过程再开始学习，且之后的第一个步长不进行更新
try:
    number_step == 0
except:
    number_step = 1
if number_step >= 4000：
if number_step == 4000:
    episode_return == 0
    state = [input_state[0], input_state[1], input_state[2]]
    action = agent.take_action(state)
else:
    next_state = [input_state[0], input_state[1], input_state[2]]
    next_action = agent.take_action(next_state)
    replay_buffer.add(state, action, reward, next_state, done)
    state = next_state
    action = next_action
    episode_return += reward
    # 当buffer数据的数量超过一定值后,才进行Q网络训练
    if replay_buffer.size() > minimal_size:
        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
        transition_dict = {
            'states': b_s,
            'actions': b_a,
            'next_states': b_ns,
            'rewards': b_r,
            'dones': b_d
        }
        agent.update(transition_dict)
number_step = number_step + 1
# 在这里可以进行神经网络参数的更新

# 单独把权重拿出来变化
save = net.state_dict()
save['fc1.weight'][0][0] += 1  # 把第一个权重加1
net.load_state_dict(save)
# 以下代码将net所有参数传到示波器中，主要实现tensor到numpy，numpy到list
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

fc1_w_1 = fc1_w_1.tolist()

fc1_w_2 = fc1_w_2.tolist()
fc1_w_3 = fc1_w_3.tolist()

fc1_b = fc1_b.tolist()

fc2_w_1 = fc2_w_1.tolist()
fc2_w_2 = fc2_w_2.tolist()
fc2_w_3 = fc2_w_3.tolist()

fc2_b = fc2_b.tolist()
return [fc1_w_1, fc1_w_2, fc1_w_3, fc1_b, fc2_w_1, fc2_w_2, fc2_w_3, fc2_b]
# return [0,0,0,0,0,0,0,0,0]

test_args = None 