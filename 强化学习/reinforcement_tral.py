global net
global agent
global replay_buffer
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
episode_return = 0
state = [1,2,3]
done = False
action = agent.take_action(state)
next_state, reward, done= [[1,2,4],1,False]
replay_buffer.add(state, action, reward, next_state, done)
state = next_state
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