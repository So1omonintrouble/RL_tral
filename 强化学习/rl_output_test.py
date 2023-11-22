import sys, os
import cloudpss
import json
import matplotlib.pyplot as plt
import time
import csv
import logging
import torch
import torch.nn.functional as F
import numpy as np
global net
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
print('ok')
cloudpss.setToken(
    'eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MSwidXNlcm5hbWUiOiJhZG1pbiIsInNjb3BlcyI6W10sInR5cGUiOiJhcHBseSIsImV4cCI6MTcyNjc0NDkzOCwiaWF0IjoxNjk1NjQwOTM4fQ.6-JjFbt7VVqasckAf0S1jPKZHDOjhLkeCjXlJcpBcfNmmp-xznb4Cp9Id8eOgsRyKFNTAB_palm4VDdtVxKN2Q')
os.environ['CLOUDPSS_API_URL'] = 'http://10.42.0.1'
model = cloudpss.Model.fetch('model/admin/octave_test')
config = model.configs[0]  # 不填默认用model的第一个config
job = model.jobs[0]  # 不填默认用model的第一个job

controller = model.getComponentByKey('component_python_test_scatter_1')

controller.args['fc1_b_i']=','.join(map(str,fc1_b))
controller.args['fc1_w_1']=','.join(map(str,fc1_w_1))
controller.args['fc1_w_2']=','.join(map(str,fc1_w_2))
controller.args['fc1_w_3']=','.join(map(str,fc1_w_3))
controller.args['fc2_b_i']=','.join(map(str,fc2_b))
controller.args['fc2_w_1']=','.join(map(str,fc2_w_1))
controller.args['fc2_w_2']=','.join(map(str,fc2_w_2))
controller.args['fc2_w_3']=','.join(map(str,fc2_w_3))

runner = model.run(job, config)
while not runner.status():
    time.sleep(0.1)
legend = runner.result.getPlotChannelNames(1)

fc1_w_1_t = []
fc1_w_2_t = []
fc1_w_3_t = []

fc1_b_t = []

fc2_w_1_t = []
fc2_w_2_t = []
fc2_w_3_t = []

fc2_b_t = []

for i in range(128):
    a=runner.result.result['plot-0']['data']['traces'][i]['y'][0]
    fc1_w_1_t.append(a)
for i in range(128):
    a=runner.result.result['plot-1']['data']['traces'][i]['y'][0]
    fc1_w_2_t.append(a)
for i in range(128):
    a=runner.result.result['plot-2']['data']['traces'][i]['y'][0]
    fc1_w_3_t.append(a)
for i in range(128):
    a=runner.result.result['plot-3']['data']['traces'][i]['y'][0]
    fc1_b_t.append(a)
for i in range(128):
    a=runner.result.result['plot-4']['data']['traces'][i]['y'][0]
    fc2_w_1_t.append(a)
for i in range(128):
    a=runner.result.result['plot-5']['data']['traces'][i]['y'][0]
    fc2_w_2_t.append(a)
for i in range(128):
    a=runner.result.result['plot-6']['data']['traces'][i]['y'][0]
    fc2_w_3_t.append(a)
for i in range(3):
    a=runner.result.result['plot-7']['data']['traces'][i]['y'][0]
    fc2_b_t.append(a)
