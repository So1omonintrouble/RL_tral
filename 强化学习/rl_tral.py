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

#执行一次仿真来获取神经网络参数
cloudpss.setToken(
    'eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MSwidXNlcm5hbWUiOiJhZG1pbiIsInNjb3BlcyI6W10sInR5cGUiOiJhcHBseSIsImV4cCI6MTcyMjkzMDE3OSwiaWF0IjoxNjkxODI2MTc5fQ.BTN41TZG3igDZ4v3nA6hBCQAThRVmbJLLPEcDaescliw7iYyX6UEiuDQ1WLfdLE8ut1CHpD2PAiQahRxVqGV5g')
os.environ['CLOUDPSS_API_URL'] = 'http://10.42.0.1'
model = cloudpss.Model.fetch('model/admin/VdcIq_Vf_tune_RL')
config = model.configs[0]  # 不填默认用model的第一个config
job = model.jobs[0]  # 不填默认用model的第一个job

runner = model.run(job, config)
while not runner.status():
    time.sleep(0.1)
legend = runner.result.getPlotChannelNames(1)

fc1_w_1 = []
fc1_w_2 = []
fc1_w_3 = []
fc1_b = []
fc2_w_1 = []
fc2_w_2 = []
fc2_w_3 = []
fc2_b = []

for i in range(128):
    a=runner.result.result['plot-0']['data']['traces'][i]['y'][-1]
    fc1_w_1.append(a)
for i in range(128):
    a=runner.result.result['plot-1']['data']['traces'][i]['y'][-1]
    fc1_w_2.append(a)
for i in range(128):
    a=runner.result.result['plot-2']['data']['traces'][i]['y'][-1]
    fc1_w_3.append(a)
for i in range(128):
    a=runner.result.result['plot-3']['data']['traces'][i]['y'][-1]
    fc1_b.append(a)
for i in range(128):
    a=runner.result.result['plot-4']['data']['traces'][i]['y'][-1]
    fc2_w_1.append(a)
for i in range(128):
    a=runner.result.result['plot-5']['data']['traces'][i]['y'][-1]
    fc2_w_2.append(a)
for i in range(128):
    a=runner.result.result['plot-6']['data']['traces'][i]['y'][-1]
    fc2_w_3.append(a)
for i in range(3):
    a=runner.result.result['plot-7']['data']['traces'][i]['y'][-1]
    fc2_b.append(a)
controller = model.getComponentByKey('component_python_test_scatter_1')
print('执行第1次仿真成功')
#为下一次仿真写入参数，执行，循环
for j in range(2):
    print('fc1_w_1[0]',fc1_w_1[0])
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

    fc1_w_1 = []
    fc1_w_2 = []
    fc1_w_3 = []
    fc1_b = []
    fc2_w_1 = []
    fc2_w_2 = []
    fc2_w_3 = []
    fc2_b = []

    for i in range(128):
        a = runner.result.result['plot-0']['data']['traces'][i]['y'][0]
        fc1_w_1.append(a)
    for i in range(128):
        a = runner.result.result['plot-1']['data']['traces'][i]['y'][0]
        fc1_w_2.append(a)
    for i in range(128):
        a = runner.result.result['plot-2']['data']['traces'][i]['y'][0]
        fc1_w_3.append(a)
    for i in range(128):
        a = runner.result.result['plot-3']['data']['traces'][i]['y'][0]
        fc1_b.append(a)
    for i in range(128):
        a = runner.result.result['plot-4']['data']['traces'][i]['y'][0]
        fc2_w_1.append(a)
    for i in range(128):
        a = runner.result.result['plot-5']['data']['traces'][i]['y'][0]
        fc2_w_2.append(a)
    for i in range(128):
        a = runner.result.result['plot-6']['data']['traces'][i]['y'][0]
        fc2_w_3.append(a)
    for i in range(3):
        a = runner.result.result['plot-7']['data']['traces'][i]['y'][0]
        fc2_b.append(a)
    print('执行第%d次仿真成功'%(j+2))


