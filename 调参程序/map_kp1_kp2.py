import sys, os
import cloudpss
import json
import time
import csv
import logging
import torch
import torch.nn.functional as F
import numpy as np
import math
import pickle
from func_timeout import func_set_timeout
import func_timeout
from scipy.io import savemat
@func_set_timeout(60)
def cloudpss_runner_wait():
    while not runner.status():
        time.sleep(1)
cloudpss.setToken('eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MSwidXNlcm5hbWUiOiJhZG1pbiIsInNjb3BlcyI6W10sInR5cGUiOiJhcHBseSIsImV4cCI6MTcyMjkzMDE3OSwiaWF0IjoxNjkxODI2MTc5fQ.BTN41TZG3igDZ4v3nA6hBCQAThRVmbJLLPEcDaescliw7iYyX6UEiuDQ1WLfdLE8ut1CHpD2PAiQahRxVqGV5g')
### 获取指定 rid 的项目
os.environ['CLOUDPSS_API_URL'] = 'http://10.42.0.1'
model = cloudpss.Model.fetch('model/admin/VdcIq_Vf_VdcIq_tune')
config = model.configs[0]  # 不填默认用model的第一个config
job = model.jobs[0]  # 不填默认用model的第一个job
kp11 = model.getComponentByKey('component_new_step_gen_5')
kp12 = model.getComponentByKey('component_new_step_gen_6')
kp21 = model.getComponentByKey('component_new_step_gen_9')
kp22 = model.getComponentByKey('component_new_step_gen_13')

Data= {}
number=1
a=list(range(1,51,1))
b=[i*0.01 for i in a]
c=[float('{:.2f}'.format(i)) for i in b]
d=[i*10 for i in c]
e=[float('{:.1f}'.format(i)) for i in d]
e.append(10)
e.append(15)
e.append(20)
for Kp1 in c:
    for Kp2 in e:
        kp11.args['V1'] = Kp1
        kp12.args['V1'] = Kp1
        kp21.args['V1'] = Kp2
        kp22.args['V1'] = Kp2
        try:
            runner = model.run(job, config)
            cloudpss_runner_wait()
        except func_timeout.exceptions.FunctionTimedOut as error1:
            print("Time out!!!")
        legend = runner.result.getPlotChannelNames(0)
        Data['a' + str(number)] = [runner.result.getPlotChannelData(0, legend[0])['x'],
                                   runner.result.getPlotChannelData(0, legend[0])['y']]
        print('第', number, '次跑完')

        number += 1

#写
file_name='map_kp1_kp2.mat'
savemat(file_name,Data)

#读
# with open('pickle_example.pickle', 'rb') as file:
#     a_dict1 = pickle.load(file)
#
# print(a_dict1)

# plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
# plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题