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
model = cloudpss.Model.fetch('model/admin/VdcIq_Vf_tune')
config = model.configs[0]  # 不填默认用model的第一个config
job = model.jobs[0]  # 不填默认用model的第一个job

lead_lag = model.getComponentByKey('component_new_lead_lag_2')
Data= {}
number=1
for i in range(-80,80,10):

    fc=5.8
    phi=i
    a=(1-math.sin(math.radians(phi)))/(1+math.sin(math.radians(phi)))
    T1=1./(2*math.pi*fc*math.sqrt(a))
    T2=a*T1
    lead_lag.args['T1']=T1
    lead_lag.args['T2']=T2


    try:
        runner = model.run(job, config)
        cloudpss_runner_wait()
    except func_timeout.exceptions.FunctionTimedOut as e:
        print("Time out!!!")
    legend = runner.result.getPlotChannelNames(0)
    Data['a'+str(number)]=[runner.result.getPlotChannelData(0,legend[0])['x'],runner.result.getPlotChannelData(0,legend[0])['y']]
    print('第',number,'次跑完')

    number+=1

#写
file_name='data.mat'
savemat(file_name,Data)
