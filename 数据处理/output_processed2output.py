import csv
import time
import numpy as np
import pandas as pd
import math
def writewrite(lists):
    try:
        # with open('e1.csv', 'a+', newline="") as output_file:
        # with open('P6x_Q1x_1.csv', 'a+', newline="") as output_file:
        with open('P6x_Q1x_2.csv', 'a+', newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(lists)
    except:
        print('冲突，正在重试')
        time.sleep(1)
        writewrite(lists)
# pd_reader = pd.read_csv("./Yo20op1084_ana_outputs.csv",header=None)
# pd_reader = pd.read_csv("./Yo20op1084_ana_prediction.csv",header=None)
# pd_reader = pd.read_csv("./P6x_Q1x_ana_outputs.csv",header=None)
pd_reader = pd.read_csv("./P6x_Q1x_ana_prediction.csv",header=None)
output=np.array(pd_reader)
output.tolist()
for list in output:
    [Ydd_x,Ydd_y,Ydq_x,Ydq_y,Yqd_x,Yqd_y,Yqq_x,Yqq_y]=list
    Ydd_raw = math.sqrt(pow(Ydd_x,2)+pow(Ydd_y,2))
    Ydq_raw = math.sqrt(pow(Ydq_x,2)+pow(Ydq_y,2))
    Yqd_raw = math.sqrt(pow(Yqd_x,2)+pow(Yqd_y,2))
    Yqq_raw = math.sqrt(pow(Yqq_x,2)+pow(Yqq_y,2))
    Ydd = 20 * math.log(Ydd_raw, 10)
    Ydq = 20 * math.log(Ydq_raw, 10)
    Yqd = 20 * math.log(Yqd_raw, 10)
    Yqq = 20 * math.log(Yqq_raw, 10)
    phdd = math.degrees(math.acos(Ydd_x / Ydd_raw))
    phdq = math.degrees(math.acos(Ydq_x / Ydq_raw))
    phqd = math.degrees(math.acos(Yqd_x / Yqd_raw))
    phqq = math.degrees(math.acos(Yqq_x / Yqq_raw))

    writewrite([Ydd,Ydq,Yqd,Yqq,phdd,phdq,phqd,phqq])