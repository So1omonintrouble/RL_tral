import csv
import time
import numpy as np
import pandas as pd
import math
def writewrite(lists):
    try:
        with open('P6x_Q1x_processed.csv', 'a+', newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(lists)
    except:
        print('冲突，正在重试')
        time.sleep(1)
        writewrite(lists)
pd_reader = pd.read_csv("./P6x_Q1x.csv",header=None)
output=np.array(pd_reader)
output.tolist()
for list in output:
    [Ydd,Ydq,Yqd,Yqq,phdd,phdq,phqd,phqq]=list
    Ydd_raw = pow(10, Ydd / 20)
    Ydq_raw = pow(10, Ydq / 20)
    Yqd_raw = pow(10, Yqd / 20)
    Yqq_raw = pow(10, Yqq / 20)
    Ydd_x=Ydd_raw*math.cos(math.radians(phdd))
    Ydd_y=Ydd_raw*math.sin(math.radians(phdd))
    Ydq_x=Ydq_raw*math.cos(math.radians(phdq))
    Ydq_y=Ydq_raw*math.sin(math.radians(phdq))
    Yqd_x=Yqd_raw*math.cos(math.radians(phqd))
    Yqd_y=Yqd_raw*math.sin(math.radians(phqd))
    Yqq_x=Yqq_raw*math.cos(math.radians(phqq))
    Yqq_y=Yqq_raw*math.sin(math.radians(phqq))
    writewrite([Ydd_x,Ydd_y,Ydq_x,Ydq_y,Yqd_x,Yqd_y,Yqq_x,Yqq_y])