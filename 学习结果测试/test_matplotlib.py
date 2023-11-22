# import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
pd_reader1 = pd.read_csv("./P6x_Q1x_1.csv",header=None)#c1为75个工况的标签
output1=np.array(pd_reader1)
output1.tolist()
pd_reader2 = pd.read_csv("./P6x_Q1x_2.csv",header=None)#c2为75个工况的预测结果
output2=np.array(pd_reader2)
output2.tolist()
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
list1=[]
list1_i=[]
list2=[]
list2_i=[]
for i in range(150*20):#1084/75
    list1_i.append(output1[i])
    list2_i.append(output2[i])
    if (i+1) % 20==0:
        list1.append(list1_i)
        list1_i=[]
        list2.append(list2_i)
        list2_i=[]

# x = [3,5,7,9,11,13,17,29,37,49,59,75,81,93,103,117,143,173,193,213]
x = [1,1.315789474,1.754385965,2.325581395,3.03030303,4,5.357142857,6.976744186,9.375,12.5,16.66666667,21.42857143,28.57142857,37.5,50,66.66666667,87.5,116.6666667,150,200]
#重复运行
# randomize = random.randint(0,149)#1083/74
randomize = 75
# randomize_i = random.choice([0,3,4,7])#phdq,phqd不算
randomize_i = 7
print("第%d个工况"%randomize)
print("第%d个扫频量"%randomize_i)
y1=[]
y2=[]
for i in range(20):
    # y1.append(list1[randomize][i][randomize_i])
    y2.append(list2[randomize][i][randomize_i])

fig, ax = plt.subplots()
# ax.plot(x, y1,label='实际结果')#扫频结果
# ax.plot(x, y2,label='预测结果')#预测结果
ax.plot(x, y2)
# ax.legend()
plt.rcParams['figure.figsize']=(5/2, 4/2)

fig.savefig('7.svg',dpi=600,format='svg')
plt.show()