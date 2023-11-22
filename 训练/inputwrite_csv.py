import csv
import time
def writewrite(lists):
    try:
        with open('P6x_Q1x_input.csv', 'a+', newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(lists)
    except:
        print('冲突，正在重试')
        time.sleep(1)
        writewrite(lists)
f = [1, 1.315789474, 1.754385965, 2.325581395, 3.03030303, 4, 5.357142857, 6.976744186, 9.375, 12.5, 16.66666667,
         21.42857143, 28.57142857, 37.5, 50, 66.66666667, 87.5, 116.6666667, 150, 200]
V = [0.9,1,1.1]
Q = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
P = [0.6,0.7,0.8,0.9,1]
for Idc in P:
    for Iq in Q:
        for Vd in V:
            for frequency in f:
                writewrite([Idc,Iq,Vd,frequency])

var1 = 1
var2 = 10
