import csv
import time
def writewrite(lists):
    try:
        with open('lambda_cal.csv', 'a+', newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(lists)
    except:
        print('冲突，正在重试')
        time.sleep(1)
        writewrite(lists)
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
        writewrite([Kp1,Kp2])