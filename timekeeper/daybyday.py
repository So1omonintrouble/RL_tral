import time
class person:
    def __init__(self,age=23):
        self.age=age
    def status(self,IsHaveWork,IsHaveTime):
        if IsHaveWork:
            print('工位工作')
        else:
            print('单独工作')
        if IsHaveTime:
            print('有空打球')
        else:
            print('今天没空')

shangzixuan=person()
shangzixuan.status(0,1)

