import sys, os
import cloudpss
import json
import matplotlib.pyplot as plt
import time
import csv
import logging
logging.basicConfig(filename='230821.log', encoding='utf-8', level=logging.INFO)
def log(Idc,Iq,voltage,frequency):
    try:
        logging.info('有功：Idc=%.4f 无功：Iq=%.4f 电压:Vd=%.1f 频率：f=%.4f', Idc, Iq, voltage, frequency)
        logging.info(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        logging.info('\n')
    except:
        print('log失败，正在重试')
        time.sleep(2)
        log(Idc,Iq,voltage,frequency)
def run_cloudpss_d_axis(i):
    try:
        runner = model.run(job, config)
        while not runner.status():
            time.sleep(0.1)
        print("end", i, "times")

        legend = runner.result.getPlotChannelNames(0)

        Ydd = runner.result.getPlotChannelData(0, legend[0])['y'][19900]
        Yqd = runner.result.getPlotChannelData(0, legend[1])['y'][19900]
        phdd = runner.result.getPlotChannelData(0, legend[2])['y'][19900]
        phqd = runner.result.getPlotChannelData(0, legend[3])['y'][19900]
        i = i + 1
        return Ydd,Yqd,phdd,phqd,i
    except:
        print('第%d次仿真错误，正在重试'%i)
        time.sleep(1000)
        run_cloudpss_d_axis(i)
def run_cloudpss_q_axis(i):
    try:
        runner = model.run(job, config)
        while not runner.status():
            time.sleep(0.1)
        print("end", i, "times")

        legend = runner.result.getPlotChannelNames(1)

        Ydq = runner.result.getPlotChannelData(1, legend[0])['y'][19900]
        Yqq = runner.result.getPlotChannelData(1, legend[1])['y'][19900]
        phdq = runner.result.getPlotChannelData(1, legend[2])['y'][19900]
        phqq = runner.result.getPlotChannelData(1, legend[3])['y'][19900]
        i = i + 1
        return Ydq,Yqq,phdq,phqq,i
    except:
        print('第%d次仿真错误，正在重试'%i)
        time.sleep(5)
        run_cloudpss_q_axis(i)
def writewrite(lists):
    try:
        with open('P-1x_Q-1x.csv', 'a+', newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(lists)
    except:
        print('冲突，正在重试')
        time.sleep(1)
        writewrite(lists)
if __name__ == '__main__':
    cloudpss.setToken('eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MSwidXNlcm5hbWUiOiJhZG1pbiIsInNjb3BlcyI6W10sInR5cGUiOiJhcHBseSIsImV4cCI6MTcyMjkzMDE3OSwiaWF0IjoxNjkxODI2MTc5fQ.BTN41TZG3igDZ4v3nA6hBCQAThRVmbJLLPEcDaescliw7iYyX6UEiuDQ1WLfdLE8ut1CHpD2PAiQahRxVqGV5g')
    ### 获取指定 rid 的项目
    os.environ['CLOUDPSS_API_URL'] = 'http://10.42.0.1'
    model = cloudpss.Model.fetch('model/admin/VdcIq_fs')
    config = model.configs[0]  # 不填默认用model的第一个config
    job = model.jobs[0]  # 不填默认用model的第一个job
    f = [1, 1.315789474, 1.754385965, 2.325581395, 3.03030303, 4, 5.357142857, 6.976744186, 9.375, 12.5, 16.66666667,
         21.42857143, 28.57142857, 37.5, 50, 66.66666667, 87.5, 116.6666667, 150, 200]
    f_source_d = model.getComponentByKey('component_new_sin_gen_1')
    f_source_q = model.getComponentByKey('component_new_sin_gen_2')
    f_FFT_vd = model.getComponentByKey('component_new_fft_2')
    f_FFT_vq = model.getComponentByKey('component_new_fft_3')
    f_FFT_id = model.getComponentByKey('component_new_fft_1')
    f_FFT_iq = model.getComponentByKey('component_new_fft_4')
    #
    # ##V

    #
    # ##P
    # Idc=[0.03*(-1),0.03*(-0.5),0.03*0,0.03*0.5,0.03*1]
    P_current_source = model.getComponentByKey('component_new_dc_current_source_1')
    #
    Q_current_source = model.getComponentByKey('component_new_step_gen_9')
    # ##Q
    #
    # ##参数修改
    # ##f
    #
    ##扰动电压选取
    # f_source_d.args['Mag'] = 1
    # f_source_q.args['Mag'] = 0
    #
    ##扰动频率选取
    # f_source_d.args['F']['source'] = '3'
    # f_source_q.args['F']['source'] = '3'
    # f_FFT_vd.args['BaseFrequency']['source'] = '3'
    # f_FFT_id.args['BaseFrequency']['source'] = '3'
    # f_FFT_vq.args['BaseFrequency']['source'] = '3'
    # f_FFT_iq.args['BaseFrequency']['source'] = '3'

    # ##OP
    # V_constant.args['Value']=V[1]
    # P_current_source.args['Im']=P[4]
    # Q_current_source.args['V0']=1
    # Q_current_source.args['V1']=1
    i = 1
    output_list = []
    output = []
    V = [311 * 0.9, 311 * 1, 311 * 1.1]
    V_constant = model.getComponentByKey('component_new_constant_9')
    # Q=[20*0.1,20*0.2,20*0.3,20*0.4,20*0.5,20*0.6,20*0.7,20*0.8,20*0.9,20*1]
    Q=[20*-0.1,20*-0.2,20*-0.3,20*-0.4,20*-0.5,20*-0.6,20*-0.7,20*-0.8,20*-0.9,20*-1]
    Q_current_source = model.getComponentByKey('component_new_step_gen_9')
    P = [0.03 * -0.8 , 0.03 * -0.9 , 0.03 * -1.0]
    P_current_source = model.getComponentByKey('component_new_dc_current_source_1')
    for Idc in P:
        writewrite(['有功', 'Idc=%.4f'%Idc])
        P_current_source.args['Im'] = Idc
        for Iq in Q:
            writewrite(['无功', 'Iq=%.4f' %Iq])
            Q_current_source.args['V0'] = Iq
            Q_current_source.args['V1'] = Iq
            for voltage in V:
                writewrite(['电压', voltage])
                V_constant.args['Value'] = voltage
                for frequency in f:
                    f_source_d.args['F']['source'] = str(frequency)
                    f_source_q.args['F']['source'] = str(frequency)
                    f_FFT_vd.args['BaseFrequency']['source'] = str(frequency)
                    f_FFT_id.args['BaseFrequency']['source'] = str(frequency)
                    f_FFT_vq.args['BaseFrequency']['source'] = str(frequency)
                    f_FFT_iq.args['BaseFrequency']['source'] = str(frequency)
                    # start = time.time()
                    dq = 'd'
                    if dq == 'd':
                        f_source_d.args['Mag'] = 1
                        f_source_q.args['Mag'] = 0
                        [Ydd,Yqd,phdd,phqd,i]=run_cloudpss_d_axis(i)
                    dq = 'q'
                    if dq == 'q':
                        f_source_d.args['Mag'] = 0
                        f_source_q.args['Mag'] = 1
                        [Ydq,Yqq,phdq,phqq,i]=run_cloudpss_q_axis(i)
                        output=[Ydd,Ydq,Yqd,Yqq,phdd,phdq,phqd,phqq]
                        output_list.append([Ydd,Ydq,Yqd,Yqq,phdd,phdq,phqd,phqq])
                        print(output,'\n时间：',time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                        log(Idc, Iq, voltage, frequency)
                    # end = time.time()
                    # print("使用时间", end - start,'s')
                    # time.sleep(1)
                    output.insert(0, frequency)
                    writewrite(output)
