"""
Created on Mon Jul  5 18:21:10 2021

@author: Soheil
"""


import numpy as np
import pandas as pd
from .radar_capture_utils import DCA1000
import matplotlib.pyplot as plt
import time


def test_radar(dca,radar,config_file_name,capture_len=2):
    print(f'Testing. Please wait {capture_len*2} seconds...')
    radar.load_config(config_file_name,print_outputs=False)
    time.sleep(capture_len)
    dca.data=[]
    radar.load_config(config_file_name,print_outputs=False)
    time.sleep(capture_len)
    if len(dca.data)>0 :
        print(f'{radar.radar_name} test success!')
        print(f'Data len:{len(dca.data)}')
        return True
    else:
        print(f'{radar.radar_name} test failed!')
        print(f'Data len:{len(dca.data)}')
        return False


def test_camera(cam,capture_len):
    cam.frames=[]
    cam.start_capture_timed(record_time=capture_len)
    time.sleep(capture_len+1)
    data_len=len(cam.frames)
    if data_len>0:
        print('camera capture success!')
        plt.figure()
        camera_frame=np.stack(cam.frames,0)[...] #::-1 to swap %colors
        plt.imshow(np.uint8(np.mean(camera_frame,0)))
        plt.figure()
        depth_frame=np.stack(cam.depth,0)
        plt.imshow(np.uint8(np.mean(depth_frame,0)))
        return True
    
def test_imu(imu, capture_len):
    imu.frames=[]
    imu.start_capture_timed(record_time=1)
    time.sleep(capture_len+.5)
    data_len=len(imu.acc)

    acc = np.array(imu.acc)
    gyro = np.array(imu.gyro)
    print(acc)
    print(acc.shape)

    if data_len>0:
        print('imu capture success!')
        print(data_len)
        plt.figure()
        plt.plot(acc[:,0])
        plt.plot(acc[:,1])
        plt.plot(acc[:,2])
        plt.figure()
        plt.plot(gyro[:,0])
        plt.plot(gyro[:,1])
        plt.plot(gyro[:,2])
        return True
