"""
Created on Wed Jul 26

@author: Thomas
"""
#import cv2
import threading
import time
import numpy as np
import pyrealsense2 as rs

class realsense_imu:
    def __init__(self,camera_name='Imu1'):
        
        self.camera_name=camera_name
        

        try:
            # Create a context object. This object owns the handles to all connected realsense devices
            self.pipeline = rs.pipeline()

            # Configure streams
            config = rs.config()
            config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200) #200/400
            config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250) #63/250
        
            self.config=config
            
        except Exception as e:
            print(e)
            pass  
        
        #self.frame_size=(res_w,res_h)

        self.data_thread=threading.Thread(target=self.worker_data_read)
#        self.file_thread=threading.Thread(target=self.worker_file_write);

        self.active_flag=True
        self.write_flag=False
        self.fps=0
        self.acc=[]
        self.gyro=[]
        self.acc_ts=[]
        self.gyro_ts=[]
#        self.depth=[]
        self.t0=time.time()
        self.capture_time=0
        self.data_thread.start()   
        
    def start(self):
        profile=self.pipeline.start(self.config)
        self.profile=profile
#        self.depth_scale=profile.get_device().first_depth_sensor().get_depth_scale()
#        depth_image = np.asanyarray(depth.get_data())
#        depth_image = depth_image * depth_scale
        
    def worker_data_read(self):
        """thread reader function"""
        while self.active_flag :
            if time.time()-self.t0<self.capture_time:
                try:
                    frames = self.pipeline.wait_for_frames()
                    acc,gyro,acc_ts,gyro_ts=self.parse_data(frames)
                    if(len(self.acc_ts) == 0 or self.acc_ts[-1] != acc_ts):
                        self.acc.append(acc)
                        self.acc_ts.append(acc_ts)
                    if(len(self.gyro_ts) == 0 or self.gyro_ts[-1] != gyro_ts):
                        self.gyro.append(gyro)
                        self.gyro_ts.append(gyro_ts)
                    self.fps=len(self.acc)/self.capture_time
                except Exception as e:
                    print(e)
                    pass
            else:
                time.sleep(0.01)
        return;

    def parse_data(self,frames):
        g_temp = frames[1].as_motion_frame().get_motion_data()
        a_temp = frames[0].as_motion_frame().get_motion_data()
        gyro=np.array([g_temp.x, g_temp.y, g_temp.z])
        accel=np.array([a_temp.x, a_temp.y, a_temp.z])
        accel_ts = frames[0].as_motion_frame().get_timestamp()
        gyro_ts = frames[1].as_motion_frame().get_timestamp()
        return accel, gyro, accel_ts, gyro_ts

    def start_capture_timed(self,record_time=1):
        self.t0=time.time()
        self.capture_time=record_time
        self.data=[]
        self.acc=[]
        self.acc_ts = []
        self.gyro_ts = []
        self.gyro=[]
        return True
    
    def kill(self):
        self.data=[]
        self.active_flag=False

class async_data_saver:
    def __init__(self,
                 imu_data=[],
                 gyro_data=[],
                 imu_ts=[],
                 gyro_ts=[],
                 path_out='./'):
        self.file_thread=threading.Thread(target=self.worker_file_write)
        self.imu_data=imu_data.copy()
        self.gyro_data=gyro_data.copy()
        self.imu_ts=imu_ts.copy()
        self.gyro_ts=gyro_ts.copy()
        self.path_out=path_out
        self.file_thread.start()   

    def worker_file_write(self):
        """thread reader function"""
        np.savez(self.path_out,
                 imu_data=self.imu_data,
                 gyro_data=self.gyro_data,
                 imu_ts = self.imu_ts,
                 gyro_ts = self.gyro_ts)
        self.imu_data=[]
        self.gyro_data=[]
        self.imu_ts=[]
        self.gyro_ts=[]
        return;

