"""
Created on Mon Jul  5 18:21:10 2021

@author: Soheil
"""
#import cv2
import threading
import time
import numpy as np
import pyrealsense2 as rs
import pickle

class realsense_camera:
    def __init__(self,camera_name='Camera1',
                 res_w=640, #640
                 res_h=480, #480s
                 fps=30):
        
        self.camera_name=camera_name
        

        try:
            # Create a context object. This object owns the handles to all connected realsense devices
            context=rs.context()
            print(context.query_devices()[0])
            
            #Thomas - first option controls hardware sync mode (0 = normal, 1 = master, 3 = full slave, 4 = gensync (arbitrary frame rate)), second option allows greater queueing of frames instead of dropping

            context.query_devices()[0].first_depth_sensor().set_option(rs.option.inter_cam_sync_mode, 0)
            context.query_devices()[0].first_depth_sensor().set_option(rs.option.frames_queue_size, 2)
            self.pipeline = rs.pipeline(context)
                        
            # Configure streams
            #Thomas - allow for pixel-pixel allignment between streams
            self.align = rs.align(rs.stream.color)
            config = rs.config()
            config.enable_stream(rs.stream.depth, res_w, res_h, rs.format.z16, fps)
            config.enable_stream(rs.stream.color, res_w, res_h, rs.format.rgb8, fps)
            
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
        self.data=[]
        self.frames=[]
        self.frames_ts=[]
        self.depth=[]
        self.depth_ts=[]
        self.intr=[]
        self.extr=[]
        self.t0=time.time()
        self.capture_time=0
        self.data_thread.start()
        
    def start(self):
        profile=self.pipeline.start(self.config)
        self.profile=profile
        self.depth_scale=profile.get_device().first_depth_sensor().get_depth_scale()
        #T - record intrinsics and extrinsics of the sensors (relative locations, properties (i.e focal length of lens))
        self.rgb_intr=profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.depth_intr=profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        self.rgb_depth_extr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to(profile.get_stream(rs.stream.depth).as_video_stream_profile())

#        depth_image = np.asanyarray(depth.get_data())
#        depth_image = depth_image * depth_scale
        
    def worker_data_read(self):
        """thread reader function"""
        while self.active_flag :
            
            if time.time()-self.t0<self.capture_time:
                try:
                    frames = self.pipeline.wait_for_frames()
                    
                    # depth = frames.get_depth_frame()
                    #if not depth: continue
                    #if(self.data == []):
                    #    print(frames)
                    #    self.data == frames
                    #else:
                    #    self.data.append(frames)
                    #    print(np.size(self.data))
                    #self.data.append(self.parse_data(frames))
                    
                    #print("1")
                    rgb,depth,rgb_ts,depth_ts=self.parse_data(frames)
                    if(len(self.frames_ts) == 0 or self.frames_ts[-1] != rgb_ts):
                        self.frames.append(rgb)
                        self.frames_ts.append(rgb_ts)
                    if(len(self.depth_ts) == 0 or self.depth_ts[-1] != depth_ts):
                        self.depth.append(depth)
                        self.depth_ts.append(depth_ts)
                    self.fps=len(self.data)/self.capture_time
                except Exception as e:
                    print(e)
                    print('Camera exception!')
                    pass
            else:
                time.sleep(0.01)
        return;

    def parse_data(self, frames, depth_scale=-1):
        #T - frame allignment is actually done here, as frames are received
        aligned_frames = self.align.process(frames)
        depth=np.array(aligned_frames[0].get_data())
        depth_ts = aligned_frames[0].get_timestamp()
        if depth_scale<0:
            depth_scale=self.depth_scale
        depth = np.uint8(np.clip(depth * depth_scale,a_min=0,a_max=1.28)*200)

        frames=np.array(aligned_frames[1].get_data())
        frames_ts = aligned_frames[1].get_timestamp()
        return frames, depth, frames_ts, depth_ts

    def start_capture_timed(self,record_time=1):
        self.t0=time.time()
        self.capture_time=record_time
        self.data=[]
        self.frames=[]
        self.frames_ts=[]
        self.depth_ts=[]
        self.depth=[]
        return True
    
    def kill(self):
        self.data=[]
        self.active_flag=False

        
def get_bird_eye_view(file_path='/media/jetson/SD/controled/2022_May_17/2022May12-0001/radar/20220512000400.npz'):
    with np.load(file_path,allow_pickle=True) as data:
    #        print(data.files)
            radar_rawdata = data['radar_data']
            camera_frame = data['camera_data'][...,:3]
    plt.figure(figsize=(20,20))
    imagergb=camera_frame[0,...,::-1]
    plt.imshow(imagergb)
    srcpts = np.float32([[400,700],[1700,720],[1200,810],[1200,610]])
    destpts = np.float32([[0,1000],[1000,1000],[500,500],[500,1500]])
    #applying PerspectiveTransform() function to transform the perspective of the given source image to the corresponding points in the destination image
    resmatrix = cv2.getPerspectiveTransform(srcpts, destpts)
    #applying warpPerspective() function to display the transformed image
    resultimage = cv2.warpPerspective(imagergb, resmatrix, (1000, 2000))
    plt.figure(figsize=(20,20))
    plt.imshow(resultimage)
    plt.gca().invert_yaxis()

class async_data_saver:
    def __init__(self,radar_data=[],
                 camera_data=[],
                 depth_data=[],
                 imu_data=[],
                 gyro_data=[],
                 path_out='./'):
        self.file_thread=threading.Thread(target=self.worker_file_write);
        self.radar_data=radar_data.copy()
        self.camera_data=camera_data.copy()
        self.depth_data=depth_data.copy()
        self.imu_data=imu_data.copy()
        self.gyro_data=gyro_data.copy()
        self.path_out=path_out
        self.file_thread.start()   

    def worker_file_write(self):
        """thread reader function"""
        np.savez(self.path_out,
                 radar_data=self.radar_data,
                 camera_data=self.camera_data,
                 depth_data=self.depth_data,
                 imu_data=self.imu_data,
                 gyro_data=self.gyro_data,
                )
        self.radar_data=[]
        self.camera_data=[]
        self.depth_data=[]

        return;

class async_timestamp_saver:
    def __init__(self,
                 ts=[],
                 intr=[],
                 extr=[],
                 path_out='./'):
        self.file_thread=threading.Thread(target=self.worker_file_write);
        self.ts=ts
        self.intr = intr,
        self.extr = extr,
        self.path_out=path_out
        self.file_thread.start()   

    def worker_file_write(self):
        """thread reader function"""
        np.savez(self.path_out,
                 intrinsics = self.intr,
                 extrinsics = self.extr,
                 ts=self.ts)
        self.ts=[]

        return;
        
class async_video_saver:
    def __init__(self,res_w=1920,res_h=1080,fps=30,data=[],path_out='./'):
        frame_size=(res_w,res_h)
        self.file_thread=threading.Thread(target=self.worker_file_write);
        self.data=data.copy()
        self.path_out=path_out
        self.video_writer=cv2.VideoWriter(path_out,
                                    cv2.VideoWriter_fourcc('M','J','P','G'),
                                    fps,
                                    frame_size)
        self.file_thread.start()   
        
    def worker_file_write(self):
        """thread reader function"""
        for frame in self.data:
            self.video_writer.write(frame)
        self.video_writer.release()
        print("Video saver done")
        self.data=[]
        return;

