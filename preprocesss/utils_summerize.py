import cv2
import copy
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from scipy import signal

def read_vid(path):
  frame_list=[]
  # Create a video capture object, in this case we are reading the video from a file
  vid_capture = cv2.VideoCapture(path)
  fps = vid_capture.get(cv2.CAP_PROP_FPS)
  frame_count = int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))
  duration = frame_count/fps
  while(vid_capture.isOpened()):
    # vid_capture.read() methods returns a tuple, first element is a bool 
    # and the second is frame
    ret, frame = vid_capture.read()
    if frame is None:
      break
    frame_list.append(frame)
  return frame_list, fps, duration

def detect_gesture_radar(dat,len_resize,filter_len=30,SNR_thresh=40):
  dcube=np.fft.fft(dat-np.expand_dims(np.mean(dat,0),0),64,1)
  dcube[:,0,...]=0
  dcube=np.fft.fftshift(dcube,1)
  dcube=np.fft.fft(dcube,32,-1)

  noise=np.mean(np.abs(dcube[:,-10:,...]),(1,2,3,4))  # the highest doppler bins are mostly noise
  signal_raw=np.mean(np.abs(dcube[:,:,...]),(1,2,3,4))
  x_ori = np.linspace(0, len_resize-1, len(signal_raw))
  x_new = np.linspace(0, len_resize-1, len_resize)
  signal_resize = np.interp(x_new,x_ori,signal_raw)   # resize to match the size of video len

  vec_dcube=np.convolve(signal_resize,np.ones(filter_len)/filter_len,'same')

  radar_detect=vec_dcube>np.mean(noise)+SNR_thresh
  radar_detect[:2] = False
  radar_detect[-2:] = False
  start_idxs_radar=np.where(radar_detect[1:]&~radar_detect[:-1])[0]
  end_idxs_radar=np.where(~radar_detect[1:]&radar_detect[:-1])[0]

  if len(start_idxs_radar)>len(end_idxs_radar):
    start_idxs_radar = start_idxs_radar[:-1]
  elif len(end_idxs_radar)>len(start_idxs_radar):
    end_idxs_radar = end_idxs_radar[1:]
  num_gestures=len(start_idxs_radar)

  data=[[i,start_idxs_radar[i],end_idxs_radar[i]] for i in range(num_gestures)]
  des=pd.DataFrame(columns=['Gesture_idx','Radar_start_idx','Radar_end_idx'],data=data)

  if num_gestures<5:
    signal_norm = vec_dcube/vec_dcube.std()
    peaks, properties = signal.find_peaks(signal_norm[:-30], prominence=1.5, width=20)
    radar_detect = np.zeros(len(vec_dcube), dtype='bool')
    for i in range(len(peaks)):
      radar_detect[np.max([peaks[i]-int(properties['widths'][i]/2),2]):np.min([peaks[i]+int(properties['widths'][i]/2),len(vec_dcube)-2])] = True

  return num_gestures, des, (vec_dcube, radar_detect)

def detect_gesture_pc(pcloud,len_resize,filtlen=30):
  pc=pd.concat(pcloud,axis=0)
  x, y, z = [], [], []
  x_buf, y_buf, z_buf = pc['x'].min(), pc['y'].min(), pc['z'].min()
  for p in pcloud:
    if len(p)>0:
      x.append(p['x'].mean()), y.append(p['y'].mean()), z.append(p['z'].mean())
      x_buf, y_buf, z_buf = p['x'].mean(), p['y'].mean(), p['z'].mean()
    else:
      x.append(x_buf), y.append(y_buf), z.append(z_buf)
  x_ori = np.linspace(0, len_resize-1, len(x))
  x_new = np.linspace(0, len_resize-1, len_resize)
  x, y, z = np.interp(x_new,x_ori,x), np.interp(x_new,x_ori,y), np.interp(x_new,x_ori,z)
  x = np.convolve(x,np.ones((filtlen))/filtlen,'same')
  y = np.convolve(y,np.ones((filtlen))/filtlen,'same')
  z = np.convolve(z,np.ones((filtlen))/filtlen,'same')
  x = (x-np.nanmean(x))/np.nanstd(x)
  y = (y-np.nanmean(y))/np.nanstd(y)
  z = (z-np.nanmean(z))/np.nanstd(z)

  x_detect, y_detect, z_detect = x>-0.5, y>-0.5, z>-0.5

  x_detect[:2], y_detect[:2], z_detect[:2]  = False, False, False
  x_detect[-2:], y_detect[-2:], z_detect[-2:]  = False, False, False
  
  start_idxs_x, start_idxs_y, start_idxs_z = np.where(x_detect[1:]&~x_detect[:-1])[0], \
                                              np.where(y_detect[1:]&~y_detect[:-1])[0], \
                                              np.where(z_detect[1:]&~z_detect[:-1])[0]
  dim_sel = np.argmin([abs(len(start_idxs_x-12)), abs(len(start_idxs_y-12)), abs(len(start_idxs_z-12))])
  pc_vec    = [z, y, x]
  pc_detect = [z_detect, y_detect, x_detect]
  radar_detect = pc_detect[dim_sel]
  start_idxs_radar=np.where(radar_detect[1:]&~radar_detect[:-1])[0]
  end_idxs_radar=np.where(~radar_detect[1:]&radar_detect[:-1])[0]

  if len(start_idxs_radar)>len(end_idxs_radar):
    start_idxs_radar = start_idxs_radar[:-1]
  elif len(end_idxs_radar)>len(start_idxs_radar):
    end_idxs_radar = end_idxs_radar[1:]
  num_gestures=len(start_idxs_radar)

  data=[[i,start_idxs_radar[i],end_idxs_radar[i]] for i in range(num_gestures)]
  des=pd.DataFrame(columns=['Gesture_idx','Radar_start_idx','Radar_end_idx'],data=data)
  return num_gestures, des, (pc_vec[dim_sel], pc_detect[dim_sel])


def detect_gesture_depth(depth_frame_list, len_resize, num_pix_min=0.01):
  num_pix=[np.mean(frame_tmp>0) for frame_tmp in depth_frame_list]
  num_pix_filt = signal.medfilt(num_pix, kernel_size=15)

  x_ori = np.linspace(0, len_resize-1, len(num_pix_filt))
  x_new = np.linspace(0, len_resize-1, len_resize)
  num_pix_filt = np.interp(x_new,x_ori,num_pix_filt)

  depth_detect=np.array([pix>num_pix_min for pix in num_pix_filt])
  depth_detect[:2] = False
  depth_detect[-2:] = False
  start_idxs_camera=np.where(depth_detect[1:]&~depth_detect[:-1])[0]
  end_idxs_camera=np.where(~depth_detect[1:]&depth_detect[:-1])[0]

  if len(start_idxs_camera)>len(end_idxs_camera):
    start_idxs_camera = start_idxs_camera[:-1]
  elif len(end_idxs_camera)>len(start_idxs_camera):
    end_idxs_camera = end_idxs_camera[1:]
  num_gestures=len(start_idxs_camera)
  
  data=[[i,start_idxs_camera[i],end_idxs_camera[i]] for i in range(num_gestures)]
  des=pd.DataFrame(columns=['Gesture_idx','Camera_start_idx','Camera_end_idx'],data=data)
  return num_gestures, des, (num_pix_filt, depth_detect)

def detect_gesture_camera(camera_frame_list, model, image_transforms, device, num_pix_min=0.01):
  output_array = []
  model.eval()
  (H,W) = (256, 256)
  for i in range(1+len(camera_frame_list)//16):
    if i<len(camera_frame_list)//16:
      img_batch = camera_frame_list[i*16:i*16+16]
    else:
      img_batch = camera_frame_list[i*16:i*16+(len(camera_frame_list)%16)]
    if len(img_batch)>0:
      img_batch = [image_transforms(Image.fromarray(img)) for img in img_batch] 
      img_batch = torch.stack(img_batch).to(device)
      logits = model(img_batch).detach().cpu()
      preds = F.softmax(logits, 1).argmax(1) * 255 # [h, w]
      for j in range(len(preds)):
        output_array.append(preds[j,:,:])
  output_last = output_array[-15:]    # output of last 0.5s (no-hand area)
  y_pix_last = np.array([np.where((np.array(frame_tmp,dtype='float')>0)==True)[0][0] for frame_tmp in output_last if (np.array(frame_tmp,dtype='float')>0).sum()>40])
  num_pix_last = np.array([np.mean(np.array(frame_tmp,dtype='float')>0) for frame_tmp in output_last])
  if (num_pix_last.mean()>num_pix_min) and (y_pix_last.mean()>H//2):   # to reject the area of resting hand in the edge of camera frame
    y_detect = round(y_pix_last.mean())-10
  else:
    y_detect = H
  num_pix=np.array([np.mean(np.array(frame_tmp[:y_detect,:],dtype='float')>0) for frame_tmp in output_array])
  num_pix_filt = signal.medfilt(num_pix, kernel_size=15)

  RGB_detect=np.array([pix>num_pix_min for pix in num_pix_filt])
  RGB_detect[:2] = False
  RGB_detect[-2:] = False
  start_idxs_camera =np.where(RGB_detect[1:]&~RGB_detect[:-1])[0]
  end_idxs_camera   =np.where(~RGB_detect[1:]&RGB_detect[:-1])[0]

  if len(start_idxs_camera)>len(end_idxs_camera):
    start_idxs_camera = start_idxs_camera[:-1]
  elif len(end_idxs_camera)>len(start_idxs_camera):
    end_idxs_camera = end_idxs_camera[1:]
  num_gestures=len(start_idxs_camera)

  data=[[i,start_idxs_camera[i],end_idxs_camera[i]] for i in range(num_gestures)]
  des=pd.DataFrame(columns=['Gesture_idx','Camera_start_idx','Camera_end_idx'],data=data)
  return num_gestures, des, (num_pix_filt, RGB_detect)




def summerize_data_RD_multiradar(dcube,
                                  pcloud,
                                  camera_data,
                                  vec_results,
                                  des,
                                  flag,
                                  window=65,
                                  fig_name='',
                                  show=True,
                                  aspect=None):
  
  ################ load results
  (vec_pc, vec_dcube, vec_depth, vec_RGB) = vec_results
  (des_pc, des_dcube, des_depth, des_RGB) = des
  (flag_depth, flag_pc) = flag
  
  depth_detect  = vec_depth[1]
  RGB_detect    = vec_RGB[1]
  dcube_detect  = vec_dcube[1]
  pc_detect     = vec_pc[1]
  camera_detect = depth_detect if flag_depth==True else RGB_detect
  radar_detect  = pc_detect if flag_pc==True else dcube_detect
  des_camera    = des_depth if flag_depth==True else des_RGB
  des_radar     = des_pc if flag_pc==True else des_dcube
  
  ###### Calculate radar_detect based on camera detection results
  sync_coeff = np.zeros(100)
  sync_coeff[:12] = np.array([2,2,2,2,2,2,1,1,1,1,0,0])
  camera_center_idx = np.array(round((des_camera['Camera_start_idx']+des_camera['Camera_end_idx'])/2), dtype='int')
  radar_center_fromCam = camera_center_idx - sync_coeff[:len(camera_center_idx)]  # Calculate radar_center_idx based on camera_center_idx
  adj_detect = np.zeros(len(camera_detect))
  for center_idx in radar_center_fromCam:
    detect_idx_start = int(np.max([center_idx-window//2,0]))
    detect_idx_end   = int(np.min([center_idx+window//2+1,len(adj_detect)]))
    adj_detect[detect_idx_start:detect_idx_end] = True

  ###### Interpolate (to match with radar data)
  radar_DT = np.log(np.mean(np.abs(dcube+1),(4,2,3))).T
  radar_RT = np.log(np.mean(np.abs(dcube[:,:,...]),(1,2,3))).T
  radar_DT = cv2.resize(radar_DT, (len(RGB_detect), radar_DT.shape[0]))
  radar_RT = cv2.resize(radar_RT, (len(RGB_detect), radar_RT.shape[0]))
  x_vec = np.linspace(0, len(adj_detect)-1, len(adj_detect))

  ## assign zero val
  radar_DT_cameradet  = copy.deepcopy(radar_DT)
  radar_DT_radardet   = copy.deepcopy(radar_DT)
  radar_DT_cameradet[:,np.where(adj_detect==0)[0]]  = np.nan
  radar_DT_radardet[:,np.where(radar_detect==0)[0]]    = np.nan

  ######
  vec_pc_norm     = (vec_pc[0]-np.nanmin(vec_pc[0]))/(np.nanmax(vec_pc[0])-np.nanmin(vec_pc[0]))
  vec_dcube_norm  = vec_dcube[0]/(np.max(vec_dcube[0])+1e-8)
  vec_depth_norm  = vec_depth[0]/(np.max(vec_depth[0])+1e-8)
  vec_RGB_norm    = vec_RGB[0]/(np.max(vec_RGB[0])+1e-8)

  pc_center_idx    = np.array(round((des_pc['Radar_start_idx']+des_pc['Radar_end_idx'])/2), dtype='int')
  dcube_center_idx = np.array(round((des_dcube['Radar_start_idx']+des_dcube['Radar_end_idx'])/2), dtype='int')
  depth_center_idx = np.array(round((des_depth['Camera_start_idx']+des_depth['Camera_end_idx'])/2), dtype='int')
  RGB_center_idx   = np.array(round((des_RGB['Camera_start_idx']+des_RGB['Camera_end_idx'])/2), dtype='int')
  vec_center_pc       = np.array([vec_pc_norm[idx] if idx in pc_center_idx else np.nan for idx in range(len(adj_detect))])
  vec_center_dcube    = np.array([vec_dcube_norm[idx] if idx in dcube_center_idx else np.nan for idx in range(len(adj_detect))])
  vec_center_depth    = np.array([vec_depth_norm[idx] if idx in depth_center_idx else np.nan for idx in range(len(adj_detect))])
  vec_center_RGB      = np.array([vec_RGB_norm[idx] if idx in RGB_center_idx else np.nan for idx in range(len(adj_detect))])

  ###### Plot
  # x: azimuth
  # z: elevation
  # y: depth
  filtlen=30
  pc=pd.concat(pcloud,axis=0)
  x=np.convolve(pc['x'],np.ones((filtlen))/filtlen,'same')
  y=np.convolve(pc['y'],np.ones((filtlen))/filtlen,'same')
  z=np.convolve(pc['z'],np.ones((filtlen))/filtlen,'same')
  v=np.convolve(pc['v'],np.ones((filtlen))/filtlen,'same')

  if show:
      plt.ion()
  else:
      plt.ioff()      
  fig, ax = plt.subplots(5, 2)
  fig.suptitle(fig_name)
  # ax[0,0].plot(pc['frame'],x,'-',label='x (Azimuth)')
  # ax[0,0].plot(pc['frame'],y,'-',label='y (Depth)')
  # ax[0,0].plot(pc['frame'],z,'-',label='z (Elevation)')
  # ax[0,0].set_xlabel('Time (Frame)')
  # ax[0,0].set_ylabel('Distance (m)')
  # ax[0,0].legend()
  #ax[0,0].set_xlim([3.8,16.2])
  #ax[0,0].set_ylim([3.8,16.2])
  #ax[0,0].title('Target Location')

  ax[0,0].plot(x_vec, vec_depth_norm,label='Normalized Hand Mask Area')
  ax[0,0].plot(x_vec, depth_detect,label='Gesture Detected')
  ax[0,0].plot(x_vec, vec_center_depth,'*', markerfacecolor='red', markersize=12)
  ax[0,0].set_xlabel('Frames')
  ax[0,0].set_ylabel('Detected')
  ax[0,0].set_title('Gesture Detection (Depth)')
  ax[0,0].legend(loc='lower right')

  ax[1,0].plot(x_vec, vec_RGB_norm,label='Normalized Hand Mask Area')
  ax[1,0].plot(x_vec, RGB_detect,label='Gesture Detected')
  ax[1,0].plot(x_vec, vec_center_RGB, '*', markerfacecolor='red', markersize=12)
  ax[1,0].set_xlabel('Frames')
  ax[1,0].set_ylabel('Detected')
  ax[1,0].set_title('Gesture Detection (RGB)')
  ax[1,0].legend(loc='lower right')

  ax[2,0].plot(x_vec, vec_pc_norm,label='Normalized Moving Hand SNR')
  ax[2,0].plot(x_vec, pc_detect,label='Gesture Detected')
  ax[2,0].plot(x_vec, adj_detect*0.95,label='Gesture Detected (Camera)')
  ax[2,0].plot(x_vec, vec_center_pc, '*', markerfacecolor='red', markersize=12)
  ax[2,0].set_xlabel('Frames')
  ax[2,0].set_ylabel('Detected')
  ax[2,0].set_title('Gesture Detection (Radar_PC)')
  ax[2,0].legend(loc='lower right')

  ax[3,0].plot(x_vec, vec_dcube_norm,label='Normalized Moving Hand SNR')
  ax[3,0].plot(x_vec, dcube_detect,label='Gesture Detected')
  ax[3,0].plot(x_vec, adj_detect*0.95,label='Gesture Detected (Camera)')
  ax[3,0].plot(x_vec, vec_center_dcube, '*', markerfacecolor='red', markersize=12)
  ax[3,0].set_xlabel('Frames')
  ax[3,0].set_ylabel('Detected')
  ax[3,0].set_title('Gesture Detection (Radar_Dcube)')
  ax[3,0].legend(loc='lower right')

  # ax[3,0].imshow(radar_RT,cmap='jet',aspect=15,vmin=7,vmax=8)
  # ax[3,0].set_xlabel('Time (Frames)')
  # ax[3,0].set_ylabel('Range (bins)')

  ax[0,1].imshow(radar_DT,cmap='jet',aspect=7,vmin=7,vmax=8.5)
  ax[0,1].set_xlabel('Time (Frames)')
  ax[0,1].set_ylabel('Doppler (bins)')
  
  ax[1,1].imshow(radar_DT_cameradet,cmap='jet',aspect=7,vmin=7,vmax=8.5)
  ax[1,1].set_xlabel('Time (Frames)')
  ax[1,1].set_ylabel('Doppler (bins)')
  
  ax[2,1].imshow(radar_DT_radardet,cmap='jet',aspect=7,vmin=7,vmax=8.5)
  ax[2,1].set_xlabel('Time (Frames)')
  ax[2,1].set_ylabel('Doppler (bins)')

  camera_frame=np.int32(camera_data)
  ax[3,1].imshow(camera_frame[...,::-1],aspect=1)
  ax[3,1].set_xticks([])
  ax[3,1].set_yticks([])

  adj_detect = adj_detect.astype(np.int)
  start_idxs_camera = np.where(adj_detect[1:]&~adj_detect[:-1])[0]
  end_idxs_camera   = np.where(~adj_detect[1:]&adj_detect[:-1])[0]
  if len(start_idxs_camera)<=2:
    ax[4,0].imshow(radar_DT,cmap='jet',aspect=7,vmin=7,vmax=8.5)
    ax[4,1].imshow(radar_DT,cmap='jet',aspect=7,vmin=7,vmax=8.5)
  else:
    idx1 = 10 if len(start_idxs_camera)>12 else -2
    idx2 = 11 if len(start_idxs_camera)>12 else -1
    ax[4,0].imshow(radar_DT[:,start_idxs_camera[idx1]:end_idxs_camera[idx1]],cmap='jet',aspect=7,vmin=7,vmax=8.5)
    ax[4,1].imshow(radar_DT[:,start_idxs_camera[idx2]:end_idxs_camera[idx2]],cmap='jet',aspect=7,vmin=7,vmax=8.5)

  fig.set_size_inches(30,20)

  # data save
  data_all = {}
  data_all['cam_depth'] = vec_depth_norm
  data_all['cam_RGB']   = vec_RGB_norm
  data_all['rad_PC']    = vec_pc_norm
  data_all['rad_dcube'] = vec_dcube_norm
  data_all['DT']        = radar_DT
  data_all['RT']        = radar_RT
  return fig, data_all

def snapshot_visualize(datapath_cam, datapath_rad, des_cam, des_rad, gesture_suc_both, window=60, gap=80):
    """
    window: window size of each snapshot (default: 60 (2s))
    gap: gap of adjacent shapshots in visualization
    """
    # Load
    dat_cam, _, _   = read_vid(datapath_cam)
    with open(datapath_rad, 'rb') as fr:
        dat_rad = pickle.load(fr)
        dat_rad_rng = dat_rad[0]
        dat_rad_Dop = dat_rad[2]
    # Concat all camera snapshots
    assert len(des_cam)==len(des_rad)==len(gesture_suc_both)
    len_snapshot = len(des_cam)
    H, W, C = dat_cam[0].shape
    dat_concat_cam = np.ones((H,len_snapshot*(W+2*gap),C))*255
    for idx in range(len(des_cam)):
      if gesture_suc_both[idx]:
        snapshot_cam = dat_cam[int(des_cam[idx])]
        dat_concat_cam[:,(idx*(W+2*gap))+(W+2*gap)//2-W//2:(idx*(W+2*gap))+(W+2*gap)//2+W//2,:] = snapshot_cam
    # Concat all radar Rng/Dop snapshots
    des_rad_rng = np.round(des_rad*dat_rad_rng.shape[1]/len(dat_cam))
    des_rad_Dop = np.round(des_rad*dat_rad_Dop.shape[1]/len(dat_cam))
    dat_concat_rad_rng = np.ones((W,len_snapshot*(W+2*gap)))*np.nan
    dat_concat_rad_Dop = np.ones((W,len_snapshot*(W+2*gap)))*np.nan
    wsize_rad_rng, wsize_rad_Dop   = int(np.round(window*dat_rad_rng.shape[1]/len(dat_cam))), int(np.round(window*dat_rad_Dop.shape[1]/len(dat_cam)))
    for idx in range(len(des_rad)):
        if gesture_suc_both[idx]:
          if des_rad_rng[idx]-wsize_rad_rng//2<0:
            snapshot_rng = dat_rad_rng[:,0:wsize_rad_rng]
            snapshot_Dop = dat_rad_Dop[:,0:wsize_rad_Dop]
          else:
            snapshot_rng = dat_rad_rng[:,int(des_rad_rng[idx])-wsize_rad_rng//2:int(des_rad_rng[idx])+wsize_rad_rng//2]
            snapshot_Dop = dat_rad_Dop[:,int(des_rad_Dop[idx])-wsize_rad_Dop//2:int(des_rad_Dop[idx])+wsize_rad_Dop//2]
          dat_concat_rad_rng[:,(idx*(W+2*gap))+(W+2*gap)//2-W//2:(idx*(W+2*gap))+(W+2*gap)//2+W//2] = cv2.resize(snapshot_rng, (W,W))
          dat_concat_rad_Dop[:,(idx*(W+2*gap))+(W+2*gap)//2-W//2:(idx*(W+2*gap))+(W+2*gap)//2+W//2] = cv2.resize(snapshot_Dop, (W,W))
    
    # Plot & Save
    Time_id = datapath_cam.split('/')[-1][:-4]
    fig, ax = plt.subplots(5, 1)
    fig.suptitle(Time_id, fontsize=35)
    fig.set_size_inches(30,20)
    ax[0].imshow(cv2.resize(dat_rad_rng, (2000,500)),cmap='jet',aspect=0.5)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].imshow(cv2.resize(dat_rad_Dop, (2000,500)),cmap='jet',aspect=0.5)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[2].imshow(dat_concat_rad_rng,cmap='jet',aspect=1)
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[3].imshow(dat_concat_rad_Dop,cmap='jet',aspect=1)
    ax[3].set_xticks([])
    ax[3].set_yticks([])  
    dat_concat_cam=np.int32(dat_concat_cam)
    ax[4].imshow(dat_concat_cam[...,::-1],aspect=1)
    ax[4].set_xticks([])
    ax[4].set_yticks([])
    return fig

def snapshot_save(save_path, episode_path, read_success, datapath_cam, datapath_rad, 
                  center_cam, center_rad, gesture_suc_both, mode_rad, window=60, window_spare=20):
  """
  - window: window size of snapshot (default: 60 (2s))
  - window_spare: spare for window (actual crop size: window+window_spare)
  """
  # Load
  split_fname     = episode_path.split('/')
  Subject_id      = split_fname[6]
  Enviroment_id   = split_fname[-4]
  Gesture_id      = split_fname[-3]
  Time_id         = split_fname[-1][:-4]
  dat_cam, _, _   = read_vid(datapath_cam)
  with open(datapath_rad, 'rb') as fr:
      dat_rad = pickle.load(fr)
  des_episode = pd.DataFrame(columns=['Episode','Order','Subject','Enviroment','Gesture','Remark_Episode','Remark_Snapshot'])

  for idx in range(len(gesture_suc_both)):
    if gesture_suc_both[idx]:
      des_episode.loc[idx] = [Time_id,idx,Subject_id,Enviroment_id,Gesture_id,read_success,None]
      # Cam
      center_snapshot_cam = int(center_cam[idx])
      center_snapshot_rad = int(center_rad[idx])
      snapshot_cam = dat_cam[np.max([center_snapshot_cam-(window//2+window_spare//2),0])
                              :np.min([center_snapshot_cam+(window//2+window_spare//2),len(dat_cam)])]
      snapshot_cam = [cv2.resize(snapshot_img, (256,256)) for snapshot_img in snapshot_cam]
      save_numpy_to_video(snapshot_cam, (256,256), save_path+f'{Time_id}-{idx}-cam.mp4')

      # Rad
      for i_mode_radar in range((len(dat_rad))):
        dat_rad_sel = dat_rad[i_mode_radar]
        model_sel = mode_rad[i_mode_radar]
        center_snapshot_rad_adjust = int(np.round(center_snapshot_rad*dat_rad_sel.shape[-1]/len(dat_cam)))
        window_adjust = int(np.round(window*dat_rad_sel.shape[-1]/len(dat_cam)))
        window_spare_adjust = int(np.round(window_spare*dat_rad_sel.shape[-1]/len(dat_cam)))
        if 'channel' in model_sel:
          snapshot_rad_sel = np.ones((dat_rad_sel.shape[0],dat_rad_sel.shape[1],dat_rad_sel.shape[2],2*(window_adjust//2+window_spare_adjust//2)))*9999
          snapshot_rad_temp  = dat_rad_sel[:,:,:,np.max([center_snapshot_rad_adjust-(window_adjust//2+window_spare_adjust//2),0])
                                          :np.min([center_snapshot_rad_adjust+(window_adjust//2+window_spare_adjust//2),dat_rad_sel.shape[-1]])]
          if idx ==0:
            snapshot_rad_sel[:,:,:,-1*snapshot_rad_temp.shape[-1]:] = snapshot_rad_temp
            if snapshot_rad_sel.shape[-1] != snapshot_rad_temp.shape[-1]:
              des_episode.loc[idx]['Remark_Snapshot'] = 'Miss'
          elif idx == len(gesture_suc_both)-1:
            snapshot_rad_sel[:,:,:,:snapshot_rad_temp.shape[-1]] = snapshot_rad_temp
            if snapshot_rad_sel.shape[-1] != snapshot_rad_temp.shape[-1]:
              des_episode.loc[idx]['Remark_Snapshot'] = 'Miss'
          else:
            snapshot_rad_sel = snapshot_rad_temp
          snapshot_rad_sel = snapshot_rad_sel.transpose(0,3,1,2).reshape((128,snapshot_rad_sel.shape[-1],-1))
          snapshot_rad_sel = cv2.resize(snapshot_rad_sel, (512,128))
        else:
          snapshot_rad_sel = np.ones((dat_rad_sel.shape[0],2*(window_adjust//2+window_spare_adjust//2)))*9999
          snapshot_rad_temp = dat_rad_sel[:,np.max([center_snapshot_rad_adjust-(window_adjust//2+window_spare_adjust//2),0])
                                            :np.min([center_snapshot_rad_adjust+(window_adjust//2+window_spare_adjust//2),dat_rad_sel.shape[-1]])]
          if idx ==0:
            snapshot_rad_sel[:,-1*snapshot_rad_temp.shape[-1]:] = snapshot_rad_temp
            if snapshot_rad_sel.shape[-1] != snapshot_rad_temp.shape[-1]:
              des_episode.loc[idx]['Remark_Snapshot'] = 'Miss'
          elif idx == len(gesture_suc_both)-1:
            snapshot_rad_sel[:,:snapshot_rad_temp.shape[-1]] = snapshot_rad_temp
            if snapshot_rad_sel.shape[-1] != snapshot_rad_temp.shape[-1]:
              des_episode.loc[idx]['Remark_Snapshot'] = 'Miss'
          else:
            snapshot_rad_sel = snapshot_rad_temp
          snapshot_rad_sel = cv2.resize(snapshot_rad_sel, (512,128))
        np.save(save_path+f'{Time_id}-{idx}-rad-{model_sel}.npy',snapshot_rad_sel)
  return des_episode

def save_numpy_to_video(vid, size, path=None):
  fps = 30.
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  size = (size[0], size[1])
  videoWrite = cv2.VideoWriter(path, fourcc, fps, size)
  for i in range(len(vid)):
      videoWrite.write(cv2.resize(vid[i], size))
  videoWrite.release()
  return