"""
- Preprocess mmWave raw data into micro-Doppler

"""
import numpy as np
import pandas as pd
from radarCaptureUtils import DCA1000
import glob
import cv2
import pickle 
import os
import scipy
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from tqdm import tqdm
# Custom
from utils_summerize import *

# In[]
process_params = {
    'STFT_window_size': 64,
    'STFT_overlap_ratio': 1-1/8,
    'STFT_nFFT': 128,
}

# In[]
flag_save = False
raw_data_path='/data/datasets/internal/gesture/raw_capture/'
save_path='/data/datasets/internal/gesture/preprocessed_v2/'
os.makedirs(save_path, exist_ok=True)

fname_list1=[filename for filename in sorted(glob.glob(raw_data_path+'/*/Collection*/Environment*/Gesture*/radar/*.npz'), key=os.path.getmtime)]
fname_list2=[filename for filename in sorted(glob.glob(raw_data_path+'/*/Capture*/Environment*/Gesture*/radar/*.npz'), key=os.path.getmtime)]
fname_list = fname_list1 + fname_list2

# In[]
des_list=[]
episode_list=[]
for radar_filename in tqdm(fname_list[:]):
    preprocessed_save = []
    ## Raw Data Load
    # if '20231206141833' in radar_filename:
    split_fname=radar_filename.split('/')
    Collection_id=split_fname[5]
    Subject_id=split_fname[6]
    Enviroment_id=split_fname[-4]
    Gesture_id=split_fname[-3]
    Time_id=split_fname[-1][:-4]
    exp_des=pd.DataFrame(columns=['Time','Subject','Enviroment','Gesture','Read_Success'],data=[[Time_id,Subject_id,Enviroment_id,Gesture_id,True]])
    depth_filename='/'.join(split_fname[:-2]+['depth']+[split_fname[-1][:-4]+'.avi'])
    camera_filename='/'.join(split_fname[:-2]+['camera']+[split_fname[-1][:-4]+'.avi'])
    try:
        with np.load(radar_filename) as f:
            rawfile = f['radar_data']
            camera_frame=f['camera_frame'][0,1,...]
        dat,pcloud,info=DCA1000.decode_data(rawfile,4)
        dat=dat.reshape((-1,64,3,4,32))
    except:
        print(f'File Read Error! {Time_id}')
        exp_des['Read_Success'] = 'Readerror'
        episode_list.append(exp_des)
        continue

    depth_frame_list, depth_fps, depth_duration      = read_vid(depth_filename)
    camera_frame_list, camera_fps, camera_duration   = read_vid(camera_filename)

    ## Radar pre-processing
    # De-clutter
    dcube  = dat - np.expand_dims(dat.mean(axis=1),axis=1)  # (Frame, Slow Time, Tx, Rx, Fast Time)
    N_F,N_st,N_tx,N_rx,N_ft = dcube.shape
    # Rng Processing (Skip for now)
    dcube_R  = np.fft.fft(dcube, n=process_params['STFT_nFFT'], axis=4)
    dcube_R  = dcube_R.reshape((-1,N_tx,N_rx,process_params['STFT_nFFT']))   # stack in time (Frame x Slow Time)
    mmWave_R = 10*np.log(abs(dcube_R).mean((1,2))+1e-9).transpose()  # Log transform & Non-coherent Integ.  
    # Azimuth
    dcube_A = np.fft.fft(dcube, n=process_params['STFT_nFFT']//2, axis=3)
    dcube_A  = dcube_A.reshape((-1,N_tx,process_params['STFT_nFFT'],N_ft))   # stack in time (Frame x Slow Time)
    mmWave_A = 10*np.log(abs(dcube_A).mean((1,3))+1e-9).transpose()  # Log transform & Non-coherent Integ.
    # Elevation
    dcube_E = np.fft.fft(dcube, n=process_params['STFT_nFFT']//2, axis=2)
    dcube_E = np.fft.fftshift(dcube_E, axes=(2))
    dcube_E  = dcube_E.reshape((-1,process_params['STFT_nFFT']//2,N_rx,N_ft))   # stack in time (Frame x Slow Time)
    mmWave_E = 10*np.log(abs(dcube_E).mean((2,3))+1e-9).transpose()  # Log transform & Non-coherent Integ.
    # Dop
    dcube_D = np.fft.fft(dcube, n=process_params['STFT_nFFT'], axis=1)
    dcube_D = np.fft.fftshift(dcube_D, axes=(1))
    mmWave_D  = 10*np.log(abs(dcube_D).mean((2,3,4))+1e-9).transpose()  # Log transform & Non-coherent Integ.
    preprocessed_save.append(mmWave_R)
    preprocessed_save.append(mmWave_A)
    preprocessed_save.append(mmWave_E)
    preprocessed_save.append(mmWave_D)
    # micro-Doppler
    dcube_mD = dcube.reshape((-1,3,4,dcube.shape[-1]))
    for scale in [1/4, 1/2, 1, 2]:
        _, _, Zxx = scipy.signal.stft(dcube_mD, 
                                        nfft=process_params['STFT_nFFT'],
                                        nperseg=process_params['STFT_window_size']*scale,
                                        noverlap=int(process_params['STFT_window_size']*process_params['STFT_overlap_ratio']*scale),
                                        window='hamming',return_onesided=False,axis=0)
        Zxx=np.fft.fftshift(Zxx,0)
        mmWave_uD = 10*np.log(abs(Zxx).mean((1,2,3))+1e-9)      # Log transform & Non-coherent Integ.
        preprocessed_save.append(mmWave_uD)



    ## Save Preprocessed data
    if flag_save:
        file_path = save_path + Time_id + '.pkl'
        with open(file_path, 'wb') as fp:
            pickle.dump(preprocessed_save, fp)
        # scipy.io.savemat('/workspace/temp.mat', {'uD_quad': preprocessed_save[4],
        #                                         'uD_half': preprocessed_save[5],
        #                                         'uD': preprocessed_save[6],
        #                                         'uD_double': preprocessed_save[7]})