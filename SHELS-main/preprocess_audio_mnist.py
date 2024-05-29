import numpy as np
import glob
import os
import sys
import scipy.io.wavfile as wavf
import scipy.signal
import json
import librosa
import multiprocessing
import argparse
import torch


def preprocess_data(src_path, meta_data_path):
    print("processing {}".format(src_path))
    print(src_path)

    classes = [0,1,2,3,4,5,6,7,8,9]


    if os.path.exists(r'/home/gridsan/aejilemele/LIS/SHELS-main/processed_audio_dataset') and os.path.exists(r'/home/gridsan/aejilemele/LIS/SHELS-main/processed_audio_targets') :
        Data = torch.load(r'/home/gridsan/aejilemele/LIS/SHELS-main/processed_audio_dataset')
        targets = torch.load(r'/home/gridsan/aejilemele/LIS/SHELS-main/processed_audio_targets')
        return Data, targets, classes
    
    metaData = json.load(open(meta_data_path))
    Data = []
    targets = []   
    for root, dirs, files in os.walk(src_path):
        counter = 0
        for filepath in files:
            digit, person, repition = filepath.rstrip(".wav").split("/")[-1].split("_")
            # digit, person, repition = files[idx].rstrip(".wav").split("/")[-1].split("_")
            fs, data = wavf.read(os.path.join(root , filepath))
            # fs, data = wavf.read(os.path.join(root , files[idx]))
            data = librosa.core.resample(y=data.astype(np.float32), orig_sr=fs, target_sr=8000, res_type="scipy")
            
            if len(data) > 8000:
                raise ValueError("data length cannot exceed padding length.")
            
            elif len(data) < 8000:# pad the audio data with zeros to ensure same size across samples
                # randomly embed that audio data within the 8000 dimension vector
                embedded_data = np.zeros(8000)
                offset = np.random.randint(low = 0, high = 8000 - len(data))
                embedded_data[offset:offset+len(data)] = data
            
            elif len(data) == 8000:
                # nothing to do here
                embedded_data = data

            f, t, Zxx = scipy.signal.stft(embedded_data, 8000, nperseg = 455, noverlap = 420, window='hann')
            # get amplitude
            Zxx = np.abs(Zxx[0:227, 2:-1])
            Zxx = np.atleast_3d(Zxx).transpose(2,0,1)
            # convert to decibel
            Zxx = librosa.amplitude_to_db(Zxx, ref = np.max)
            Zxx = torch.from_numpy(Zxx)
            target = torch.as_tensor(int(digit))
            Data.append(Zxx)
            targets.append(target)
        # else:
        #     break
        
    Data = torch.stack(Data)

    assert Data.shape == (30000, 1, 227, 227), f'data size{Data.size()}'
    targets = torch.stack(targets)
    torch.save(Data, r'/home/gridsan/aejilemele/LIS/SHELS-main/processed_audio_dataset')
    torch.save(targets, r'/home/gridsan/aejilemele/LIS/SHELS-main/processed_audio_targets')
    return Data, targets, classes
    
