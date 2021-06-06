import os, sys, pickle, random
import numpy as np
import librosa
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as Data
from torch.utils.data import DataLoader, Dataset

# def audio_norm(data):
#     max_data = np.max(data)
#     min_data = np.min(data)
#     data = (data-min_data)/(max_data-min_data+1e-6)
#     return data-0.5
def audio_norm(data):
    max_data = np.max(np.absolute(data)) 
    return data/(max_data+1e-6)*0.5
    
def get_feature(file_name,hop_len):
    y,sr = librosa.core.load(file_name, sr=48000)
    mfcc = librosa.feature.mfcc( y = y, sr = sr, hop_length = hop_len, n_mfcc= 20 )
    spectral_center = librosa.feature.spectral_centroid(y = y, sr = sr, hop_length = hop_len)
    chroma = librosa.feature.chroma_stft(y = y, sr = sr, hop_length = hop_len)
    spectral_contrast = librosa.feature.spectral_contrast( y = y, sr = sr, hop_length=hop_len)
    return mfcc, spectral_center, chroma, spectral_contrast

def load_split_file(path, ratio): # load pic and save as pickle
    train_ratio = ratio
    train_data_x = []
    train_data_y = []
    test_data_x = []
    test_data_y = []
    dirs = os.listdir(path)
    for i, document in enumerate(dirs):
        if(document!='.DS_Store'):
            files = os.listdir(f'{path}/{document}')
            record_label = int(document.split('_')[0])
            res = []
            for file_name in files:
                if(file_name!='.DS_Store'):
                    fname = f'{path}/{document}/{file_name}'
                    #hop_len = 512
                    # y,sr = librosa.core.load(fname, sr=48000)
                    # file_mfcc = librosa.feature.mfcc( y = y, sr = sr, hop_length = hop_len, n_mfcc= 20 )
                    res.append(fname)
            random.shuffle(res)
            num = train_ratio * len(res)
            for d, file_record in enumerate(res):
                if d < num:
                    train_data_x.append(file_record)
                    train_data_y.append(record_label)
                else:
                    test_data_x.append(file_record)
                    test_data_y.append(record_label)
    # with open(f'{path}/dataset.pickle', 'wb') as file:
    #     pickle.dump([data_x, data_y], file)
    return np.array(train_data_x), np.array(train_data_y),np.array(test_data_x),np.array(test_data_y)

class AudioDataset(Dataset):
    def __init__(self, data, label, sr=48000, hop_len=512):
        self.data = data
        self.label = label
        self.sr = sr
        self.hop_len = hop_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_name = self.data[index]
        all_feature = np.zeros((469,40),dtype = np.float32)
        mfcc, spectral_center, chroma, spectral_contrast = get_feature(file_name, self.hop_len)
        all_feature[:,0:20] = mfcc.T
        all_feature[:,20:21] = spectral_center.T
        all_feature[:,21:33] = chroma.T
        all_feature[:,33:41] = spectral_contrast.T
        
        all_feature = all_feature[np.newaxis, :]
        target = torch.tensor(self.label[index], dtype = torch.long)
        
        return all_feature, target
