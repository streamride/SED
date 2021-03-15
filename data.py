from torch.utils.data import DataLoader, Dataset
import librosa
import torch
import pandas as pd
import numpy as np


class AudioDataset(Dataset):
    def __init__(self, df, le, ohe, audio_data_path: str, augmentations = None):
        self.df = df
        self.le = le
        self.ohe = ohe
        self.audio_data_path = audio_data_path
        self.augmentations = augmentations
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        audio = self.df.loc[idx, 'filename']
        x, sr = librosa.load(self.audio_data_path / audio)
        
        
        target = self.df.loc[idx, 'target']
        target_label = self.le.transform([target])
        target_ohe = self.ohe.transform(np.array(target).reshape(-1,1))
        
        
        
        return torch.tensor(x), torch.tensor([target_label]).squeeze(), torch.tensor(target_ohe)