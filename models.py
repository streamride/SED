import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from efficientnet_pytorch import EfficientNet
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

from sklearn.model_selection import train_test_split
from sklearn import metrics
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from pathlib import Path

import soundfile
from IPython.display import Audio as display_audio
import librosa

def interpolate(x, ratio):
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output, frames_num):
    pad = framewise_output[:, -1:, :].repeat(1, frames_num - framewise_output.shape[1], 1)
    output = torch.cat((framewise_output, pad), dim=1)
    return output



class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.att = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
        self.cla = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        attentions = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        cla = torch.sigmoid(self.cla(x))
        x = torch.sum(attentions*cla, dim=2)
        return x, attentions, cla
    
    
class SedCNN(nn.Module):
    def __init__(self, n_classes, n_fft=1024, hop_length=256, n_mels=128, sr=22050, fc_output=1024):
        super().__init__()
        
        self.spectrogram = Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
        
        self.logmel_extractor = LogmelFilterBank(sr=sr, n_fft=n_fft, n_mels=n_mels)
        
        
        
        self.cnn = EfficientNet.from_pretrained('efficientnet-b3', in_channels=1)
        
        self.fc = nn.Linear(1536, fc_output)

        self.att_block = AttentionBlock(fc_output, n_classes)
        self.bn0 = nn.BatchNorm2d(n_mels)
        
        
    def forward(self, x):
        x = self.spectrogram(x)
        
        x = self.logmel_extractor(x)
        
        n_frames = x.size(2)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        x = self.cnn.extract_features(x)

    
        x = torch.mean(x, dim=3)
        

        
        x1 = F.max_pool1d(x, kernel_size=3)
        x2 = F.avg_pool1d(x, kernel_size=3)
        
        x = x1 + x2
        

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1,2)

        x = F.relu_(self.fc(x))

        x = x.transpose(1,2)
        x = F.dropout(x, p=0.5, training=self.training)

        
        clip_output, attentions, segment_output = self.att_block(x)
        
        segment_output = segment_output.transpose(1,2)
        
        frame_output = interpolate(segment_output, 30)
        
        frame_output = pad_framewise_output(frame_output, n_frames)
        
        

        return frame_output, clip_output

def predict_models(models, x):
    predictions = []
    frames = []
    for model in models:
        
        frame, pred = model(x)
        frames += frame
        predictions.append(pred.argmax(dim=1).detach().numpy()[0])
    return sum(frames)/ len(models), np.bincount(np.array(predictions)).argmax()

TIME_COEF = 0.01
def get_prediction(models, x: torch.tensor, label_encoder, class_map):
    x = x.unsqueeze(0)
    
    frames, clip = predict_models(models, x)
    clip = class_map[label_encoder.inverse_transform([clip])[0]]
    
    thresholded = frames.squeeze() >= 0.5
    event_list = []
    
    for target_index in range(thresholded.shape[1]):
            detected = np.argwhere(thresholded[:, target_index]).reshape(-1)
            
            if len(detected) == 0:
                continue
            head_index = 0
            tail_index = 0
            
            while True:
                if tail_index+1 == len(detected) or detected[tail_index+1] - detected[tail_index] != 1:
                    onset = TIME_COEF*detected[head_index]
                    offset = TIME_COEF*detected[tail_index]
                    event = {
                        "onset":onset.numpy(),
                        "offset":offset.numpy(),
                        "target": target_index,
                        "event": class_map[label_encoder.inverse_transform([target_index])[0]]
                    }
                    event_list.append(event)
                    head_index = tail_index + 1
                    tail_index = tail_index + 1
                    if head_index >= len(detected):
                        break
                    
                else:
                    tail_index += 1
                    
    preds = pd.DataFrame(event_list)
    return preds, clip
    