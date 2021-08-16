from configparser import ConfigParser
import numpy as np
import os
from scipy.io import wavfile
import kaldiio
import torch


class FeatureComputer:
    def __init__(self,cfg):

        self.feature_dim = cfg.audio_feature.fbank_dim
        self.low_frame_rate_stack  = cfg.audio_feature.low_frame_rate_stack
        self.low_frame_rate_stride = cfg.audio_feature.low_frame_rate_stride


        cmvn = np.loadtxt(cfg.audio_feature.cmvn_npy_file)
        
        self.means = torch.FloatTensor(cmvn[0])
        self.vars = torch.FloatTensor(cmvn[1])
        

    def computer_feature(self, wav_path):
        spect = kaldiio.load_mat(wav_path)
        spect = torch.FloatTensor(np.array(spect))
        spect = (spect + self.means) * self.vars
        spect = self.statc_frame(spect.numpy())

        return spect

    def statc_frame(self,inputs):
        m = self.low_frame_rate_stack
        n = self.low_frame_rate_stride

        lfr_input = []
        T = inputs.shape[0]
        T_lfr = int(np.ceil(T / n))
        for t_i in range(T_lfr):
            if m <= T - t_i * n:
                lfr_input.append(np.hstack(inputs[t_i * n:t_i * n + m]))
            else:
                num_padding = m - (T - t_i * n)
                frame = np.hstack(inputs[t_i * n:])
                for _ in range(num_padding):
                    frame = np.hstack((frame, inputs[-1]))
                lfr_input.append(frame)
        return np.vstack(lfr_input)

    def get_feature_dim(self):
        return self.feature_dim * self.low_frame_rate_stack