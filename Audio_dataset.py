import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
import os


def scale(X, x_min, x_max):
    """Standardize acoustic data"""
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    return x_min + nom/denom


def get_white_noise(signal, SNR):
    rms_s = np.sqrt(np.mean(signal ** 2))

    rms_n = np.sqrt(rms_s ** 2 / (pow(10, SNR / 10)))

    STD_n = rms_n
    noise = np.random.normal(0, STD_n, signal.shape[0])
    return noise


def add_noise(signal, SNR=8):
  """Add additive and multiplicative noise"""
  additive_noise = get_white_noise(signal, SNR)
  multiplicative_noise = np.random.randint(1, 3, size=signal.shape[0])
  signal_with_noise = multiplicative_noise * signal + additive_noise

  return signal_with_noise


class AudioDataset(Dataset):
    """Create dataset for torch.dataloader -
    return: signal with noise, signal without noise and id of audio"""
    def __init__(self, path, SNR):
        self.path = path
        self.SNR = SNR

    def __len__(self):
        return len(os.listdir(self.path))

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        audio_id = index
        audio_name = os.path.join(self.path, os.listdir(self.path)[index])
        x, sr = librosa.load(audio_name, sr=48000)
        x = scale(x[5000:75000], -1, 1)

        #Add nosie (multiplicative and addidtive (white noise) using SNR
        x_noise = scale(add_noise(x, self.SNR), -1, 1)
        audiofile = torch.from_numpy(x)
        audiofile_with_noise = torch.from_numpy(x_noise)

        return audiofile_with_noise, audiofile, audio_id
