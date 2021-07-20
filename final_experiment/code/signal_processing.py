
import torch
import pywt
import skimage
import scipy.signal as signal
import numpy as np
from einops import repeat, rearrange
import matplotlib.pyplot as plt

def process_window_psd(window):
    # Extract power spectral density using welch's method
    freqs, psd = signal.welch(window,fs=128,nperseg=256 if window.shape[1] > 256 else window.shape[1])
    
    bands = [(4,8),(8,16),(16,32),(32,64)]
    features = []
    for band_from,band_to in bands:
        freq_mask = np.logical_and(freqs>=band_from,freqs<band_to)
        psd_band = psd[:,freq_mask]
        features.append(np.mean(psd_band,axis=1))
        
    features = torch.FloatTensor(features).transpose(0,1)
    # Normalization
    m = features.mean(0, keepdim=True)
    s = features.std(0, unbiased=False, keepdim=True)
    features -= m
    features /= s
    return features


def get_wavelet_energy(cD):
    cD = np.square(cD)
    return np.sum(cD, axis=-1)


def process_window_wavelet(window):
    mother_wavelet = 'db4'
    N = 5
    cA, cD = pywt.dwt(window, mother_wavelet)
    # First Detail coefficient is disregarded. Noise frequencies -> (64-128Hz)
    features = []
    for i in range(N-1):
        cA, cD = pywt.dwt(cA, mother_wavelet)
        features.append(get_wavelet_energy(cD))

    features = torch.FloatTensor(features).transpose(0,1)
    # Normalization
    m = features.mean(0, keepdim=True)
    s = features.std(0, unbiased=False, keepdim=True)
    features -= m
    features /= s
    return features

def process_window_raw(window):
    features = torch.FloatTensor(window)
    # Normalization
    m = features.mean(0, keepdim=True)
    s = features.std(0, unbiased=False, keepdim=True)
    features -= m
    features /= s
    return features
    
    
# Unused
def remove_baseline_mean(signal_data):
    # Take first three senconds of data
    signal_baseline = np.array(signal_data[:,:,:128*3]).reshape(40,32,128,-1)
    # Mean of three senconds of baseline will be deducted from all windows
    signal_noise = np.mean(signal_baseline,axis=-1)
    # Expand mask
    signal_noise = repeat(signal_noise,'a b c -> a b (d c)',d=60)
    return signal_data[:,:,128*3:] - signal_noise