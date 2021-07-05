
import torch
import pywt
import skimage
import scipy.signal
import numpy as np
from einops import repeat, rearrange
import matplotlib.pyplot as plt

def remove_baseline_mean(signal_data):
    # Take first three senconds of data
    signal_baseline = np.array(signal_data[:,:,:128*3]).reshape(40,32,128,-1)
    # Mean of three senconds of baseline will be deducted from all windows
    signal_noise = np.mean(signal_baseline,axis=-1)
    # Expand mask
    signal_noise = repeat(signal_noise,'a b c -> a b (d c)',d=60)
    return signal_data[:,:,128*3:] - signal_noise


def process_video(video, feature='psd', window_step = 128):
    # Transform to frequency domain
    fft_vals = np.fft.rfft(video, axis=-1)
     # Get frequencies for amplitudes in Hz
    samplingFrequency = 128
    fft_freq = np.fft.rfftfreq(video.shape[-1], 1.0/samplingFrequency)
    # Delta, Theta, Alpha, Beta, Gamma
    bands = [(4,8),(8,12),(12,30),(30,45)]
    
    band_mask = np.array([np.logical_or(fft_freq < f, fft_freq > t) for f,t in bands])
    band_mask = repeat(band_mask,'a b -> a c b', c=32)
    band_data = np.array(fft_vals)
    band_data = repeat(band_data,'a b -> c a b', c=len(bands))
     
    band_data[band_mask] = 0
    
    band_data = np.fft.irfft(band_data)

    windows = skimage.util.view_as_windows(band_data, (len(bands),32,128), step=window_step).squeeze()
    # (5, 32, 60, 128)
    windows = rearrange(windows, 'a b c d -> b c a d')
    
    if feature == 'psd':
        features = scipy.signal.periodogram(windows)[1]
        features = np.mean(features, axis=-1)
    # elif feature == 'de':
    #     features = np.apply_along_axis(calculate_de, -1, windows)

    
    features = rearrange(features, 'a b c -> (a b) c')
    features = torch.FloatTensor(features)

    return features

def get_wavelet_energy(cD):
    cD = np.square(cD)
    return np.sum(cD, axis=-1)

def process_video_wavelet(video):
    ## Edge cases for sample size < 2s
    window_size = 256 if video.shape[-1] > 256 else 128
    step_size = 128 if video.shape[-1] > 256 else 64
    video_windows = skimage.util.view_as_windows(video.numpy(), (32,window_size), step=step_size).squeeze()
    video_windows = video_windows.transpose(1,0,2)
    mother_wavelet = 'db4'
    N = 5
    cA, cD = pywt.dwt(video_windows, mother_wavelet)
    # First Detail coefficient is disregarded. Noise frequencies -> (64-128Hz)
    features = []
    for i in range(N-1):
        cA, cD = pywt.dwt(cA, mother_wavelet)
        features.append(get_wavelet_energy(cD))

    features = torch.FloatTensor(features)

    # time_domain = True
    # if time_domain:
    #     features = np.transpose(features,(2,1,0))

    features = rearrange(features, 'a b c -> (a b) c') # (4,32,59)

    # Normalization
    m = features.mean(0, keepdim=True)
    s = features.std(0, unbiased=False, keepdim=True)
    features -= m
    features /= s

    return features
    
    