
import torch
import pywt
import skimage
import scipy.signal
import numpy as np
from einops import repeat, rearrange

def remove_baseline_mean(signal_data):
    # Take first three senconds of data
    signal_baseline = np.array(signal_data[:,:,:128*3]).reshape(40,32,128,-1)
    # Mean of three senconds of baseline will be deducted from all windows
    signal_noise = np.mean(signal_baseline,axis=-1)
    # Expand mask
    signal_noise = repeat(signal_noise,'a b c -> a b (d c)',d=60)
    return signal_data[:,:,128*3:] - signal_noise


def process_video(video, feature='psd'):
    # Transform to frequency domain
    fft_vals = np.fft.rfft(video, axis=-1)
     # Get frequencies for amplitudes in Hz
    samplingFrequency = 128
    fft_freq = np.fft.rfftfreq(video.shape[-1], 1.0/samplingFrequency)
    # Delta, Theta, Alpha, Beta, Gamma
    bands = [(0,4),(4,8),(8,12),(12,30),(30,45)]
    
    band_mask = np.array([np.logical_or(fft_freq < f, fft_freq > t) for f,t in bands])
    band_mask = repeat(band_mask,'a b -> a c b', c=32)
    band_data = np.array(fft_vals)
    band_data = repeat(band_data,'a b -> c a b', c=5)
     
    band_data[band_mask] = 0
    
    band_data = np.fft.irfft(band_data)

    windows = skimage.util.view_as_windows(band_data, (5,32,128), step=128).squeeze()
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

def process_video_wavelet(video, feature='energy', time_domain=False):
    band_widths = [32,16,8,4]
    features = []
    for i in range(5):
        if i == 0:
            # Highest frequencies (64-128Hz) are not used
            cA, cD = pywt.dwt(video.numpy(), 'db4')
        else:
            cA, cD = pywt.dwt(cA, 'db4')
            
            cA_windows = skimage.util.view_as_windows(cA, (32,band_widths[i-1]*2), step=band_widths[i-1]).squeeze()
            cA_windows = np.transpose(cA_windows[:59,:,:],(1,0,2))
            if feature == 'energy':
                cA_windows = np.square(cA_windows)
                cA_windows = np.sum(cA_windows, axis=-1)
                features.append(cA_windows)
            elif feature == 'entropy':
                cA_windows = np.square(cA_windows) * np.log(np.square(cA_windows))
                cA_windows = -np.sum(cA_windows, axis=-1)
                features.append(cA_windows)

            else:
                raise 'Error, invalid wavelet feature'
                
    if time_domain:
        features = np.transpose(features,(2,1,0))
    features = rearrange(features, 'a b c -> (a b) c')
    features = torch.FloatTensor(features)
    
    # Normalization
    m = features.mean(0, keepdim=True)
    s = features.std(0, unbiased=False, keepdim=True)
    features -= m
    features /= s
    return features