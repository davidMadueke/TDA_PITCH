from scipy.io import wavfile
from scipy.signal import *
import numpy as np

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def signal_downSample(signal, downsampleRate):
    i = 0
    signal_length = len(signal)
    new_sig_list = []
    while i < signal_length:
        new_sig_list.append(signal[i])
        i += downsampleRate
    return np.array(new_sig_list)

def import_audio_signal(filename):
    fs,data = wavfile.read(filename)
    try:
        signal1 = data[:, 0]
        signal2 = data[:, 1]
    except:
        sig = data[:]
        signal1=sig
        signal2= sig
    return fs, signal1, signal2
