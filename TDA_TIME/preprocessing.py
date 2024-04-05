import librosa
from scipy.signal import *


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


# Chose to use scipy.signal.decimate to perform downsampling as supposed to direct implementation like Fireaizen et al
def signal_downSample(signal, downsampleFactor):
    return decimate(signal,downsampleFactor, ftype='fir')


def import_audio_signal(filename, startTime: float = 0, duration: float = 30):
    data, sampleRate = librosa.load(filename,
                                    sr=None,  # We will implement downsampling using scipy instead
                                    mono=True,
                                    offset=startTime,
                                    duration=duration)
    return data, sampleRate
