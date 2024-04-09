from gudhi.cubical_complex import CubicalComplex
import gudhi
from TDA_SPECGRAM.waveformToLogSpecgram import WaveformToLogSpecgram
import torch
import librosa
import os
import numpy as np
from matplotlib import pyplot as plt

class CubicalComplex:

    def __init__(self, specgram):
        self.cc = gudhi.cubical_complex.CubicalComplex(top_dimensional_cells=specgram)

    def getInfo(self):
        return print(f"Cubical complex is of dimension {self.cc.dimension()} - {self.cc.num_simplices()} simplices.")

    def bettiNumbers(self):
        return self.cc.bettiNumbers()

    def computePersistence(self):
        # Returns a List of pairs(dimension, pair(birth, death)) – the persistence of the complex.
        return self.cc.persistence()


if __name__ == '__main__':
    INPUT_PATH = 'assets/trial1.wav'

    # Get the absolute path to the audio file
    INPUT_PATH = os.path.join(os.path.dirname(__file__), INPUT_PATH)

    waveform, sampleRate = librosa.load(INPUT_PATH,
                                    sr=16000,  # We will implement downsampling using scipy instead
                                    mono=True,
                                    offset=30,
                                    duration=2)
    specgram = WaveformToLogSpecgram(
        sample_rate=sampleRate,
        n_fft=512*2,
        fmin=27.5,
        bins_per_octave=12*4,
        freq_bins=88*4,
        frane_len=1024
    ).process(waveforms=np.array(waveform))

    #specgram = librosa.feature.melspectrogram(y=waveform, sr=sampleRate, window='hamming', fmin=27.5, n_fft=512*2).transpose()

    specgramArray = specgram.numpy().squeeze()
    # NOTE - specgramArray: (num_frames, freq_bins)

    fig, ax = plt.subplots()
    img = librosa.display.specshow(specgramArray, ax=ax)
    fig.colorbar(img, ax=ax)
    plt.show()

    complex = CubicalComplex(specgramArray)
    complex.getInfo()
    print("Persistence of the complex: ", complex.computePersistence())

    print('Hello There')