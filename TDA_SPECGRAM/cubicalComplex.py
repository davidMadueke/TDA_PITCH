from gudhi.cubical_complex import CubicalComplex
import copy
from gudhi.persistence_graphical_tools import plot_persistence_diagram, plot_persistence_barcode
from gudhi.representations.vector_methods import Landscape
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

    def persistenceInDim(self, dimensionNumber: int):
        return self.cc.persistence_intervals_in_dimension(dimensionNumber)


if __name__ == '__main__':
    INPUT_PATH = 'assets/A0_55Hz.wav'

    # Get the absolute path to the audio file
    INPUT_PATH = os.path.join(os.path.dirname(__file__), INPUT_PATH)

    waveform, sampleRate = librosa.load(INPUT_PATH,
                                    sr=16000,
                                    mono=True,
                                    offset=0,
                                    duration=None)
    specgramObject = WaveformToLogSpecgram(
        sample_rate=sampleRate,
        n_fft=512*2,
        fmin=27.5,
        bins_per_octave=12*4,
        freq_bins=88*4,
        frane_len=1024
    )
    specgram = specgramObject.process(waveforms=np.array(waveform))

    melSpecgram = librosa.feature.melspectrogram(y=waveform,
                                                 sr=sampleRate,
                                                 window='hamming',
                                                 fmin=27.5,
                                                 n_fft=512*2,
                                                 center=True,
                                                 win_length=1024,
                                                 pad_mode='reflect')
    melSpecgramScaled = librosa.amplitude_to_db(np.abs(melSpecgram), ref=np.max)
    specgramArray = specgram.numpy().squeeze()
    # NOTE - specgramArray: (num_frames, freq_bins)

    fig, ax = plt.subplots()
    img = librosa.display.specshow(melSpecgramScaled, ax=ax, sr=sampleRate, y_axis='log', hop_length=320)
    fig.colorbar(img, ax=ax)
    plt.show()

    complex = CubicalComplex(melSpecgram)
    complex.getInfo()
    a = complex.computePersistence()
    persistenceInDim1 = complex.cc.persistence_intervals_in_dimension(1)
    persistenceInDim0 = complex.cc.persistence_intervals_in_dimension(0)
    print("Persistence of the complex: ", a)

    plot_persistence_diagram(persistence=persistenceInDim0)

    # Create a meshgrid for plotting
    num_frames = specgramObject.num_frames
    frame_len = specgramObject.frame_len
    time = np.arange(num_frames) * (frame_len / sampleRate)  # Time in seconds
    X, Y = np.meshgrid(time, specgramObject.log_idxs)

    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(X, Y, specgramArray, shading='auto')
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Custom Log Spectrogram')
    plt.ylim(specgramObject.log_idxs[0].numpy(), specgramObject.log_idxs[-1].numpy())  # Adjust ylim to match the custom frequency scale
    plt.show()

    persistentLandscape = Landscape(
        num_landscapes=5,
        resolution=10)

    num_diagrams = 1

    persistenceInDim0_finite = copy.deepcopy(persistenceInDim0)
    persistenceInDim0_finite[np.isinf(persistenceInDim0[:, 1]), 1] = 10
    diagramList = [persistenceInDim0_finite]

    landscapes = persistentLandscape.fit_transform(diagramList)
    # Plot the landscapes
    plt.figure(figsize=(10, 6))
    for i, l in enumerate(landscapes, start=1):
        plt.subplot(1, num_diagrams, i)
        plt.plot(l[:10], label="Landscape 1")
        plt.plot(l[10:20], label="Landscape 2")
        plt.plot(l[20:30], label="Landscape 3")
        plt.plot(l[30:40], label="Landscape 4")
        plt.plot(l[40:], label="Landscape 5")
        plt.title(f"Diagram {i} Landscapes")
        plt.xlabel("Sample Points")
        plt.ylabel("Amplitude")
        plt.legend()

    plt.tight_layout()
    plt.show()


    print('Hello There')