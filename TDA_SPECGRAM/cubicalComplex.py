import gudhi
import copy
from gudhi.persistence_graphical_tools import plot_persistence_diagram, plot_persistence_barcode
from gudhi.representations.vector_methods import Landscape
from TDA_SPECGRAM.waveformToLogSpecgram import WaveformToLogSpecgram
import torch
import librosa
import os
import numpy as np
from matplotlib import pyplot as plt

gudhi.persistence_graphical_tools._gudhi_matplotlib_use_tex = False

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
    INPUT_PATH = 'assets/trial4.flac'

    # Get the absolute path to the audio file
    INPUT_PATH = os.path.join(os.path.dirname(__file__), INPUT_PATH)

    waveform, sampleRate = librosa.load(INPUT_PATH,
                                    sr=16000,
                                    mono=True,
                                    offset=0,
                                    duration=2)
    specgramObject = WaveformToLogSpecgram(
        sample_rate=sampleRate,
        n_fft=512*2,
        fmin=27.5,
        bins_per_octave=12*4,
        freq_bins=88*4,
        frame_len=1024
    )
    specgram = specgramObject.reassigned_process(waveforms=np.array(waveform))

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

    # NOTE - Reassigned Spectrogram has inherent randomness closer to the low end that could ocurr
    freqs, times, mags = librosa.reassigned_spectrogram(y=waveform, sr=sampleRate,
                                                        n_fft=512*2, reassign_times=True,
                                                        pad_mode='reflect')
    mags_db = librosa.amplitude_to_db(mags, ref=np.max)
    fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True)
    img = librosa.display.specshow(mags_db, x_axis="s", y_axis="log", sr=sampleRate,
                                   hop_length=512*2 // 4, ax=ax)
    ax.set(title="Spectrogram", xlabel=None)
    ax.label_outer()
    #ax[1].scatter(times, freqs, c=mags_db, cmap="magma", alpha=0.1, s=5)
    #ax[1].set_title("Reassigned spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    complex = CubicalComplex(mags_db)
    complex.getInfo()
    a = complex.computePersistence()
    persistenceInDim1 = complex.cc.persistence_intervals_in_dimension(1)
    persistenceInDim0 = complex.cc.persistence_intervals_in_dimension(0)
    print("Persistence of the complex: ", a)

    plot_persistence_diagram(persistence=a)

    # Create a meshgrid for plotting
    num_frames = specgramObject.num_frames
    frame_len = specgramObject.frame_len
    hop_len = specgramObject.hop_length
    time = np.arange(num_frames) * (hop_len / sampleRate)  # Time in seconds
    reassigned_times = specgramObject.reassignedtimes
    X, Y = np.meshgrid(time, specgramObject.log_freqs)

    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(X, Y, specgramArray, shading='auto')
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel(f'Log-Frequency (Hz)')
    plt.title('Custom Log Spectrogram')
    plt.ylim(specgramObject.log_freqs[0].numpy(), min(specgramObject.log_freqs[-1].numpy(),1000))  # Adjust ylim to match the custom frequency scale
    plt.show()

    persistentLandscape = Landscape(
        num_landscapes=5,
        resolution=10,
        keep_endpoints=True)

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