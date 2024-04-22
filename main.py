import os
from TDA_TIME import *
from TDA_PITCH import *

import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
INPUT_FILENAME = 'assets/trial3.wav'

# Get the absolute path to the audio file
INPUT_PATH = os.path.join(os.path.dirname(__file__), INPUT_FILENAME)


if __name__ == '__main__':
    # signal = TDA(INPUT_PATH, duration=2.0, debug=True, shortFileName=INPUT_FILENAME)
    #pointCloud = signal.pointCloud(windowSize=34,umap_dim=3)
    #signal.viewPointCloud3d(pointCloud)

    #diag = signal.persistentHomology(showSimplex=True, showBettiNum=True, filter=None)
    #signal.plotHomology(diag, biggerDiagram=False, diagramType=0)

    dataset_folder_1 = r"C:\Users\David\Documents\GitHub\DATASETS FOR PROJECTS\MuseSyn"
    features_folder_1 = r"C:\Users\David\Documents\GitHub\DATASETS FOR PROJECTS\MuseSyn\features"

    specgram_setting = SpectrogramSetting()
    data_module_1 = PianoRollEstimatorDataModule(spectrogram_setting=specgram_setting,
                                                 dataset_folder=dataset_folder_1,
                                                 feature_folder=features_folder_1)

    #data_module_1.prepare_data()
    trainDataset = data_module_1.get_test_dataset()
    specgram, pianoroll_padded, pianoroll_mask = trainDataset[0]

    # Plot spectrogram
    fig = plt.figure(figsize=(10, 4))
    librosa.display.specshow(np.squeeze(specgram),
                             sr=Constants.sample_rate,
                             x_axis='time', y_axis='linear',
                             cmap=get_cmap('viridis'),
                             hop_length=SpectrogramSetting.hop_length,
                             fmin=SpectrogramSetting.f_min,
                             win_length=SpectrogramSetting.win_length)
    plt.colorbar(format="%+2.f dB")
    plt.title("Spectrogram")
    plt.tight_layout()
    plt.show()

    # Plot piano roll
    fig = plt.figure(figsize=(10, 4))
    librosa.display.specshow(pianoroll_padded,
                             sr=Constants.sample_rate,
                             x_axis='time', y_axis='linear',
                             hop_length=SpectrogramSetting.hop_length,
                             cmap=get_cmap('binary'))
    plt.title("Piano Roll")
    plt.tight_layout()
    plt.show()
print("Hello")