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

    dataset_folder_2 = r"C:\Users\David\Documents\GitHub\DATASETS FOR PROJECTS\MIR-1K"
    features_folder_2 = r"C:\Users\David\Documents\GitHub\DATASETS FOR PROJECTS\MIR-1K\features"

    specgram_setting = SpectrogramSetting()
    data_module_1 = PianoRollEstimatorDataModule(spectrogram_setting=specgram_setting,
                                                 dataset_folder=dataset_folder_1,
                                                 feature_folder=features_folder_1)

    #data_module_1.prepare_data()
    #trainDataset = data_module_1.get_test_dataset()
    #specgram, pianoroll_padded, pianoroll_mask = trainDataset[3]

    data_module_2 = F0EstimatorDataModule(spectrogram_setting=specgram_setting,
                                          dataset_folder=dataset_folder_2,
                                          feature_folder=features_folder_2)
    data_module_2.prepare_data()
    trainDataset2 = data_module_2.get_train_dataset()
    specgram2, pitch_vector, pitch_vector_mask = trainDataset2[0]

    # Create a meshgrid for plotting
    #num_frames = trainDataset.spectrogram_parameters["num_frames"]
    #log_freqs = trainDataset.spectrogram_parameters["log_freqs"]
    #frame_len = SpectrogramSetting.frame_len
    #hop_len = SpectrogramSetting.hop_length
    #time = np.arange(num_frames) * (hop_len / SpectrogramSetting.sample_rate)  # Time in seconds
    #X, Y = np.meshgrid(time, log_freqs)

    # Plot the spectrogram
    #plt.figure(figsize=(10, 6))
    #plt.pcolormesh(X, Y, np.squeeze(specgram), shading='auto')
    #plt.colorbar(label='Magnitude (dB)')
    #plt.xlabel('Time (s)')
    #plt.ylabel(f'Log-Frequency (Hz)')
    #plt.title('Log_2 Spectrogram of MuseSyn Truncated Dataset Example')
    #plt.ylim(log_freqs[0].numpy(),
    #         min(log_freqs[-1].numpy(), 100000))  # Adjust ylim to match the custom frequency scale
    #plt.show()

    ######################################

    # Create a meshgrid for plotting
    num_frames2 = data_module_2.dataset.spectrogram_parameters["num_frames"]
    log_freqs2 = trainDataset2.dataset.spectrogram_parameters["log_freqs"]
    frame_len2 = SpectrogramSetting.frame_len
    hop_len = SpectrogramSetting.hop_length
    time = np.arange(num_frames2) * (hop_len / SpectrogramSetting.sample_rate)  # Time in seconds
    X, Y = np.meshgrid(time, log_freqs2)

    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(X, Y, np.squeeze(specgram2), shading='auto')
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel(f'Log-Frequency (Hz)')
    plt.title('Log_2 Spectrogram of MIR-1K Truncated Dataset Example')
    plt.ylim(log_freqs2[0].numpy(),
             min(log_freqs2[-1].numpy(), 100000))  # Adjust ylim to match the custom frequency scale
    plt.show()


    # Plot piano roll
    #fig = plt.figure(figsize=(10, 4))
    #librosa.display.specshow(pianoroll_padded,
    #                         sr=1/Constants.pianoroll_hop_time,
    #                         x_axis='time', y_axis='linear',
    #                         hop_length=SpectrogramSetting.hop_length,
    #                         cmap=get_cmap('binary'))
    #plt.title("Piano Roll")
    #plt.tight_layout()
    #plt.show()
    print("Hello")