import os
import pickle

import librosa
import torch
import torchaudio
from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np

from TDA_PITCH.settings import Constants, SpectrogramSetting


# Create the f0-estimator dataset
class MIR1K(Dataset):
    """
    offical documentation
    frame size of 40ms -> 640 sample points
    hop size of 20 ms -> 320 sample points
    each 10ms hop size has one pitch vector

    Args:
        Dataset (_type_): return specgram, pitch
    """
    def __init__(self, metadata: pd.DataFrame, spectrogram_setting: SpectrogramSetting):
        super.__init__()
        self.metadata = metadata
        self.spectrogram_setting = spectrogram_setting

        # Calculate dataset length
        durations = self.metadata['duration'].to_numpy()
        self.n_segments_all = np.ceil(durations / Constants.segment_length).astype(int)
        self.length = np.sum(self.n_segments_all)


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        pass


# Create the PianoRollEstimatorDataset
class MuseSyn(Dataset):
    def __init__(self, metadata: pd.DataFrame, spectrogram_setting: SpectrogramSetting):
        super.__init__()
        self.metadata = metadata
        self.spectrogram_setting = spectrogram_setting

        # Calculate dataset length
        durations = self.metadata['duration'].to_numpy()
        self.n_segments_all = np.ceil(durations / Constants.segment_length).astype(int)
        self.length = np.sum(self.n_segments_all)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        for i, row in self.metadata.iterrows():
            # check if we are looking for a segment in current piece, if not, move to the next.
            if index >= self.n_segments_all[i]:
                index -= self.n_segments_all[i]
                continue

            # get item data
            # load spectrogram and pianorolls
            spectrogram_file = os.path.join(row['spectrograms_folder'],
                                            f'{self.spectrogram_setting.to_string()}.pkl')
            pianoroll_file = row['pianoroll_file']
            spectrogram_full = pickle.load(open(spectrogram_file, 'rb'))
            if len(spectrogram_full.shape) == 2:
                spectrogram_full = np.expand_dims(spectrogram_full, axis=0)
            pianoroll_full = pickle.load(open(pianoroll_file, 'rb'))

            # get segment
            start = index * Constants.input_length
            end = start + Constants.input_length
            spectrogram = spectrogram_full[:, :, start:min(spectrogram_full.shape[2], end)]
            pianoroll = pianoroll_full[:, start:min(pianoroll_full.shape[1], end)]

            # initialise padded spectrogram and pianoroll
            spectrogram_padded = np.zeros(
                (spectrogram_full.shape[0], spectrogram_full.shape[1], Constants.input_length), dtype=float)
            pianoroll_padded = np.zeros((88, Constants.input_length), dtype=float)
            pianoroll_mask = np.zeros((88, Constants.input_length), dtype=float)
            # overwrite with values
            spectrogram_padded[:, :, :spectrogram.shape[2]] = spectrogram
            pianoroll_padded[:, :pianoroll.shape[1]] = pianoroll
            pianoroll_mask[:, :pianoroll.shape[1]] = 1.

            return spectrogram_padded, pianoroll_padded, pianoroll_mask

