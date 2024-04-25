import os
import pickle

import librosa
import torch
import torch.nn.functional as F
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

    NOTE - In Librosa, left channels in stereo audio is always the second array - i.e mywaveform[1]
    We are choosing the left channel as that has the isolated vocal recording

    Args:
        Dataset (_type_): return specgram, pitch
    """
    def __init__(self, metadata: pd.DataFrame, spectrogram_setting: SpectrogramSetting):
        super().__init__()
        self.metadata = metadata
        self.spectrogram_setting = spectrogram_setting
        self.spectrogram_parameters = None

        # Calculate dataset length
        self.durations = self.metadata['duration'].to_numpy()
        self.n_segments_all = np.ceil(self.durations / Constants.segment_length).astype(int)
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
            spectrogram_parameters_file = os.path.join(row['spectrograms_folder'],
                                                       f'{self.spectrogram_setting.to_string()}_parameters.pkl')
            spectrogram_full = pickle.load(open(spectrogram_file, 'rb'))
            if len(spectrogram_full.shape) == 2:
                spectrogram_full = np.expand_dims(spectrogram_full, axis=0)
            self.spectrogram_parameters = pickle.load(open(spectrogram_parameters_file, 'rb'))

            pitch_vector_file = row['pitch_vector_file']
            pitch_vector_full = pickle.load(open(pitch_vector_file, 'rb'))


            # get segment
            start = index * Constants.pitch_vector_input_length
            end = start + Constants.pitch_vector_input_length
            spectrogram = spectrogram_full[:, :, start:min(spectrogram_full.shape[2], end)]
            pitch_vector = pitch_vector_full[start:min(pitch_vector_full.shape[0], end)]
            # => [freq_bins x T]
            pitch_vector_onehot = self.hz_to_onehot(pitch_vector,
                                                    fmin=self.spectrogram_setting.f_min,
                                                    freq_bins=self.spectrogram_setting.freq_bins,
                                                    bins_per_octave=self.spectrogram_setting.bins_per_octave)

            # initialise padded spectrogram and pitch_vector
            spectrogram_padded = np.zeros(
                (spectrogram_full.shape[0],
                 spectrogram_full.shape[1], Constants.pitch_vector_input_length), dtype=float)
            pitch_vector_padded = np.zeros((self.spectrogram_setting.freq_bins,
                                            Constants.pitch_vector_input_length),
                                           dtype=float)
            pitch_vector_mask = np.zeros((self.spectrogram_setting.freq_bins,
                                        Constants.pitch_vector_input_length),
                                           dtype=float)
            # overwrite with values
            # => [Batch x Freq_Bins x Num_Frames]
            spectrogram_padded[:, :, :spectrogram.shape[2]] = spectrogram
            self.spectrogram_parameters["num_frames"] = spectrogram_padded.shape[2]
            pitch_vector_padded[:, :pitch_vector_onehot.shape[1]] = pitch_vector_onehot
            pitch_vector_mask[:, :pitch_vector_onehot.shape[1]] = 1.

            return spectrogram_padded, pitch_vector_padded, pitch_vector_mask

    @staticmethod
    def hz_to_onehot(hz, fmin, freq_bins, bins_per_octave):
        # input: [T]
        # output: [freq_bins x T]
        hz = torch.tensor(hz)
        indexes = ( torch.log((hz+0.0000001)/fmin) / np.log(2.0**(1.0/bins_per_octave)) + 0.5 ).long()
        assert(torch.max(indexes) < freq_bins)
        mask = (indexes >= 0).long()
        # => [T x 1]
        mask = torch.unsqueeze(mask, dim=1)
        # => [T x freq_bins]
        onehot = F.one_hot(torch.clip(indexes, 0), freq_bins)
        onehot = onehot * mask # mask the freq below fmin
        # => [freq_bins x T]
        onehot_transposed = torch.transpose(onehot, dim0=0, dim1=1).numpy()
        return onehot_transposed


# Create the PianoRollEstimatorDataset
class MuseSyn(Dataset):
    def __init__(self, metadata: pd.DataFrame, spectrogram_setting: SpectrogramSetting):
        super().__init__()
        self.metadata = metadata
        self.spectrogram_setting = spectrogram_setting
        self.spectrogram_parameters = None

        # Calculate dataset length
        self.durations = self.metadata['duration'].to_numpy()
        self.n_segments_all = np.ceil(self.durations / Constants.segment_length).astype(int)
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
            spectrogram_parameters_file = os.path.join(row['spectrograms_folder'],
                                                       f'{self.spectrogram_setting.to_string()}_parameters.pkl')
            pianoroll_file = row['pianoroll_file']
            spectrogram_full = pickle.load(open(spectrogram_file, 'rb'))
            if len(spectrogram_full.shape) == 2:
                spectrogram_full = np.expand_dims(spectrogram_full, axis=0)
            self.spectrogram_parameters = pickle.load(open(spectrogram_parameters_file, 'rb'))
            pianoroll_full = pickle.load(open(pianoroll_file, 'rb'))

            # get segment
            start = index * Constants.pianoroll_input_length
            end = start + Constants.pianoroll_input_length
            spectrogram = spectrogram_full[:, :, start:min(spectrogram_full.shape[2], end)]
            pianoroll = pianoroll_full[:, start:min(pianoroll_full.shape[1], end)]

            # initialise padded spectrogram and pianoroll
            spectrogram_padded = np.zeros(
                (spectrogram_full.shape[0],
                 spectrogram_full.shape[1], Constants.pianoroll_input_length), dtype=float)
            pianoroll_padded = np.zeros((88, Constants.pianoroll_input_length), dtype=float)
            pianoroll_mask = np.zeros((88, Constants.pianoroll_input_length), dtype=float)
            # overwrite with values
            # => [Batch x Freq_Bins x Num_Frames]
            spectrogram_padded[:, :, :spectrogram.shape[2]] = spectrogram
            self.spectrogram_parameters["num_frames"] = spectrogram_padded.shape[2]
            pianoroll_padded[:, :pianoroll.shape[1]] = pianoroll
            pianoroll_mask[:, :pianoroll.shape[1]] = 1.
            return (torch.tensor(spectrogram_padded),
                    torch.tensor(pianoroll_padded), torch.tensor(pianoroll_mask))

