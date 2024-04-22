import os
import pickle
import torch
import pytorch_lightning as pl
from typing import Any, Optional
import librosa
import pandas as pd
import pretty_midi as pm

from TDA_PITCH.settings import Constants, TrainingParam
from TDA_SPECGRAM.waveformToLogSpecgram import WaveformToLogSpecgram
import TDA_PITCH.utilities.utils as ut

pm.pretty_midi.MAX_TICK = 1e10


class BaseDataModule(pl.LightningDataModule):
    """Base Datamodule - DO NOT CREATE DATA MODULE HERE!

            Shared functions.
            """

    def __init__(self):
        super().__init__()

    @staticmethod
    def prepare_spectrograms(metadata: pd.DataFrame,
                             spectrogram_setting: Any):
        """Calculate spectrograms and save the pre-calculated features.

            Args:
                metadata: metadata to the dataset.
                spectrogram_setting: the spectrogram setting (type and parameters).
            Returns:
                No return, save pre-calculated features instead.
            """

        for i, row in metadata.iterrows():
            print(f'Preparing spectrogram {i + 1}/{len(metadata)}', end='\r')

            # get audio file and spectrogram file
            audio_file = row['audio_file']
            audio_data, sample_rate = librosa.load(audio_file, sr=Constants.sample_rate)
            spectrogram_file = os.path.join(row['spectrograms_folder'],
                                            f'{spectrogram_setting.to_string()}.pkl')

            # if already calculated, skip
            if os.path.exists(spectrogram_file):
                continue

            # calculate spectrogram
            specgram_object = WaveformToLogSpecgram(sample_rate=spectrogram_setting.sample_rate,
                                                    n_fft=spectrogram_setting.n_fft,
                                                    fmin=spectrogram_setting.f_min,
                                                    bins_per_octave=spectrogram_setting.bins_per_octave,
                                                    freq_bins=spectrogram_setting.freq_bins,
                                                    frame_len=spectrogram_setting.frame_len,
                                                    hop_length=320)
            try:
                if spectrogram_setting.type == 'Reassigned_log2':
                    spectrogram = specgram_object.reassigned_process(audio_data)
                elif spectrogram_setting.type == 'STFT_log2':
                    spectrogram = specgram_object.stft_process(audio_data)
                else:
                    raise ValueError(f"{spectrogram_setting.type} is not a valid spectrogram",
                                     f" Valid spectrograms are {spectrogram_setting.types.values()} ")  # Raise an error
            except ValueError as e:
                print(e)
            # save feature
            ut.mkdir(row['spectrograms_folder'])
            pickle.dump(spectrogram, open(spectrogram_file, 'wb'), protocol=2)

        print()

    def train_dataloader(self):
        # Override train_dataloader
        print('Get train dataloader')
        dataset = self.get_train_dataset()
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        data_loader = torch.utils.data.dataloader.DataLoader(dataset,
                                                             batch_size=TrainingParam.batch_size,
                                                             sampler=sampler,
                                                             drop_last=True)
        return data_loader

    def val_dataloader(self):
        # Override val_dataloader
        print('Get validation dataloader')
        dataset = self.get_valid_dataset()
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        data_loader = torch.utils.data.dataloader.DataLoader(dataset,
                                                             batch_size=TrainingParam.batch_size,
                                                             sampler=sampler,
                                                             drop_last=True)
        return data_loader

    def test_dataloader(self):
        # Override test_dataloader
        print('Get test dataloader')
        dataset = self.get_test_dataset()
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        data_loader = torch.utils.data.dataloader.DataLoader(dataset,
                                                             batch_size=TrainingParam.batch_size,
                                                             sampler=sampler,
                                                             drop_last=True)
        return data_loader
