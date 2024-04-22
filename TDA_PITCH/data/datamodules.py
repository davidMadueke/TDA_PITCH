import os

import librosa
import pandas as pd
import pretty_midi as pm
from typing import Any, Optional
import pickle
import torch
import pytorch_lightning as pl

from TDA_PITCH.data.basedatamodule import BaseDataModule
from TDA_PITCH.data.datasets import MuseSyn
from TDA_PITCH.settings import Constants, SpectrogramSetting
import TDA_PITCH.utilities.utils as ut


pm.pretty_midi.MAX_TICK = 1e10


class PianoRollEstimatorDataModule(BaseDataModule):
    """Pianoroll Transcription DataModule.

            Args:
                dataset_folder: folder to the MuseSyn dataset.
                feature_folder: folder to save pre-calculated features.
            """

    def __init__(self, spectrogram_setting: Any,
                 dataset_folder: str,
                 feature_folder: str):
        super().__init__()

        self.spectrogram_setting = spectrogram_setting
        self.metadata_train = self.get_metadata(dataset_folder, feature_folder, 'train', three_train_pianos=True)
        self.metadata_valid = self.get_metadata(dataset_folder, feature_folder, 'valid', three_train_pianos=True)
        self.metadata_test = self.get_metadata(dataset_folder, feature_folder, 'test', three_train_pianos=True)

    @staticmethod
    def get_metadata(dataset_folder: str,
                     feature_folder: str,
                     split: str,
                     three_train_pianos: Optional[bool] = True):
        """Get metadata for the dataset.

            Args:
                dataset_folder: folder to the MuseSyn dataset.
                feature_folder: folder to save pre-calculated features.
                split: train/test/valid split.
                three_train_pianos: whether to use only three pianos for model training
            Returns:
                metadata: (pd.DataFrame) dataset metadata.
            """

        if three_train_pianos:
            pianos = Constants.pianos if split == 'test' else Constants.pianos[:-1]
        else:
            pianos = Constants.pianos

        print(f'Get {split} metadata, {len(pianos)} pianos')

        metadata_file = f'metadata/cache/{split}-pianos={len(pianos)}.csv'
        if os.path.exists(metadata_file):
            return pd.read_csv(metadata_file)

        metadata = []
        print(os.path.isfile(rf'metadata/MuseSyn_{split}.txt'), os.path.abspath(rf'metadata/MuseSyn_{split}.txt'))
        for i, row in pd.read_csv(rf'metadata/MuseSyn_{split}.txt', header=0).iterrows():
            name = row['name']
            for piano in pianos:
                # get information for each piece
                midi_file = os.path.join(dataset_folder, 'midi', name + '.mid')
                audio_file = os.path.join(dataset_folder, 'flac', piano, name + '.flac')
                spectrograms_folder = os.path.join(feature_folder, 'spectrograms', piano, name)
                pianoroll_file = os.path.join(feature_folder, 'pianoroll', name + '.pkl')
                duration = pm.PrettyMIDI(midi_file).get_end_time()

                # udpate metadata
                metadata.append({'name': name,
                                 'piano': piano,
                                 'midi_file': midi_file,
                                 'audio_file': audio_file,
                                 'split': split,
                                 'spectrograms_folder': spectrograms_folder,
                                 'pianoroll_file': pianoroll_file,
                                 'duration': duration})
                print(f"Adding: , {name}, {piano}, {split}, {duration} ")

        # to DataFrame and save metadata
        metadata = pd.DataFrame(metadata)
        ut.mkdir(os.path.split(metadata_file)[0])
        metadata.to_csv(metadata_file)
        return metadata


    @staticmethod
    def prepare_pianorolls(metadata: pd.DataFrame):
        """Calculate pianorolls and save pre-calculated feature.

            Args:
                metadata: metadata to the dataset.
            Returns:
                No return, save pre-calculated features instead.
            """

        for i, row in metadata.iterrows():
            print(f'Preparing pianoroll {i+1}/{len(metadata)}', end='\r')

            # get midi file and pianoroll file
            midi_file = row['midi_file']
            pianoroll_file = row['pianoroll_file']

            # if already calculated, skip
            if os.path.exists(pianoroll_file):
                continue

            # calculate pianoroll and save feature
            midi_data = pm.PrettyMIDI(midi_file)
            pianoroll = midi_data.get_piano_roll(fs=1./Constants.hop_time)[21:21+88]  # 88 piano keys
            ut.mkdir(os.path.split(pianoroll_file)[0])
            pickle.dump(pianoroll, open(pianoroll_file, 'wb'), protocol=2)

        print()

    def prepare_data(self) -> None:
        # Override prepare_data
        for metadata in [self.metadata_train, self.metadata_valid, self.metadata_test]:
            self.prepare_pianorolls(metadata)
            self.prepare_spectrograms(metadata, self.spectrogram_setting)

    def get_train_dataset(self):
        return MuseSyn(self.metadata_train, self.spectrogram_setting)

    def get_valid_dataset(self):
        return MuseSyn(self.metadata_valid, self.spectrogram_setting)

    def get_test_dataset(self):
        return MuseSyn(self.metadata_test, self.spectrogram_setting)

