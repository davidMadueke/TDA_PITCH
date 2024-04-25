import os
import numpy as np
import librosa
import pandas as pd
import pretty_midi as pm
from typing import Any, Optional
import pickle
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split

from TDA_PITCH.data.basedatamodule import BaseDataModule
from TDA_PITCH.data.datasets import MuseSyn, MIR1K
from TDA_PITCH.settings import Constants, SpectrogramSetting, TrainingParams
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
            pianoroll = midi_data.get_piano_roll(fs=1./Constants.pianoroll_hop_time)[21:21 + 88]  # 88 piano keys
            ut.mkdir(os.path.split(pianoroll_file)[0])
            pickle.dump(pianoroll, open(pianoroll_file, 'wb'), protocol=2)

        print()

    def prepare_audio(self, audio_file):
        audio_data, sample_rate = librosa.load(audio_file, sr=Constants.sample_rate)
        return audio_data, sample_rate

    def prepare_data(self) -> None:
        # Override prepare_data
        if not TrainingParams.PREPARE_FLAG:
            for metadata in [self.metadata_train, self.metadata_valid, self.metadata_test]:
                self.prepare_pianorolls(metadata)
                self.prepare_spectrograms(metadata, self.spectrogram_setting)

    def get_train_dataset(self):
        return MuseSyn(self.metadata_train, self.spectrogram_setting)

    def get_valid_dataset(self):
        return MuseSyn(self.metadata_valid, self.spectrogram_setting)

    def get_test_dataset(self):
        return MuseSyn(self.metadata_test, self.spectrogram_setting)


class F0EstimatorDataModule(BaseDataModule):
    """F0 Estimator DataModule.

            Args:
                dataset_folder: folder to the MIR-1K dataset.
                feature_folder: folder to save pre-calculated features.

            NOTE - In Librosa, left channels in stereo audio is always the second array - i.e mywaveform[1]
            We are choosing the left channel as that has the isolate vocal recording
            """

    def __init__(self, spectrogram_setting: Any,
                 dataset_folder: str,
                 feature_folder: str):
        super().__init__()

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.spectrogram_setting = spectrogram_setting
        self.metadata = self.get_metadata(dataset_folder, feature_folder)
        self.dataset = MIR1K(self.metadata, self.spectrogram_setting)

    @staticmethod
    def get_metadata(dataset_folder: str,
                     feature_folder: str):
        """Get metadata for the dataset.

            Args:
                dataset_folder: folder to the MIR-1K dataset.
                feature_folder: folder to save pre-calculated features.
            Returns:
                metadata: (pd.DataFrame) dataset metadata.
            """

        print(f'Get metadata')

        metadata_file = f'metadata/cache/MIR-1K.csv'

        if os.path.exists(metadata_file):
            return pd.read_csv(metadata_file)

        metadata = []
        audio_dir = os.path.join(dataset_folder, 'Wavfile')
        audio_list = os.listdir(audio_dir)
        for i, row in enumerate(audio_list):
            name = row.split('.')[0]  # Remove the file extension from the name
            # get information for each piece
            pitch_contour_file = os.path.join(dataset_folder, 'PitchLabel', name + '.pv')
            audio_file = os.path.join(dataset_folder, 'Wavfile', name + '.wav')
            spectrograms_folder = os.path.join(feature_folder, 'spectrograms', name)
            one_hot_pitch_vector_file = os.path.join(feature_folder, 'pitchvector', name + '.pkl')
            duration = librosa.get_duration(path=audio_file)

            # udpate metadata
            metadata.append({'name': name,
                             'audio_file': audio_file,
                             'pitch_contour_file': pitch_contour_file,
                             'pitch_vector_file': one_hot_pitch_vector_file,
                             'spectrograms_folder': spectrograms_folder,
                             'duration': duration})
            print(f"Adding: {name}, {duration} ")

        # to DataFrame and save metadata
        metadata = pd.DataFrame(metadata)
        ut.mkdir(os.path.split(metadata_file)[0])
        metadata.to_csv(metadata_file)
        return metadata


    @staticmethod
    def prepare_pitchvectors(metadata: pd.DataFrame):
        """Calculate pitch vector and save pre-calculated feature.

            Args:
                metadata: metadata to the dataset.
            Returns:
                No return, save pre-calculated features instead.
            """

        for i, row in metadata.iterrows():
            print(f'Preparing pitchvector {i+1}/{len(metadata)}', end='\r')

            # get midi file and pianoroll file
            pitch_contour_file = row['pitch_contour_file']
            pitch_vector_file = row['pitch_vector_file']

            # if already calculated, skip
            if os.path.exists(pitch_vector_file):
                continue

            # calculate pitch vector and save feature
            pitch_contour = np.loadtxt(pitch_contour_file)
            # convert midi to frequency
            pitch_vector = 2**((pitch_contour - 69)/12) * 440
            ut.mkdir(os.path.split(pitch_vector_file)[0])
            pickle.dump(pitch_vector, open(pitch_vector_file, 'wb'), protocol=2)

        print()

    def prepare_audio(self, audio_file):
        audio_data, sample_rate = librosa.load(audio_file, sr=Constants.sample_rate, mono=False)
        return audio_data[1], sample_rate

    def prepare_data(self) -> None:
        # Override prepare_data
        self.prepare_pitchvectors(self.metadata)
        self.prepare_spectrograms(self.metadata, self.spectrogram_setting)

        generator = torch.Generator().manual_seed(Constants.random_seed)
        self.train_dataset, self.valid_dataset = random_split(self.dataset,
                                                            lengths=[0.6, 0.4],
                                                            generator=generator)
        self.valid_dataset, self.test_dataset = random_split(self.valid_dataset,
                                                           lengths=[0.5, 0.5],
                                                           generator=generator)

    def get_train_dataset(self):
        return self.train_dataset

    def get_valid_dataset(self):
        return self.valid_dataset

    def get_test_dataset(self):
        return self.test_dataset