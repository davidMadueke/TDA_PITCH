import os
import sys
from typing import Optional, Any
import numpy as np
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pypianoroll

from TDA_PITCH.models.basemodel import BaseModel
from TDA_PITCH.models.containers import init_layer, init_bn, init_gru, ConvBlock_PR
from TDA_PITCH.models.layers.dila_conv_layers import dila_conv_block
from TDA_PITCH.settings import TrainingParams, Constants, SpectrogramSetting
from TDA_PITCH.utilities.evaluation_utils import Eval
import TDA_PITCH.utilities.utils as ut

torch.autograd.set_detect_anomaly(True)


class PianorollEstimatorModel(BaseModel):
    """pianoroll Estimator model.

        Args:
            in_channels: number of channels of the input feature.
            freq_bins: number of frequency bins of the input feature.
        """

    def __init__(self, in_channels: int, freq_bins: int):
        super().__init__()

        # convolutional blocks
        self.conv_block1 = ConvBlock_PR(in_channels=in_channels, out_channels=20)
        self.conv_block2 = ConvBlock_PR(in_channels=20, out_channels=40)

        # flatten convolutional layer output and feed output to a linear layer, followed by a batch normalisation
        self.fc3 = nn.Linear(in_features=freq_bins * 40, out_features=200, bias=False)
        self.bn3 = nn.BatchNorm1d(200)

        # 2 bi-GRU layers followed by a time-distributed dense output layer
        self.gru = nn.GRU(input_size=200, hidden_size=100, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(200, 88, bias=True)

        # initialise model weights and loss function
        self.init_weight()
        self.init_loss_function()
        self.init_parameters()

    def init_weight(self):
        """Initialise model weights"""
        init_layer(self.fc3)
        init_bn(self.bn3)
        init_gru(self.gru)
        init_layer(self.fc)

    def init_loss_function(self):
        """Initialise loss function"""
        self.loss_fn = nn.MSELoss()

    def init_parameters(self):
        self.learning_rate = TrainingParams.LEARNING_RATE
        self.weight_decay = 2e-5
        self.schedular_step_size = 2
        self.schedular_gamma = 0.9
        self.moniter_metric = 'valid_loss'

    def forward(self, spectrogram: torch.Tensor, inplace: bool = False):
        """Get model output.

        Parameters:
            spectrogram: (dtype: float, shape: (batch_size, in_channels, freq_bins, input_length)), audio spectrogram as input.
            inplace: (dtype: bool), applys the inplace flag to the dropout layers (use only during inference)
        Returns:
            pianoroll: (torch.Tensor, dtype: float, shape: (batch_size, 88, input_length)), pianoroll output
        """
        conv_hidden = self.conv_block1(spectrogram)  # [batch_size, 20, freq_bins, input_length]
        conv_hidden = F.dropout(conv_hidden, p=0.2, training=self.training, inplace=inplace)  # same as above
        conv_output = self.conv_block2(conv_hidden)  # [batch_size, 40, freq_bins, input_length]
        conv_output = F.dropout(conv_output, p=0.2, training=self.training, inplace=inplace)  # same as above

        conv_output = conv_output.transpose(1, 3).flatten(2)  # [batch_size, input_length, freq_bins*40]
        linear_output = F.relu(
            self.bn3(self.fc3(conv_output).transpose(1, 2)).transpose(1, 2))  # [batch_size, input_length, 200]
        linear_output = F.dropout(linear_output, p=0.5, training=self.training, inplace=inplace)  # same as above

        rnn_output, _ = self.gru(linear_output)  # [batch_size, input_length, 200]
        rnn_output = F.dropout(rnn_output, p=0.5, training=self.training, inplace=inplace)  # same as above
        pianoroll = F.elu(self.fc(rnn_output).transpose(1, 2))  # [batch_size, 88, input_length]

        return pianoroll

    def prepare_batch_data(self, batch):
        spectrogram, pianoroll, pianoroll_mask = batch
        spectrogram = spectrogram.float()
        pianoroll_targ = pianoroll.float()
        pianoroll_mask = pianoroll_mask.float()
        return (spectrogram, pianoroll_mask), pianoroll_targ  # input_data, target_data

    def predict(self, input_data, inplace: bool = False):
        spectrogram, pianoroll_mask = input_data
        pianoroll_pred = self(spectrogram, inplace=inplace) * pianoroll_mask  # add mask to ignore paddings
        return pianoroll_pred  # output_data

    def predict_step(self, batch, batch_index):
        spectrogram, _, _ = batch
        spectrogram = spectrogram.float()
        output_data = self(spectrogram, inplace=True)

        batch_multitracks = []
        # Take each output data in batch convert it to midi
        for i in range(TrainingParams.BATCH_SIZE):
            pr_output = (output_data[i, :, :] > Constants.velocity_threshold).bool()
            # [88 X Input_Length] => [128 x Input_Length] for piano roll vectors
            pr_output_full = torch.zeros(128, Constants.pianoroll_input_length)
            pr_output_full[21:21+88, :] = pr_output
            # [128 X Input_Length] => [Input_Length x 128] for the correct format of MIDI files
            pr_output_MIDI = torch.transpose(pr_output_full, dim0=0, dim1=1)
            track = pypianoroll.StandardTrack(pianoroll=pr_output_MIDI.numpy())
            # NOTE: The Multitrack class is used for piano roll viz and creating .mid files
            multitrack = pypianoroll.Multitrack(name=rf'Batch_Index={batch_index}_{i}', resolution=24)
            multitrack.append(track=track)

            # Create a midi file and store it in the outputs directory under the {batch_index}_index
            output_path = 'outputs/'
            ut.mkdir(output_path)
            output_file_path = os.path.join(output_path, rf'Batch_{batch_index}_{i}.mid')
            pypianoroll.write(path=output_file_path, multitrack=multitrack)

            # Append the multitrack object to the multitracks list and return it
            batch_multitracks.append(multitrack)

        return batch_multitracks


    def get_loss(self, output_data, target_data):
        loss = self.loss_fn(output_data, target_data)
        return loss

    def evaluate(self, output_data, target_data):
        pianoroll_pred = output_data
        pianoroll_targ = target_data
        # F-measure
        ps, rs, fs, accs = [], [], [], []
        results_n_on, results_n_onoff = [], []
        for i in range(TrainingParams.BATCH_SIZE):
            pr_target = (pianoroll_targ[i, :, :] > Constants.velocity_threshold).bool()
            pr_output = (pianoroll_pred[i, :, :] > Constants.velocity_threshold).bool()

            # framewise evaluation
            p, r, f, acc = Eval.framewise_evaluation(pr_output, pr_target)
            ps, rs, fs, accs = ps + [p], rs + [r], fs + [f], accs + [acc]

            # notewise evaluation
            result_n_on, result_n_onoff = Eval.notewise_evaluation(pr_target, pr_output,
                                                                   hop_time=Constants.pianoroll_hop_time)
            if result_n_on is not None:
                results_n_on.append(result_n_on)
                results_n_onoff.append(result_n_onoff)

        ps_n_on, rs_n_on, fs_n_on = list(zip(*results_n_on))
        ps_n_onoff, rs_n_onoff, fs_n_onoff = list(zip(*results_n_onoff))

        # return logs
        logs = {'epoch': self.current_epoch,
                'precision': np.mean(ps),
                'recall': np.mean(rs),
                'f-score': np.mean(fs),
                'accuracy': np.mean(accs),
                'precision_n_on': np.mean(ps_n_on),
                'recall_n_on': np.mean(rs_n_on),
                'f-score_n_on': np.mean(fs_n_on),
                'precision_n_onoff': np.mean(ps_n_onoff),
                'recall_n_onoff': np.mean(rs_n_onoff),
                'f-score_n_onoff': np.mean(fs_n_onoff)}
        return logs

class F0EstimatorModel(BaseModel):
    def __init__(self,
                 n_har=12,
                 bins_per_octave=SpectrogramSetting.bins_per_octave,
                 dilation_modes=['log_scale', 'fixed', 'fixed', 'fixed'],
                 dilation_rates=[48,48,48,48],
                 channels=[32,64,128,128],
                 dil_kernel_sizes=[[1,3],[1,3],[1,3],[1,3]]):
        super().__init__()
        self.dilation_modes = dilation_modes
        self.n_har = n_har
        self.bins_per_octave = bins_per_octave

        # [b x 1 x T x 88*8] => [b x 32 x T x 88*4]
        self.block_1 = dila_conv_block(1, channels[0], self.bins, n_har=n_har, dilation_mode=dilation_modes[0],
                                       dilation_rate=dilation_rates[0], dil_kernel_size=dil_kernel_sizes[0],
                                       kernel_size=[3, 3], padding=[1, 1])

        self.half_bins = self.bins // 2
        # => [b x 64 x T x 88*4]
        self.block_2 = dila_conv_block(channels[0], channels[1], self.half_bins, 3, dilation_mode=dilation_modes[1],
                                       dilation_rate=dilation_rates[1], dil_kernel_size=dil_kernel_sizes[1],
                                       kernel_size=[3, 3], padding=[1, 1])
        # => [b x 128 x T x 88*4]
        self.block_3 = dila_conv_block(channels[1], channels[2], self.half_bins, 3, dilation_mode=dilation_modes[2],
                                       dilation_rate=dilation_rates[2], dil_kernel_size=dil_kernel_sizes[2],
                                       kernel_size=[3, 3], padding=[1, 1])
        # => [b x 128 x T x 88*4]
        self.block_4 = dila_conv_block(channels[2], channels[3], self.half_bins, 3, dilation_mode=dilation_modes[3],
                                       dilation_rate=dilation_rates[3], dil_kernel_size=dil_kernel_sizes[3],
                                       kernel_size=[3, 3], padding=[1, 1])

        self.conv_5 = nn.Conv2d(channels[3], channels[3] // 2, kernel_size=[1, 1])
        self.conv_6 = nn.Conv2d(channels[3] // 2, 1, kernel_size=[1, 1])

        # initialise loss function and parameters
        self.init_loss_function()

    def forward(self, spectrogram):
        # input: [b x freq_bins=352 x num_frames]
        # output: [b x freq_bins=352 x num_frames ]

        # [b x 1 x n_bins x num_frames] => [b x 1 x num_frames x n_bins]
        #x = spectrogram[None, :]
        x = torch.transpose(spectrogram, dim0=1, dim1=2)

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)

        # [b x 128 x T x 352] => [b x 64 x T x 352]
        x = self.conv_5(x)
        x = torch.relu(x)
        x = self.conv_6(x)
        x = torch.sigmoid(x)

        x = torch.squeeze(x, dim=1)
        # x = torch.clip(x, 1e-4, 1 - 1e-4)
        x = torch.transpose(x, dim0=1, dim1=2)
        # => [b x freq_bins=352 x num_frames]
        return x

    def init_loss_function(self):
        """Initialise loss function"""
        self.loss_fn_weight = torch.full((352,),20) # Following HarmoF0 recommendation for MIR-1K
        self.loss_fn = nn.CrossEntropyLoss(weight=self.loss_fn_weight, reduction='sum')

    def init_parameters(self):
        self.learning_rate = TrainingParams.LEARNING_RATE
        self.weight_decay = 2e-5
        self.schedular_step_size = 2
        self.schedular_gamma = 0.9
        self.moniter_metric = 'valid_loss'

    def prepare_batch_data(self, batch):
        spectrogram, pitch_vector, pitch_vector_mask = batch
        spectrogram = spectrogram.float()
        pitch_vector_targ = pitch_vector.float()
        pitch_vector_mask = pitch_vector_mask
        return (spectrogram, pitch_vector_mask), pitch_vector_targ

    def predict(self, input_data):
        spectrogram, pitch_vector_mask = input_data
        pitch_vector_pred = self(spectrogram) * pitch_vector_mask
        return pitch_vector_pred  # Output_data

    def evaluate(self, output_data, target_data):
        pitch_vector_pred = output_data
        pitch_vector_targ = target_data

        # RPA and RCA
        RPA, RCA = [], []
        for i in range(TrainingParams.BATCH_SIZE):
            pr_target = pitch_vector_targ[i, :, :]
            pr_output = pitch_vector_pred[i, :, :]

            # Total mir_eval.melody scores evaluation
            scores = Eval.f0_annotations_evaluation(pr_output, pr_target, hop_length=Constants.pitch_vector_hop_time)
            RPA_individual = scores['Raw Pitch Accuracy']
            RCA_individual = scores['Raw Chroma Accuracy']

            RPA.append(RPA_individual)
            RCA.append(RCA_individual)

        # return logs
        logs = {'epoch': self.current_epoch,
                'Raw Pitch Accuracy (RPA)': np.mean(RPA),
                'Raw Chroma Accuracy (RCA)': np.mean(RCA)}
        return logs