import torch
import torchaudio
import math
import numpy as np
import librosa

class WaveformToLogSpecgram:
    def __init__(self, sample_rate, n_fft, fmin, bins_per_octave, freq_bins, frame_len, hop_length=320):  # , device

        self.fmin = fmin
        e = freq_bins / bins_per_octave
        self.fmax = self.fmin * (2 ** e)

        self.n_fft = n_fft
        hamming_window = torch.hann_window(self.n_fft)  # .to(device)
        # => [1 x 1 x n_fft]
        self.hamming_window = hamming_window[None, :]


        self.sample_rate = sample_rate
        fre_resolution = self.sample_rate / n_fft

        self.idxs = torch.arange(0, freq_bins)  # , device=device

        self.log_idxs = self.fmin * (2 ** (self.idxs / bins_per_octave)) / fre_resolution

        # Linear interpolation： y_k = y_i * (k-i) + y_{i+1} * ((i+1)-k)
        self.log_idxs_floor = torch.floor(self.log_idxs).long()
        self.log_idxs_floor_w = (self.log_idxs - self.log_idxs_floor).reshape([1, freq_bins])
        self.log_idxs_ceiling = torch.ceil(self.log_idxs).long()
        self.log_idxs_ceiling_w = (self.log_idxs_ceiling - self.log_idxs).reshape([1, freq_bins])

        self.waveform_to_specgram = torchaudio.transforms.Spectrogram(n_fft, hop_length=hop_length)  # .to(device)

        assert (bins_per_octave % 12 == 0)
        bins_per_semitone = bins_per_octave // 12

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80, stype='magnitude')
        self.frame_len = frame_len
        self.hop_length = hop_length
        self.num_frames = None

        self.reassignedtimes = None

    def stft_process(self, waveforms):
        # inputs: [num_frames x frame_len]
        # outputs: [num_frames x n_bins]

        if isinstance(waveforms, np.ndarray):
            waveforms = torch.tensor(waveforms).squeeze(dim=0)

        # start from the 0
        # Pad the waveform properly
        waveform_pad = torch.cat([waveforms[:self.frame_len // 2].flip(dims=[0]), waveforms], dim=0)

        wav_len = len(waveforms)
        self.num_frames = int((wav_len - self.frame_len) // self.hop_length) + 1
        frames = torch.zeros([self.num_frames, self.frame_len])
        for i in range(self.num_frames):
            begin = i * self.hop_length
            end = begin + self.frame_len
            frames[i, :] = waveform_pad[begin:end]

        waveformFrames = frames[:, :] * self.hamming_window

        specgram = torch.stft(waveformFrames,
                              n_fft=self.n_fft,
                              hop_length=self.hop_length,
                              window=None,
                              center=False,
                              return_complex=True).permute(2, 0, 1)
        specgram = torch.abs(specgram)
        specgram = specgram**2
        # => [num_frames x n_fft//2 ]

        # Interpolate log spectrogram
        specgram = (specgram[:,:, self.log_idxs_floor] * self.log_idxs_floor_w +
                    specgram[:, :, self.log_idxs_ceiling] * self.log_idxs_ceiling_w)
        # => [freq_bins x T]
        specgram = torch.transpose(specgram, 1, 2)

        specgram_normalised = specgram / torch.max(specgram)
        specgram_db = self.amplitude_to_db(specgram_normalised)
        # specgram_db = specgram_db[:, :, :-1] # remove the last frame.
        # specgram_db = specgram_db.permute([0, 2, 1])
        specgram_db_2 = specgram_db.squeeze().numpy()
        return specgram_db

    def reassigned_process(self, waveforms):
        # inputs: [num_frames x frame_len]
        # outputs: [num_frames x n_bins]

        if isinstance(waveforms, np.ndarray):
            waveforms = torch.tensor(waveforms).squeeze(dim=0)

        # start from the 0
        # Pad the waveform properly
        waveform_pad = torch.cat([waveforms[:self.frame_len // 2].flip(dims=[0]), waveforms], dim=0)

        # NOTE - Reassigned Spectrogram has inherent randomness closer to the low end that could ocurr
        _, self.reassignedtimes, specgram = librosa.reassigned_spectrogram(y=waveform_pad.numpy(),
                                                                           sr=self.sample_rate,
                                                                           window='hann',
                                                                           n_fft=512 * 2,
                                                                           hop_length=self.hop_length,
                                                                           center=False,
                                                                           reassign_times=True,
                                                                           pad_mode='reflect')
        self.num_frames = self.reassignedtimes.shape[1]  # Num frames is the number of frames taken from reassignedTimes
        specgram = torch.tensor(specgram)
        # specgram = torch.abs(specgram)
        # specgram = specgram ** 2
        # => [num_frames x n_fft//2 ]

        # Interpolate log spectrogram
        specgram = (specgram[self.log_idxs_floor, :] * torch.transpose(self.log_idxs_floor_w, 0, 1) +
                    specgram[self.log_idxs_ceiling, :] * torch.transpose(self.log_idxs_ceiling_w, 0, 1))
        # => [freq_bins x T]
        # specgram = torch.transpose(specgram)

        specgram_normalised = specgram / torch.max(specgram)
        specgram_db = self.amplitude_to_db(specgram_normalised)
        # specgram_db = specgram_db[:, :-2] # remove the last two frames.
        # specgram_db = specgram_db.permute([0, 2, 1])
        return specgram_db
