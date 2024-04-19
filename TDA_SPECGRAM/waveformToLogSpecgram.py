import torch
import torchaudio
import math
import numpy as np
import librosa

class WaveformToLogSpecgram:
    def __init__(self, sample_rate, n_fft, fmin, bins_per_octave, freq_bins, frane_len, hop_length=320):  # , device

        e = freq_bins / bins_per_octave
        fmax = fmin * (2 ** e)

        self.n_fft = n_fft
        hamming_window = torch.hann_window(self.n_fft)  # .to(device)
        # => [1 x 1 x n_fft]
        self.hamming_window = hamming_window[None, None, :]

        # torch.hann_window()

        self.sample_rate = sample_rate
        fre_resolution = self.sample_rate / n_fft

        idxs = torch.arange(0, freq_bins)  # , device=device

        self.log_idxs = fmin * (2 ** (idxs / bins_per_octave)) / fre_resolution

        # Linear interpolation： y_k = y_i * (k-i) + y_{i+1} * ((i+1)-k)
        self.log_idxs_floor = torch.floor(self.log_idxs).long()
        self.log_idxs_floor_w = (self.log_idxs - self.log_idxs_floor).reshape([1, 1, freq_bins])
        self.log_idxs_ceiling = torch.ceil(self.log_idxs).long()
        self.log_idxs_ceiling_w = (self.log_idxs_ceiling - self.log_idxs).reshape([1, 1, freq_bins])

        self.waveform_to_specgram = torchaudio.transforms.Spectrogram(n_fft, hop_length=hop_length)  # .to(device)

        assert (bins_per_octave % 12 == 0)
        bins_per_semitone = bins_per_octave // 12

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80, stype='magnitude')
        self.frame_len = frane_len
        self.hop_length = hop_length

    def process(self, waveforms):
        # inputs: [b x num_frames x frame_len]
        # outputs: [b x num_frames x n_bins]

        if isinstance(waveforms, np.ndarray):
            waveforms = torch.tensor(waveforms)
        if (len(waveforms.size()) == 1):
            waveforms = waveforms[None, :]

        # start from the 0
        # Pad the waveform properly
        waveform_pad = torch.cat([waveforms[:, :self.frame_len // 2].flip(dims=[1]), waveforms], dim=1)

        b, wav_len = waveforms.shape
        assert b == 1
        self.num_frames = int((wav_len - self.frame_len) // self.hop_length) + 1
        batch = torch.zeros([1, self.num_frames, self.frame_len])
        for i in range(self.num_frames):
            begin = i * self.hop_length
            end = begin + self.frame_len
            batch[:, i, :] = waveform_pad[:, begin:end]

        batchWaveforms = batch[0, :, :] * self.hamming_window
        batchWaveforms = batchWaveforms.squeeze(dim=0)  # Remove the batch dimension if it's not needed
        specgram = torch.stft(batchWaveforms,
                              n_fft=self.n_fft,
                              hop_length=self.hop_length,
                              window=None,
                              center=False,
                              return_complex=True).permute(2, 0, 1)
        specgram = torch.abs(specgram)
        specgram = specgram**2
        # => [num_frames x n_fft//2 x 1]
        # specgram = torch.unsqueeze(specgram, dim=2)

        # Interpolate log spectrogram
        specgram = specgram[:, :, self.log_idxs_floor] * self.log_idxs_floor_w + specgram[:, :,
                                                                              self.log_idxs_ceiling] * self.log_idxs_ceiling_w
        # => [b x freq_bins x T]
        specgram = torch.transpose(specgram, 1, 2)

        specgram_normalised = specgram / torch.max(specgram)
        specgram_db = self.amplitude_to_db(specgram_normalised)
        print('debug')
        # specgram_db = specgram_db[:, :, :-1] # remove the last frame.
        # specgram_db = specgram_db.permute([0, 2, 1])
        return specgram_db
