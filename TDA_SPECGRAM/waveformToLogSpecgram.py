import torch
import torchaudio
import math
import numpy as np

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

        log_idxs = fmin * (2 ** (idxs / bins_per_octave)) / fre_resolution

        # Linear interpolation： y_k = y_i * (k-i) + y_{i+1} * ((i+1)-k)
        self.log_idxs_floor = torch.floor(log_idxs).long()
        self.log_idxs_floor_w = (log_idxs - self.log_idxs_floor).reshape([1, 1, freq_bins])
        self.log_idxs_ceiling = torch.ceil(log_idxs).long()
        self.log_idxs_ceiling_w = (self.log_idxs_ceiling - log_idxs).reshape([1, 1, freq_bins])

        self.waveform_to_specgram = torchaudio.transforms.Spectrogram(n_fft, hop_length=hop_length)  # .to(device)

        assert (bins_per_octave % 12 == 0)
        bins_per_semitone = bins_per_octave // 12

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
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
        # waveform = F.pad(waveform, [self.frame_len//2, 0], mode='reflect')
        waveform_pad = waveforms[:, :self.frame_len // 2].flip(dims=[1, ])
        torch.cat([waveform_pad, waveforms], dim=1)

        b, wav_len = waveforms.shape
        assert b == 1
        num_frames = int((wav_len - self.frame_len) // self.hop_length) + 1
        batch = torch.zeros([1, num_frames, self.frame_len])
        for i in range(num_frames):
            begin = i * self.hop_length
            end = begin + self.frame_len
            batch[:, i, :] = waveforms[:, begin:end]

        waveforms = batch[0,:,:] * self.hamming_window
        specgram = torch.fft.fft(waveforms)
        specgram = torch.abs(specgram[:, :, :self.n_fft // 2 + 1])
        specgram = specgram * specgram
        # => [num_frames x n_fft//2 x 1]
        # specgram = torch.unsqueeze(specgram, dim=2)

        # => [b x T x freq_bins]
        specgram = specgram[:, :, self.log_idxs_floor] * self.log_idxs_floor_w + specgram[:, :,
                                                                                 self.log_idxs_ceiling] * self.log_idxs_ceiling_w

        # => [b x freq_bins x T]
        specgram = torch.transpose(specgram, 1, 2)
        specgram_db = self.amplitude_to_db(specgram)
        # specgram_db = specgram_db[:, :, :-1] # remove the last frame.
        # specgram_db = specgram_db.permute([0, 2, 1])
        return specgram_db
