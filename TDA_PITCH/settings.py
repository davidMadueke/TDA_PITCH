
class Constants:
    sample_rate = 16000
    hop_time = 0.01 # For Pretty Midi piano Rolls
    # TODO change hop_time to be in concordance with specgram hop size
    segment_length = 2 # transcribe 2s segments at a time
    input_length = int(segment_length * (1 / hop_time))
    velocity_threshold = 30

    spectrogram_max_length = 802
    pianoroll_max_length = 401

    pianos = ['Gentleman', 'Giant', 'Grandeur', 'Maverick']


class SpectrogramSetting(object):
    """Spectrogram setting

    refer to metadata/spectrogram_settings.csv for different parameters in experiment.
    """

    type = 'Reassigned_log2'  # select from ['Reassigned_log2', 'STFT_log2', 'mel']
    types = {"1": 'Reassigned_log2', "2": 'STFT_log2'}
    sample_rate = Constants.sample_rate
    win_length = 1024
    n_fft = 1024
    n_mels = 256
    bins_per_octave = 12*4
    n_octaves = 8
    n_harms = 6
    freq_bins = 480
    freq_per_octave = freq_bins / bins_per_octave
    f_min = 27.5
    log2_f_max = f_min * (2 ** freq_per_octave)
    channels = 1

    def to_string(self):
        if self.type == 'Reassigned_log2':
            return '-'.join([self.type,
                             f'sample_rate={self.sample_rate}',
                             f'win_length={self.win_length}'])
        if self.type == 'STFT_log2':
            return '-'.join([self.type,
                             f'sample_rate={self.sample_rate}',
                             f'win_length={self.win_length}'])
        elif self.type == 'Mel':
            return '-'.join([self.type,
                             f'sample_rate={self.sample_rate}',
                             f'win_length={self.win_length}',
                             f'n_mels={self.n_mels}'])


class TrainingParam:
    learning_rate = 0.001
    batch_size = 8

# We are creating two different networks, a fully connected CNN f0 estimator and a PianoRoll
# Transcriber Estimator that combines early CNN layers follwoed by Bi-GRU units

class ModelSetting(object):
    type = 'f0_Estimator' # select from ['f0_estimator', 'pianoroll_estimator']

    def to_string(self):
        if self.type == 'f0_estimator':
            return '-'.join([self.type])
        if self.type == 'pianoroll_estimator':
            return '-'.join([self.type])