
class Constants:
    random_seed = 42 # For all the Pseudorandom RNG generators
    sample_rate = 16000
    pianoroll_frame_length = 160
    pianoroll_hop_time = pianoroll_frame_length / sample_rate # For Pretty Midi piano Rolls
    segment_length = 2  # transcribe 2s segments at a time
    pianoroll_input_length = int(segment_length * (1 / pianoroll_hop_time))
    # i.e desired_frames = desired_duration * sample_rate / frame length
    velocity_threshold = 30

    spectrogram_max_length = 802
    pianoroll_max_length = 401

    pianos = ['Gentleman', 'Giant', 'Grandeur', 'Maverick']


class SpectrogramSetting(object):
    """Spectrogram setting

    refer to metadata/spectrogram_settings.csv for different parameters in experiment.
    """

    types = {"Reassigned Spectrogram": 'Reassigned_log2',
             "Regular Spectrogram": 'STFT_log2'}

    type = types["Reassigned Spectrogram"]
    sample_rate = Constants.sample_rate
    win_length = 1024
    frame_len = 1024
    n_fft = 1024
    n_mels = 256
    bins_per_octave = 12*4
    n_octaves = 8
    n_harms = 6
    freq_bins = 88 * 4
    freq_per_octave = freq_bins / bins_per_octave
    f_min = 27.5
    log2_f_max = f_min * (2 ** freq_per_octave)
    hop_length = 320
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