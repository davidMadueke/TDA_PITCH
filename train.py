import warnings
import pytorch_lightning as pl
import torch

from TDA_PITCH import SpectrogramSetting
from TDA_PITCH.data.datamodules import PianoRollEstimatorDataModule, F0EstimatorDataModule
from TDA_PITCH.models.models import PianorollEstimatorModel, F0EstimatorModel
from TDA_PITCH.settings import TaskSetting

warnings.filterwarnings('ignore')


def train(dataset_folder, feature_folder,
          spectrogram_setting: SpectrogramSetting,
          task_setting: TaskSetting):
    # get datamodule, model and experiment_name
    if task_setting.type == 'pianoroll_estimator':
        model = PianorollEstimatorModel(in_channels=spectrogram_setting.channels,
                                        freq_bins=spectrogram_setting.freq_bins)
        datamodule = PianoRollEstimatorDataModule(spectrogram_setting=spectrogram_setting,
                                                  dataset_folder=dataset_folder,
                                                  feature_folder=feature_folder)
        experiment_name = f'{task_setting.to_string()}-{spectrogram_setting.to_string()}-base_model'
    elif task_setting.type == 'f0_estimator':
        model = F0EstimatorModel(bins_per_octave=spectrogram_setting.bins_per_octave)
        datamodule = F0EstimatorDataModule(spectrogram_setting=spectrogram_setting,
                                           dataset_folder=dataset_folder,
                                           feature_folder=feature_folder)
        experiment_name = f'{task_setting.to_string()}-{spectrogram_setting.to_string()}-base_model'

    # get trainer and train
    logger = pl.loggers.TensorBoardLogger('tensorboard_logs',
                                          name=experiment_name,
                                          default_hp_metric=False)
    trainer = pl.Trainer(
        accelerator='gpu',
        logger=logger,
        log_every_n_steps=20,
        reload_dataloaders_every_n_epochs=True,
        enable_checkpointing=True,
        enable_model_summary=True,
        enable_progress_bar=True,
        max_epochs=10,

        # Debug flags
        overfit_batches=OVERFIT_BATCH_NUMBER,  # For debug, make sure this is 0 when training
        fast_dev_run=False,
        detect_anomaly=False,
    )
    trainer.fit(model=model, datamodule=datamodule)

    #prediction = trainer.predict(model=model, datamodule=datamodule, ckpt_path="last")
    #print(prediction[0])

if __name__ == '__main__':
    OVERFIT_BATCH_NUMBER = 0
    dataset_folder = r"C:\Users\David\Documents\GitHub\DATASETS FOR PROJECTS\MuseSyn"
    features_folder = r"C:\Users\David\Documents\GitHub\DATASETS FOR PROJECTS\MuseSyn\features"

    # train
    specgram_setting = SpectrogramSetting()
    task = TaskSetting()
    train(dataset_folder=dataset_folder, feature_folder=features_folder,
          spectrogram_setting=specgram_setting, task_setting=task)