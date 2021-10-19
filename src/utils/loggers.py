from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger, WandbLogger


def get_tensorboard_logger(trainer: Trainer) -> TensorBoardLogger:
    """Safely get TensorBoard logger from Trainer."""

    if isinstance(trainer.logger, TensorBoardLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, TensorBoardLogger):
                return logger

    raise Exception("TensorBoardLogger not found in trainer.")


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )
