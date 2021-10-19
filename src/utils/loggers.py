from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger, WandbLogger
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.utils import make_grid


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


def log_images(
    name: str,
    images: Union[torch.Tensor, np.ndarray, Image.Image],
    trainer: Optional[Trainer] = None,
    global_step: Optional[int] = None,
    **kwargs,
):
    if global_step is None:
        global_step = trainer.global_step if trainer is not None else None
    if isinstance(images, torch.Tensor):
        if images.dim() == 4:  # [N, C, H, W]
            image = make_grid(
                images.cpu(),
                nrow=kwargs.get("nrow", 8),
                padding=kwargs.get("padding", 2),
                normalize=kwargs.get("normalize", False),
                value_range=kwargs.get("value_range", None),
                scale_each=kwargs.get("scale_each", False),
                pad_value=kwargs.get("pad_value", 0),
            )
        elif images.dim() == 3:  # [C, H, W]
            image = images
    elif isinstance(images, np.ndarray):
        image = to_tensor(images)
    elif isinstance(images, Image.Image):
        image = to_tensor(images)

    try:
        import wandb

        run = get_wandb_logger(trainer).experiment
    except Exception:
        pass
    else:
        run.log({name: wandb.Image(to_pil_image(image), caption=f"{name}_{global_step}")})

    try:
        run = get_tensorboard_logger(trainer).experiment
    except Exception:
        pass
    else:
        run.add_image(name, image, global_step=global_step)
