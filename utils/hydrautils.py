"""
Some functions copied from M3LEO: https://github.com/spaceml-org/M3LEO
"""

from __future__ import annotations

import functools
import os
from glob import glob

import hydra
import lightning.pytorch as pl
import torch
from loguru import logger
from omegaconf import OmegaConf


def find_hydra_run_path(outputs_dir, wandb_runid):
    """
    find the hydra run path that contains the wandb runid
    """

    files = glob(f"**/*{wandb_runid}*", root_dir=outputs_dir, recursive=True)
    if len(files) == 0:
        raise ValueError("no file found")

    files = [f for f in files if "/wandb/" in f]
    if len(files) == 0:
        raise ValueError("no wandb log found")

    r = os.path.join(outputs_dir, files[0].split("/wandb/")[0])
    return r


def load_ckpt_from_hydra_run(
    hydra_run_path: str,
    loading_from_state_dict=True,
    enable_loading_weights: bool = True,
    model: pl.LightningModule = None,
    config=None,
) -> pl.LightningModule:
    """
    loads a checkpoint model from a run's output hydra log
    hydra_run_path: the file path to the hydra run
    loading_from_state_dict: if True, load model using state_dict, otherwise use PyTorch Lightning checkpoint
    config: pass config if already loaded
    model: pass model if already instantiated
    returns: a pytorch lighting module
    """

    # load config
    if config is None:
        config_file = f"{hydra_run_path}/.hydra/config.yaml"
        if not os.path.isfile(config_file):
            raise ValueError(f"config file {config_file} not found")

        config = OmegaConf.load(config_file)

    # look for checkpoint
    ckpts_paths = [
        f"{hydra_run_path}/*{'/*'*trailings}/*ckpt" for trailings in range(6)
    ]
    ckpts = functools.reduce(
        lambda lista, elemento: glob(elemento) + lista, ckpts_paths, []
    )
    ckpts = sorted(ckpts)
    print(ckpts)
    if len(ckpts) == 0:
        raise ValueError(f"no checkpoints found in {hydra_run_path}")

    if len(ckpts) > 1:
        print(f"there are {len(ckpts)} checkpoints, attempting to use the best one")
        try:
            best_ckpt = glob(f"{hydra_run_path}/*/*best*")[0]
        except IndexError:  # if no "best" model exists
            print(f"could not load best model, attempting last model instead")
            best_ckpt = ckpts[-1]
    else:
        print(f"there is {len(ckpts)} checkpoint")
        best_ckpt = ckpts[-1]

    logger.info(f"loaded checkpoint: {best_ckpt}")

    if model is None:
        logger.info("Creating Model")
        # instantiate model class
        model = hydra.utils.instantiate(config.model)

    # load model
    if enable_loading_weights:
        logger.info("Loading weights from model checkpoint")
        if loading_from_state_dict:
            logger.info("Loading using state_dict")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
        else:
            logger.info("Loading using checkpoint from PyTorch Lightning")
            model = model.load_from_checkpoint(best_ckpt)

        logger.info("---------------------------------")
        logger.info(f"model checksum  {print_checksum_of_model(model):.4f}")
        logger.info("---------------------------------")
    else:
        logger.warning(
            "The weights of the pretrained model are not loaded. The model will be initialized with random weights."
        )
        logger.warning("---------------------------------")
        logger.warning(f"model checksum  {print_checksum_of_model(model):.4f}")
        logger.info("---------------------------------")

    return model


def print_checksum_of_model(model):
    return sum(torch.abs(p).sum() for p in model.parameters()).detach().cpu().numpy()
