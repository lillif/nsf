"""
Training script copied from M3LEO: https://github.com/spaceml-org/M3LEO
"""

from __future__ import annotations

import os
import sys

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


import dotenv
import hydra
import lightning.pytorch as pl
import omegaconf
import torch
import wandb
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

# MixedPrecisionPlugin
from loguru import logger
from omegaconf import DictConfig

from utils import find_hydra_run_path, load_ckpt_from_hydra_run

os.environ["HYDRA_FULL_ERROR"] = "1"


# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir

# load environment variables from `.env` file if it exists
dotenv.load_dotenv(override=True)


@hydra.main(
    version_base="1.3",
    config_path="configs/goes_small",
    config_name="train.yaml",
)
def main(config: DictConfig):
    # ------- seeds -------

    # extract and set model and data seeds
    seed = config.seed if "seed" in config else 42
    logger.info(f"training with seed {seed}")
    seed_everything(seed, workers=True)

    # ------- wandb logging -------

    # set up wandb config
    wandb.config = omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )

    # get wandb experiment tags from config
    tags = config.tags if "tags" in config else []
    if isinstance(tags, str):
        tags = tags.split()

    # get experiment name from config
    experiment_name = config.experiment_name

    # set up wandb logger
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.info(f"output dir: {output_dir}")

    wandb_logger = WandbLogger(
        name=experiment_name,
        project=config.wandb.project,
        entity=config.wandb.entity,
        mode=config.wandb.mode,
        tags=tags,
        save_dir=output_dir,
    )

    # log command to wandb
    log_cmd_wandb = config.log_cmd_wandb if "log_cmd_wandb" in config else False
    logger.info(log_cmd_wandb)
    if log_cmd_wandb:
        cmd = " ".join(sys.argv)
        logger.info(f"Command executed: {cmd}")

    # log config to wandb
    log_config_as = config.log_config_as if "log_config_as" in config else "yaml"
    if log_config_as == "yaml":
        yaml_str = omegaconf.OmegaConf.to_yaml(config)
        logger.debug(f"Hydra-config: {yaml_str}")
    else:
        logger.debug(f"Hydra-config: {config}")

    # ------- dataloader -------
    logger.info("instantiating dataloader")
    dataloader = hydra.utils.instantiate(config.dataloader)

    # ------- model -------

    # load checkpoint or instantiate model from scratch
    if "load_checkpoint" in config.keys():
        hr = find_hydra_run_path(
            outputs_dir=config.load_checkpoint.outputs_dir,
            wandb_runid=config.load_checkpoint.wandb_runid,
        )
        logger.info(f"hydra run path for previous model: {hr}")
        model = load_ckpt_from_hydra_run(hr)
    else:
        logger.info("instantiating model")
        model = hydra.utils.instantiate(config.model)

    # ------- callbacks -------

    # Checkpoint callback
    # Define the checkpoint callback path
    dirpath = os.path.join(output_dir, "checkpoints")

    # Define the monitored metric
    monitor_metric = (
        config.get("monitor_metric") if config.get("monitor_metric") else "val/loss"
    )
    # Define the checkpoint callback
    val_checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        monitor=monitor_metric,
        save_top_k=1,
        save_last=True,
        mode="min",
        filename=f"{experiment_name}" + "-best-val-loss-epoch={epoch:02d}",
        auto_insert_metric_name=False,
    )
    callbacks = [
        val_checkpoint_callback,
    ]

    # Define whether to log additional metrics to wandb
    # Defaults to False if not included in config
    log_metrics = config.get("log_metrics") if config.get("log_metrics") else False

    if log_metrics:
        logger.info("logging ")
        # Define the checkpoint callback
        accuracy_checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor=config.get("monitor_metric") if config.get("log_metrics") else "val/accuracy",
            save_top_k=1,
            mode="max",
            filename=f"{experiment_name}" + "-highest-val-acc-epoch={epoch:02d}",
            auto_insert_metric_name=False,
        )

        callbacks.append(accuracy_checkpoint_callback)

    # ------- training details -------

    # Define plugins for trainer
    plugins = None

    # Define the precision for the model
    precision = config.get("precision") if config.get("precision") else "32-true"

    # Define whether to accumulate gradients before running optimizer
    accumulate_grad_batches = (
        config.get("accumulate_grad_batches")
        if config.get("accumulate_grad_batches")
        else 1
    )
    # Define whether to use deterministic algorithms
    if config.get("use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)

    # Define when to run validation loop
    check_val_every_n_epoch = (
        config.check_val_every_n_epoch
        if hasattr(config, "check_val_every_n_epoch")
        else 1
    )
    # Define strategy for trainer
    strategy = (
        config.get("strategy")
        if config.get("strategy")
        else "auto"
    )

    ## get gpu settings from config if available
    accelerator = "gpu" if config.get("gpus") and config.get("gpus") > 0 else "auto"
    devices = config.get("gpus") if config.get("gpus") else 1



    logger.info(f"Using accelerator: {accelerator} with {devices} devices")

    # ------- training -------
    trainer = pl.Trainer(
        num_nodes=config.get("num_nodes", 1),
        accelerator=accelerator,
        strategy=strategy,
        devices=devices,
        plugins=plugins,
        max_epochs=config.max_epochs,
        precision=precision,
        log_every_n_steps=config.log_every_n_steps,
        logger=wandb_logger,
        callbacks=callbacks,
        fast_dev_run=False,
        limit_train_batches=config.limit_train_batches,
        limit_val_batches=config.limit_val_batches,
        limit_test_batches=config.limit_test_batches,
        accumulate_grad_batches=accumulate_grad_batches,
        check_val_every_n_epoch=check_val_every_n_epoch,
        num_sanity_val_steps=0,  # Disable sanity check validation steps
    )

    trainer.fit(model, dataloader)

    # ------- testing -------
    logger.info(f"---- getting best model from {output_dir}")
    best_model = load_ckpt_from_hydra_run(output_dir)

    trainer.test(model=best_model, datamodule=dataloader)


if __name__ == "__main__":
    main()
