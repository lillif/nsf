from lightning import LightningDataModule
from loguru import logger
from torch.utils.data import DataLoader

from constants import SPLITS_DICT

from .olr_dataset import OlrDataset
from .olr_utils import get_list_olrfiles, get_split


class OlrDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        dataset_kwargs: dict,
        splits_dict: dict = SPLITS_DICT,
        filetype: str = "npz",
        batch_size: int = 16,
        num_workers: int = 1,
        pin_memory: bool = False,
        prefetch_factor: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory

        self.data_dir = data_dir
        self.splits_dict = splits_dict

        self.filetype = filetype

        self.batch_size = batch_size
        self.num_workers = num_workers

        # get all filenames in data_dir
        filenames = get_list_olrfiles(data_path=self.data_dir, ext=self.filetype)

        logger.info(f"Found {len(filenames)} files in {self.data_dir}")

        # split filenames based on train/test/val criteria
        train_files = get_split(filenames, splits_dict["train"])
        test_files = get_split(filenames, splits_dict["test"])
        val_files = get_split(filenames, splits_dict["val"])

        dataset_kwargs = dict(dataset_kwargs)  # ensure dataset_kwargs is a dictionary

        self.train_dataset = OlrDataset(
            **dataset_kwargs | {"filepaths": train_files},
        )

        self.test_dataset = OlrDataset(
            **dataset_kwargs | {"filepaths": test_files},
        )

        self.val_dataset = OlrDataset(
            **dataset_kwargs | {"filepaths": val_files},
        )

    def prepare_data(self):
        self.train_dataset.prepare_data()
        self.test_dataset.prepare_data()
        self.val_dataset.prepare_data()

    def setup(self, stage):
        self.train_dataset.setup(stage)
        self.test_dataset.setup(stage)
        self.val_dataset.setup(stage)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            persistent_workers=True,
            prefetch_factor=self.hparams.prefetch_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=self.hparams.prefetch_factor,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=self.hparams.prefetch_factor,
        )
