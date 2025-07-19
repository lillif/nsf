from typing import Callable

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset


class OlrDataset(Dataset):
    def __init__(
        self,
        filepaths: list[str],
        transforms: Callable,
        filetype: str = "npz",
        load_coords: bool = False,
        npz_image_key: str = "image",
    ):
        self.filepaths = filepaths
        self.transforms = transforms
        self.filetype = filetype
        self.load_coords = load_coords
        self.npz_image_key = npz_image_key

    def setup(self, stage):
        pass

    def prepare_data(self):
        pass

    def __getitem__(self, idx) -> np.ndarray:
        item = {}
        if self.filetype == "npz":
            try:
                data = np.load(self.filepaths[idx])
            except Exception as e:
                logger.error(f"Error loading file {self.filepaths[idx]}: {e}")
                # try loading the next file instead
                # NOTE this is recursive!
                return self.__getitem__((idx + 1) % len(self.filepaths))
            if self.load_coords:
                item["lat"] = data["lat"][np.newaxis, ...]
                item["lon"] = data["lon"][np.newaxis, ...]

            item["image"] = data[self.npz_image_key][np.newaxis, ...]

        else:
            raise ValueError("Invalid file type")

        # use the transform corresponding to this item's label
        if self.transforms:
            item = self.transforms(item)

        for key, x in item.items():
            if not isinstance(x, torch.Tensor):
                logger.error(
                    f"Transforms did not return a torch tensor for key {key}, but {type(x)}"
                )

        return item

    def __len__(self):
        return len(self.filepaths)
