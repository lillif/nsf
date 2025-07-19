import json
from typing import Optional

import numpy as np
import torch
import torchvision.transforms
import torchvision.transforms.functional as F
from loguru import logger


## --- 1 Transforms operating on data dictionaries --- ##


class MeanStdNormaliseTransform:
    def __init__(self, normalise_info_path: str, key: str = "image"):
        with open(normalise_info_path) as f:
            normalise_dict = json.load(f)
        self.mean = normalise_dict["mean"]
        self.std = normalise_dict["std"]
        self.key = key

    def __call__(self, data_dict):
        data = data_dict[self.key]
        # normalise data
        data = (data - self.mean) / self.std
        # update dictionary
        data_dict[self.key] = data
        # logger.debug(f"in meanstdnorm, data has x many nan: {np.isnan(data).sum()}")
        return data_dict


class MinMaxNormaliseTransform:
    def __init__(
        self,
        minmax_path: str,
        clip: bool = True,
        scale_mean_std: Optional[tuple[float, float]] = None,
        key: str = "image",
    ):
        with open(minmax_path) as f:
            minmax_dict = json.load(f)
        self.min = minmax_dict["min"]
        self.max = minmax_dict["max"]
        self.clip = clip
        if scale_mean_std is not None:
            self.rescale = True
            self.mean = scale_mean_std[0]
            self.std = scale_mean_std[1]
        else:
            self.rescale = False

        self.key = key

    def __call__(self, data_dict):
        data = data_dict[self.key]

        # standardise data
        data = (data - self.min) / (self.max - self.min)

        if self.clip:
            # clip values to [0, 1]
            data = np.clip(data, 0, 1)

        if self.rescale:
            # normalise data (scale to [-1, 1])
            data = (data - self.mean) / self.std

        # update dictionary
        data_dict[self.key] = data

        return data_dict


class NanMeanFillTransform:
    """
    Replaces NaN values in a numpy array with the mean of the non-NaN values of that patch
    """

    def __init__(self, key="image"):
        self.key = key

    def __call__(self, data_dict):
        data = data_dict[self.key]
        # replace NaN values

        if torch.isnan(data).any():
            data = torch.nan_to_num(data, nan=torch.nanmean(data))
            # update dictionary
            data_dict[self.key] = data
        # logger.debug(f"in nanmeanfill, data has x many nan: {np.isnan(data).sum()}")

        return data_dict


## 2 --- Turn dict of numpy arrays to dict of torch tensors --- ##


class DictNumpyToTensorTransform:
    """
    Convert numpy array to PyTorch tensor
    """

    def __init__(self, dtype=torch.float32, keys: list[str] = ["image", "lat", "lon"]):
        self.dtype = dtype
        self.keys = keys

    def __call__(self, data_dict, **kwargs):
        for key in self.keys:
            data = data_dict[key]
            # Convert to tensor
            tensor = torch.as_tensor(data, dtype=self.dtype)
            data_dict[key] = tensor

            # logger.debug(f"in dictnumpytotensor, data has x many nan: {np.isnan(data).sum()}")

        return data_dict


## 3 --- Transforms operating on dict of torch tensors --- ##


class CopyChannelsTransform:
    """
    Copy the channels of the input image to the specified number of channels.

    Args:
        channels (int): Number of channels to copy the image to
    """

    def __init__(self, channels: int, key: str = "image"):
        self.channels = channels
        self.key = key

    def __call__(self, data_dict):
        """
        data (torch.Tensor): Input image tensor of shape (H, W) -> (C, H, W)
        """
        data = data_dict[self.key]
        # copy channels - TODO does this work for numpy??
        # logger.info(f"data shape before copying channels: {data.shape}")
        data = data.repeat(self.channels, 1, 1)

        # logger.info(f"data shape after copying channels: {data.shape}")
        # update dictionary
        data_dict[self.key] = data
        # logger.debug(f"in copychannels, data has x many nan: {np.isnan(data).sum()}")

        return data_dict


## --- 4.1 Cropping transforms --- ##


class CropTensorTransform:
    """
    Crop the input tensor to the specified size.
    Discards the bottom and right parts of the image.
    """

    def __init__(self, size: int, keys: list[str] = ["image", "lat", "lon"]):
        self.size = size
        self.keys = keys

    def __call__(self, data_dict):
        for key in self.keys:
            data = data_dict[key]
            # crop tensor
            assert (
                len(data.shape) == 3
            ), f"Input tensor must have shape (N, H, W) but is {data.shape}"  # N is number of inputs, e.g. 3: 1 (OLR) + 2 (Coords)
            if data.shape[1] < self.size or data.shape[2] < self.size:
                msg = "Image is too small to crop to the specified size"
                msg += f" (image shape: {data.shape}, crop size: {self.size})"
                raise ValueError(msg)
            # crop tensor
            data = data[:, : self.size, : self.size]  # crops all inputs

            # update dictionary
            data_dict[key] = data
        return data_dict


class RandomCropTensorTransform:
    """
    Randomly crop the input image to the specified size.
    """

    def __init__(self, size: int, keys: list[str] = ["image", "lat", "lon"]):
        self.size = size
        self.keys = keys

    def __call__(self, data_dict):
        # perform random cropping on first item in dictionary
        key_0 = self.keys[0]
        data = data_dict[key_0]
        if data.shape[1] < self.size or data.shape[2] < self.size:
            msg = "Image is too small to crop to the specified size"
            msg += f" (image shape: {data.shape}, crop size: {self.size})"
            raise ValueError(msg)

        # find valid crop
        for _ in range(50):
            # print(data.shape)
            x = torch.randint(0, data.shape[1] - self.size, (1,)).item()
            y = torch.randint(0, data.shape[2] - self.size, (1,)).item()
            data = data[:, x : x + self.size, y : y + self.size]

            # needs to not have nans filled before calling this transform for this check to work
            if not torch.isnan(data).any():
                # logger.info(f"Found valid crop at x={x}, y={y}")
                break
            data = data_dict[key_0]

        # check if image was cropped
        if data.shape[1] > self.size or data.shape[2] > self.size:
            logger.warning(
                "Image was not cropped successfully, will crop to the last attempted crop"
            )
            data = data[:, x : x + self.size, y : y + self.size]

        # update dictionary
        data_dict[key_0] = data

        # logger.debug(f"in randomcroptensor, data has x many nan: {np.isnan(data).sum()}")

        # crop other items in dictionary using the same indices
        if len(self.keys) > 1:
            for key in self.keys[1:]:
                data = data_dict[key]
                data = data[:, x : x + self.size, y : y + self.size]
                data_dict[key] = data
        return data_dict


class ResizeTensorTransform:
    def __init__(self, size=224, keys: list[str] = ["image", "lat", "lon"]):
        """
        Args:
            size: Desired output size of the image.
            keys: List of keys in the dictionary to resize.
        """
        self.size = (size, size)
        self.keys = keys

    def __call__(self, data_dict):
        """
        Args:
            data_dict (dict): Dictionary containing the data to be resized.
        Returns:
            Updated dictionary with resized data.
        """
        for key in self.keys:
            data = data_dict[key]
            # rotate tensor by k * 90 degrees
            data = F.resize(
                data,
                self.size,
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            )
            # update dictionary
            data_dict[key] = data
        return data_dict


## 4.2 --- data augmentation transforms --- ##


class RandomRotate90TensorTransform:
    """
    Randomly rotate the input tensor by a multiple of 90 degrees.
    TODO update for N x H x W tensors
    """

    def __init__(self, keys: list[str] = ["image", "lat", "lon"]):
        self.keys = keys

    def __call__(self, data_dict):
        # choose random rotation once, then apply same rotation to all items in dictionary
        k = torch.randint(0, 4, (1,)).item()
        for key in self.keys:
            data = data_dict[key]
            # rotate tensor by k * 90 degrees
            data = torch.rot90(data, k=k, dims=(1, 2))
            # update dictionary
            data_dict[key] = data
        return data_dict


class RandomFlipTensorTransform:
    """
    Randomly flip the input tensor along the x and/or y axis.
    TODO test for N x H x W tensors
    """

    def __init__(self, keys: list[str] = ["image", "lat", "lon"]):
        self.keys = keys

    def __call__(self, data_dict):
        # flip tensor
        dim1flip = True if torch.rand(1).item() > 0.5 else False
        dim2flip = True if torch.rand(1).item() > 0.5 else False

        for key in self.keys:
            data = data_dict[key]
            # flip tensor
            if dim1flip:
                data = torch.flip(data, [1])
            if dim2flip:
                data = torch.flip(data, [2])
            # update dictionary
            data_dict[key] = data
        return data_dict


# class LogitTransform:
#     def __init__(
#         self,
#         alpha: float = 0.05, ## used 0.05 in realnvp paper... 1e-6 suggested by chatgpt
#         key: str = "image",
#     ):
#         self.alpha = alpha
#         self.key = key

#     def __call__(self, data_dict):
#         data = data_dict[self.key]
#         # check that data is in [0, 1]
#         if not np.all((data >= 0) & (data <= 1)):
#             logger.info(
#                 f"Data is not in the range [0, 1]. Min: {data.min()}, Max: {data.max()}"
#             )
#             logger.info(data)
#             logger.info(np.nanmin(data), np.nanmax(data))
#             raise ValueError("Data must be in the range [0, 1] for logit transform.")

#         # apply logit transform
#         data = torch.logit(self.alpha + (1 - 2 * self.alpha) * data)

#         # update dictionary
#         data_dict[self.key] = data

#         return data_dict


class LogitTransform:
    def __init__(self, alpha: float = 0.05, key: str = "image"):
        self.alpha = alpha
        self.key = key

    def __call__(self, data_dict):
        data = data_dict[self.key]
        if not torch.all((data >= 0) & (data <= 1)):
            logger.error(
                f"Data is not in the range [0, 1]. Min: {data.min()}, Max: {data.max()}"
            )
            logger.error("clipping data to [0, 1]")
            # clip data to [0, 1]
            data = torch.clamp(data, 0, 1)

        # apply logit transform
        data = torch.logit(self.alpha + (1 - 2 * self.alpha) * data)

        # update dictionary
        data_dict[self.key] = data

        return data_dict
