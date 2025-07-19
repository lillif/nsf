from typing import Optional

from torchvision.transforms import Compose

## transforms operating on data dict (torch)
## turn data dict (numpy) into data dict (torch)
## transforms operating on data dict (numpy)
from .transforms import (
    CopyChannelsTransform,
    CropTensorTransform,
    DictNumpyToTensorTransform,
    LogitTransform,
    MinMaxNormaliseTransform,
    NanMeanFillTransform,
    RandomCropTensorTransform,
    RandomFlipTensorTransform,
    RandomRotate90TensorTransform,
    ResizeTensorTransform,
)

## TODO add image_key and allowed_keys as input and pass on to relevant transforms


class OlrTransform:

    """Transform pipeline for GOES OLR files."""

    def __init__(
        self,
        normalise_method: Optional[str] = None,
        normalise_path: Optional[str] = None,
        fill_nan: bool = True,
        crop_size: Optional[int] = None,
        copy_channels: Optional[int] = None,
        random_rotate: bool = False,
        random_flip: bool = False,
        random_crop: bool = False,
        resize_size: Optional[int] = None,
        minmax_rescale: bool = False,
        load_coords: bool = False,
        logit: bool = False,
        logit_alpha: float = 0.05,
    ):
        if load_coords:
            all_keys = ["image", "lat", "lon"]
        else:
            all_keys = ["image"]
        transforms_list = []

        ## 1 - Normalise
        if normalise_method == "minmax":
            transforms_list.append(
                MinMaxNormaliseTransform(
                    minmax_path=normalise_path,
                    clip=True,
                    scale_mean_std=(0.5, 0.5) if minmax_rescale else None,
                )
            )
        elif normalise_method is not None:
            raise ValueError(f"Unknown normalisation method: {normalise_method}")

        ## 2 - Convert numpy array dict items to torch tensors
        transforms_list.append(DictNumpyToTensorTransform(keys=all_keys))

        ## 3 - Apply data augmentation transforms
        if random_crop:
            assert crop_size is not None
            transforms_list.append(
                RandomCropTensorTransform(size=crop_size, keys=all_keys)
            )
        elif crop_size is not None:
            transforms_list.append(CropTensorTransform(size=crop_size, keys=all_keys))

        if resize_size is not None:
            transforms_list.append(
                ResizeTensorTransform(size=resize_size, keys=all_keys)
            )

        if random_rotate:
            transforms_list.append(RandomRotate90TensorTransform(keys=all_keys))

        if random_flip:
            transforms_list.append(RandomFlipTensorTransform(keys=all_keys))

        ## 4 - Fill nans in the image
        if fill_nan:
            transforms_list.append(NanMeanFillTransform())

        ## 5 - Logit transform
        if logit:
            transforms_list.append(LogitTransform(alpha=logit_alpha))

        ## 6 - Copy image channels
        if copy_channels is not None:
            transforms_list.append(CopyChannelsTransform(channels=copy_channels))

        self.transform = Compose(transforms_list)

    def __call__(self, sample):
        s = self.transform(sample)
        return s
