from experiments.images_olr import Flow
from olr_data import OlrDataModule, OlrTransform
from utils import find_hydra_run_path, load_ckpt_from_hydra_run
from constants import SPLITS_DICT
import lightning as pl


transform = OlrTransform(
    normalise_method="minmax",
    normalise_path="/work/bb1153/b382145/computer_vision/data/healpix/minmax_tropics_first15dayseachmonth_5hourly_all3datasets.json",
    fill_nan=True,
    crop_size=64,
    random_crop=True,
    minmax_rescale=False, # keep in 0 to 1 range
    load_coords=False,
    logit=False,#True,
    # logit_alpha=0.05,
)


dataset_kwargs = {
    "transforms": transform,
    "filetype": "npz",
    "load_coords": False,
    "npz_image_key": "image",
}

dataloader = OlrDataModule(
    data_dir = "/work/bb1153/b382145/computer_vision/data/healpix/goes/whole_region/",
    splits_dict = SPLITS_DICT,
    dataset_kwargs = dataset_kwargs,
    filetype = "npz",
    batch_size = 4,
    num_workers = 1,
    pin_memory = False,
    prefetch_factor = 2,
)


model = Flow()


trainer = pl.Trainer(
    max_epochs=1,
    limit_train_batches=5,
    limit_val_batches=2,
    limit_test_batches=2,
)

trainer.fit(model, dataloader)