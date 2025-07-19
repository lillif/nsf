import numpy as np

# default split configuration for the datamodule
SPLITS_DICT = {
    "train": {
        "years": [2024],
        "months": np.arange(1, 13).tolist(),
        "days": np.arange(1, 16).tolist(),
    },
    "val": {
        "years": [2024],
        "months": np.arange(1, 13).tolist(),
        "days": np.arange(20, 25).tolist(),
    },
    "test": {
        "years": [2024],
        "months": np.arange(1, 13).tolist(),
        "days": np.arange(27, 32).tolist(),
    },
}
