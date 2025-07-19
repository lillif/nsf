import glob
from datetime import datetime

import pandas as pd
from loguru import logger

def get_list_olrfiles(
    data_path: str,
    ext: str = "nc",
) -> list[str]:
    """
    Get list of olr files with a certain extension in a directory
    """
    file_list = glob.glob(f"{data_path}/*_OLR*{ext}")
    return sorted(file_list)



def get_dates_from_files(filenames: list[str]) -> list[datetime]:
    """
    Extract dates from a list of filenames.

    Args:
        filenames (List[str]): A list of filenames.

    Returns:
        List[str]: A list of dates extracted from the filenames.
    """

    dates = [
        datetime.strptime(filename.split("/")[-1].split("_")[0], "%Y%m%dT%H%M%S")
        for filename in filenames
    ]
    return dates


def get_split(files: list, split_dict: dict) -> tuple[list, list]:
    """
    Split files based on dataset specification.

    Args:
        files (List): A list of files to be split.
        split_dict (DictConfig): A dictionary-like object containing the dataset specification.

    Returns:
        Tuple[List, List]: A tuple containing two lists: the training set and the validation set.
    """
    # Extract dates from filenames
    filenames = [file.split("/")[-1] for file in files]
    dates = get_dates_from_files(filenames)
    # Convert to dataframe for easier manipulation
    df = pd.DataFrame({"filename": filenames, "files": files, "date": dates})

    # Check if years, months, and days are specified
    if "years" not in split_dict.keys() or split_dict["years"] is None:
        logger.info("No years specified for split. Using all years.")
        split_dict["years"] = df.date.dt.year.unique().tolist()
    if "months" not in split_dict.keys() or split_dict["months"] is None:
        logger.info("No months specified for split. Using all months.")
        split_dict["months"] = df.date.dt.month.unique().tolist()
    if "days" not in split_dict.keys() or split_dict["days"] is None:
        logger.info("No days specified for split. Using all days.")
        split_dict["days"] = df.date.dt.day.unique().tolist()

    # Determine conditions specified split
    condition = (
        (df.date.dt.year.isin(split_dict["years"]))
        & (df.date.dt.month.isin(split_dict["months"]))
        & (df.date.dt.day.isin(split_dict["days"]))
    )

    # Extract filenames based on conditions
    split_files = df[condition].files.tolist()

    # Check if files are allocated properly
    if len(split_files) == 0:
        raise ValueError("No files found. Check split specification.")

    # Sort files
    split_files.sort()

    return split_files
