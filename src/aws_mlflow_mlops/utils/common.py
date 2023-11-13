import json
import os
from pathlib import Path
from typing import Any

import joblib
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations

from src.aws_mlflow_mlops import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a yaml file and returns a ConfigBox object.

    Args:
        path_to_yaml (Path): Path to yaml file.

    Raises:
        FileNotFoundError: If the yaml file is not found.
        BoxValueError: If the yaml file is not valid.

    Returns:
        ConfigBox: ConfigBox object.
    """
    try:
        with open(path_to_yaml, "r") as yaml_file:
            yaml_config = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
        return ConfigBox(yaml_config)
    except FileNotFoundError as file_error:
        logger.error(f"FileNotFoundError: {file_error}")
        raise file_error
    except BoxValueError as box_error:
        logger.error(f"BoxValueError: {box_error}")
        raise box_error
    except Exception as error:
        logger.error(f"Unexpected error while reading yaml file: {path_to_yaml}")
        raise error


@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True) -> None:
    """
    Create directories from a list of directory paths.

    This function iterates over a list of directory paths and creates each directory.
    If the directory already exists, it is not created again. This function can optionally
    log a message for each directory it creates.

    Parameters:
    path_to_directories (list): A list of strings, where each string is a path to a directory
                                that needs to be created.
    verbose (bool): A flag that indicates whether to log the creation of directories.
                    Defaults to True, meaning that logging is enabled.

    Returns:
    None
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Directory created: {path}")


@ensure_annotations
def save_json(path: Path, data: dict) -> None:
    """
    Save a dictionary to a JSON file.

    This function takes a dictionary and saves it to the specified path as a JSON file.
    The JSON file is formatted with an indentation of 4 spaces for readability. If the
    function successfully saves the file, it logs an informational message with the file's path.

    Parameters:
    path (Path): A Path object representing the file path where the JSON data will be saved.
    data (dict): The dictionary data to be saved to a JSON file.

    Returns:
    None
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"JSON file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Load a JSON file and return its content as a ConfigBox object.

    This function opens a JSON file, reads its content, and converts it into a ConfigBox object,
    which allows attribute-style access to the dictionary values. After loading the file, it logs
    a message with the file's path.

    Parameters:
    path (Path): A Path object representing the file path from where the JSON data will be loaded.

    Returns:
    ConfigBox: A ConfigBox object containing the data loaded from the JSON file.
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"JSON file loaded from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path) -> None:
    """
    Save data to a binary file using joblib.

    This function serializes any Python object ('data') and saves it to a specified path
    using joblib's dump function, which is particularly useful for saving large numpy arrays
    or models efficiently. It also logs an informational message indicating where the file
    has been saved.

    Parameters:
    data (Any): The data to be saved. This can be any Python object serializable by joblib.
    path (Path): A Path object representing the file path where the binary data will be saved.

    Returns:
    None
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Load data from a binary file using joblib.

    This function deserializes data from a binary file located at the given path using joblib's
    load function. It can deserialize any object saved with joblib's dump function. After loading
    the data, it logs an informational message indicating the source of the data.

    Parameters:
    path (Path): A Path object representing the file path from where the binary data will be loaded.

    Returns:
    Any: The deserialized Python object from the binary file.
    """
    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data


def get_size(path: Path) -> str:
    """
    Calculate the size of a file in kilobytes.

    This function takes the path to a file and calculates its size in kilobytes (KB),
    rounded to the nearest kilobyte. It returns the size as a string.

    Parameters:
    path (Path): A Path object representing the file path whose size is to be calculated.

    Returns:
    str: A string representing the file size in kilobytes, formatted as '~ XX KB'.
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"
