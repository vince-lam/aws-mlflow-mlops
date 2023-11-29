import logging
import os
import typing
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")

project_name = "aws_mlflow_mlops"

files_to_make = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    "main.py",
    "app.py",
    "Dockerfile",
    "notebooks/experiments.ipynb",
    "templates/index.html",
]


def create_files(files_to_make: typing.List[str]) -> None:
    """
    Creates directories and files based on the provided list of file paths.

    Parameters
    ----------
    files_to_make : List[str]
        A list of file paths to create. Each file path can include directories and a file name.
        Directories are created if they don't exist. Files are created as empty files if they
        don't exist or if they exist but are empty.

    Returns
    -------
    None
    """
    for filepath_str in files_to_make:
        filepath = Path(filepath_str)
        filedir, filename = os.path.split(filepath)

        if filedir != "":
            os.makedirs(filedir, exist_ok=True)
            logging.info(f"Creating directory: {filedir} for file: {filename}")

        if not filepath.exists() or filepath.stat().st_size == 0:
            with open(filepath, "w"):
                pass
            logging.info(f"Creating empty file: {filepath}")
        else:
            logging.info(f"File already exists: {filepath}")


if __name__ == "__main__":
    create_files(files_to_make)
